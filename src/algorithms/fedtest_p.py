import os
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset

from .utils import (
    BaseParams,
    BaseServer,
    check_losses,
    clone_cpu_state,
    extract_prototypes,
    fmt_num,
    get_model,
    prepare_input_batch,
    proto_aggregate,
)
from .utils.augment import strong_augment, weak_augment
from .utils.ssl import build_fixmatch_loaders, iterate_ssl_batches


def get_path(args):
    args.file_name = (
        f"{args.common_name}"
        f"_{fmt_num(args.lambda_s)}"
        f"_{fmt_num(args.lambda_p)}"
        f"_{fmt_num(args.lambda_u)}"
        f"_{fmt_num(args.tau)}"
    )
    return os.path.join(args.log_path, f"{args.file_name}_{args.cur_time}.log")


@dataclass
class Params(BaseParams):
    unlabeled_ratio: int
    global_protos: torch.Tensor
    global_valid: torch.Tensor
    global_radii: torch.Tensor
    radius_valid: torch.Tensor
    lambda_p: float
    lambda_s: float
    lambda_u: float
    tau: float


@dataclass
class RadiiParams(BaseParams):
    global_protos: torch.Tensor
    global_valid: torch.Tensor
    proto_radius_quantile: float = 0.95


def mse_distance(features, prototypes):
    """返回样本与原型之间逐特征维平均的平方欧氏距离。"""
    return (features[:, None, :] - prototypes[None, :, :]).square().mean(dim=2)


@torch.no_grad()
def estimate_prototypes_worker(p: BaseParams):
    """客户端使用当前轮次最新的全局模型，在有标签数据上提取类别原型。"""
    device = torch.device(p.client_gpu)

    model = get_model(
        p.model_name,
        p.dataset,
        p.num_class,
        p.feature_dim,
    ).to(device)

    model.load_state_dict(p.model_state)
    model.eval()

    labeled_indices = torch.where(p.train_set.is_labeled.bool())[0].tolist()

    labeled_x = prepare_input_batch(p.train_set.x[labeled_indices], p.dataset)
    labeled_y = p.train_set.y[labeled_indices]
    loader = DataLoader(
        TensorDataset(labeled_x, labeled_y),
        batch_size=p.batch_size,
        shuffle=False,
    )

    prototypes, class_count = extract_prototypes(
        model,
        loader,
        p.num_class,
        p.feature_dim,
        device,
        return_counts=True,
    )

    return {
        "protos": prototypes,
        "class_count": class_count,
    }


@torch.no_grad()
def estimate_radii_worker(p: RadiiParams):
    """客户端使用当前轮次最新的全局模型与全局原型，重估类别半径。"""
    device = torch.device(p.client_gpu)
    model = get_model(p.model_name, p.dataset, p.num_class, p.feature_dim).to(device)
    model.load_state_dict(p.model_state)
    model.eval()

    num_class = p.num_class
    prototypes = p.global_protos.to(device)
    valid = p.global_valid.to(device).bool()

    labeled_indices = torch.where(p.train_set.is_labeled.bool())[0].tolist()
    loader = DataLoader(
        Subset(p.train_set, labeled_indices),
        batch_size=p.batch_size,
        shuffle=False,
    )

    distance_by_class = [[] for _ in range(num_class)]

    for x, y, *_ in loader:
        x, y = x.to(device), y.to(device)
        x = prepare_input_batch(x, p.dataset)

        features = model.extractor(x)
        distance = mse_distance(features, prototypes)
        distance[:, ~valid] = float("inf")

        for i in range(x.size(0)):
            class_id = int(y[i])
            if bool(valid[class_id]):
                distance_by_class[class_id].append(distance[i, class_id])

    radii = torch.zeros(num_class, device=device)
    radius_count = torch.zeros(
        num_class,
        dtype=torch.long,
        device=device,
    )

    for class_id in range(num_class):
        if not distance_by_class[class_id]:
            continue

        distances = torch.stack(distance_by_class[class_id])
        radii[class_id] = torch.quantile(
            distances,
            p.proto_radius_quantile,
        )
        radius_count[class_id] = distances.numel()

    return {
        "radii": radii.cpu().detach().clone(),
        "radius_count": radius_count.cpu().detach().clone(),
    }


@torch.no_grad()
def build_candidate_mask(features, prototypes, radii, valid):
    """按全局 MSE 半径构建无标签样本的候选类别集合。"""
    distance = mse_distance(features, prototypes)
    valid = valid.bool()
    candidate_mask = (distance <= radii.unsqueeze(0)) & valid.unsqueeze(0)
    candidate_size = candidate_mask.sum(dim=1)
    return candidate_mask, candidate_size, distance


def candidate_prototype_contrastive_loss(
    features,
    prototypes,
    global_valid,
    candidate_mask,
    candidate_distance,
    tau,
):
    """以候选原型加权和为正样本、其余有效原型为负样本的对比损失。"""
    usable = candidate_mask.any(dim=1)
    if not bool(usable.any()):
        return features.sum() * 0.0

    features = features[usable]
    mask = candidate_mask[usable]
    distance = candidate_distance[usable]
    inverse_distance = mask.float() / distance.clamp_min(1e-12)
    weights = inverse_distance / inverse_distance.sum(dim=1, keepdim=True)
    positive_prototypes = weights @ prototypes
    positive_logits = -(features - positive_prototypes).square().mean(dim=1)
    positive_logits = positive_logits / tau

    negative_mask = global_valid.unsqueeze(0) & ~mask
    negative_logits = -mse_distance(features, prototypes) / tau
    negative_logits = negative_logits.masked_fill(~negative_mask, float("-inf"))
    logits = torch.cat((positive_logits.unsqueeze(1), negative_logits), dim=1)
    return (torch.logsumexp(logits, dim=1) - positive_logits).mean()


def singleton_pseudo_label_loss(
    features, classifier, candidate_mask, candidate_size, lambda_u: float = 0.0
):
    """仅对唯一候选类别计算硬伪标签交叉熵。当 lambda_u 为 0 时直接跳过分类器前向与损失计算。"""
    singleton_mask = candidate_size.eq(1)
    pseudo_labels = candidate_mask.to(dtype=torch.int64).argmax(dim=1)
    if lambda_u <= 0.0 or not bool(singleton_mask.any()):
        return features.sum() * 0.0, singleton_mask, pseudo_labels
    logits = classifier(features[singleton_mask])
    loss = F.cross_entropy(logits, pseudo_labels[singleton_mask])
    return loss, singleton_mask, pseudo_labels


def train(p: Params):
    device = torch.device(p.client_gpu)
    model_l = get_model(p.model_name, p.dataset, p.num_class, p.feature_dim).to(device)
    model_l.load_state_dict(p.model_state)
    has_unlabeled_loss = (p.lambda_u > 0.0) or (p.lambda_p > 0.0)
    if has_unlabeled_loss:
        model_g = get_model(p.model_name, p.dataset, p.num_class, p.feature_dim).to(
            device
        )
        model_g.load_state_dict(p.model_state)
        model_g.eval()
        for parameter in model_g.parameters():
            parameter.requires_grad_(False)
    else:
        model_g = None

    loaders = build_fixmatch_loaders(
        p.train_set,
        p.batch_size,
        p.unlabeled_ratio,
    )
    if loaders.labeled_loader is None:
        raise ValueError("fedtest 客户端缺少有标签样本")
    use_contrastive = bool((p.global_valid & p.radius_valid).any())
    global_protos = p.global_protos.to(device)
    global_valid = p.global_valid.to(device).bool()
    candidate_valid = global_valid & p.radius_valid.to(device).bool()
    global_radii = p.global_radii.to(device)
    optimizer = torch.optim.SGD(
        model_l.parameters(), lr=p.lr, momentum=p.momentum, weight_decay=p.weight_decay
    )
    loss_sum, sample_count = 0.0, 0
    calibrate_loss_sum, calibrate_loss_count = 0.0, 0
    loss_u_sum, loss_u_count = 0.0, 0
    loss_contrast_sum, loss_contrast_count = 0.0, 0
    total_loss_sum, total_loss_count = 0.0, 0
    candidate_stats = {
        "candidate_total": 0,
        "candidate_empty": 0,
        "candidate_single": 0,
        "candidate_multi": 0,
        "candidate_size_sum": 0,
        "candidate_recall": 0,
        "singleton_correct": 0,
        "singleton_used_count": 0,
        "singleton_pseudo_label_correct": 0,
    }
    model_l.train()
    for _ in range(p.epochs):
        for labeled_batch, unlabeled_batch in iterate_ssl_batches(loaders):
            assert labeled_batch is not None
            x_l_raw, y_l = labeled_batch
            x_u_raw = unlabeled_batch[0]
            x_l_raw, y_l = x_l_raw.to(device), y_l.to(device)
            x_u_raw = x_u_raw.to(device)

            # 仅反向传播的有标签分支使用弱增强。
            x_l_weak = weak_augment(x_l_raw, p.dataset)
            features_l = model_l.extractor(x_l_weak)
            loss_x = F.cross_entropy(model_l.classifier(features_l), y_l)
            loss = loss_x
            valid_labeled = global_valid[y_l]
            if bool(valid_labeled.any()):
                loss_cal = F.mse_loss(
                    features_l[valid_labeled],
                    global_protos[y_l[valid_labeled]],
                )
                valid_count = int(valid_labeled.sum())
                calibrate_loss_sum += loss_cal.item() * valid_count
                calibrate_loss_count += valid_count
            else:
                loss_cal = loss_x * 0.0
            loss = loss + p.lambda_s * loss_cal

            loss_u = loss_x * 0.0
            loss_contrast = loss_x * 0.0
            singleton_mask = torch.zeros(
                x_u_raw.size(0), dtype=torch.bool, device=device
            )
            pseudo_labels = torch.zeros(
                x_u_raw.size(0), dtype=torch.long, device=device
            )
            candidate_mask = torch.zeros(
                (x_u_raw.size(0), p.num_class), dtype=torch.bool, device=device
            )
            candidate_size = torch.zeros(
                x_u_raw.size(0), dtype=torch.long, device=device
            )
            if has_unlabeled_loss and use_contrastive:
                # 候选集合及反距离权重严格来自冻结 teacher 的原始输入。
                with torch.no_grad():
                    candidate_features = model_g.extractor(
                        prepare_input_batch(x_u_raw, p.dataset)
                    )
                    candidate_mask, candidate_size, candidate_distance = (
                        build_candidate_mask(
                            candidate_features,
                            global_protos,
                            global_radii,
                            candidate_valid,
                        )
                    )

                # 所有无标签反向传播项均使用强增强的 student 特征。
                x_u_strong = strong_augment(x_u_raw, p.dataset)
                features_u = model_l.extractor(x_u_strong)
                loss_u, singleton_mask, pseudo_labels = singleton_pseudo_label_loss(
                    features_u,
                    model_l.classifier,
                    candidate_mask,
                    candidate_size,
                    p.lambda_u,
                )
                loss_contrast = candidate_prototype_contrastive_loss(
                    features_u,
                    global_protos,
                    global_valid,
                    candidate_mask,
                    candidate_distance,
                    p.tau,
                )
                loss = loss + p.lambda_u * loss_u + p.lambda_p * loss_contrast

            singleton_count = int(singleton_mask.sum())
            contrast_count = int(candidate_mask.any(dim=1).sum())
            loss_u_sum += loss_u.item() * singleton_count
            loss_u_count += singleton_count
            loss_contrast_sum += loss_contrast.item() * contrast_count
            loss_contrast_count += contrast_count
            # 真实 y_u 仅在所有训练损失构造完成后的诊断分支中读取。
            with torch.no_grad():
                y_u = unlabeled_batch[1].to(device)
                candidate_stats["candidate_total"] += y_u.numel()
                candidate_stats["candidate_empty"] += int(candidate_size.eq(0).sum())
                candidate_stats["candidate_single"] += int(candidate_size.eq(1).sum())
                candidate_stats["candidate_multi"] += int(candidate_size.gt(1).sum())
                candidate_stats["candidate_size_sum"] += int(candidate_size.sum())
                rows = torch.arange(y_u.size(0), device=device)
                candidate_stats["candidate_recall"] += int(
                    candidate_mask[rows, y_u].sum()
                )
                singleton_pseudo_label_correct = int(
                    (singleton_mask & pseudo_labels.eq(y_u)).sum()
                )
                candidate_stats["singleton_correct"] += singleton_pseudo_label_correct
                candidate_stats["singleton_used_count"] += singleton_count
                candidate_stats["singleton_pseudo_label_correct"] += (
                    singleton_pseudo_label_correct
                )
            optimizer.zero_grad()
            check_losses(loss, locals())
            loss.backward()
            optimizer.step()
            loss_sum += loss_x.item() * y_l.size(0)
            sample_count += y_l.size(0)
            total_loss_sum += loss.item() * y_l.size(0)
            total_loss_count += y_l.size(0)

    labeled_indices = torch.where(p.train_set.is_labeled.bool())[0]
    labeled_x = prepare_input_batch(p.train_set.x[labeled_indices], p.dataset)
    labeled_y = p.train_set.y[labeled_indices]
    labeled_loader = DataLoader(
        TensorDataset(labeled_x, labeled_y),
        batch_size=p.batch_size,
    )
    local_protos, class_count = extract_prototypes(
        model_l,
        labeled_loader,
        p.num_class,
        p.feature_dim,
        device,
        return_counts=True,
    )
    return {
        "state": clone_cpu_state(model_l.state_dict()),
        "loss": total_loss_sum / max(1, total_loss_count),
        "loss_x_sum": loss_sum,
        "loss_x_count": sample_count,
        "loss_calibrate_sum": calibrate_loss_sum,
        "loss_calibrate_count": calibrate_loss_count,
        "loss_u_sum": loss_u_sum,
        "loss_u_count": loss_u_count,
        "loss_contrast_sum": loss_contrast_sum,
        "loss_contrast_count": loss_contrast_count,
        "loss_total_sum": total_loss_sum,
        "loss_total_count": total_loss_count,
        "protos": local_protos,
        "class_count": class_count,
        **candidate_stats,
    }


class Server(BaseServer):
    def __init__(self, args):
        if args.ssl in ("none", "client"):
            raise ValueError("fedtest 要求 sample、double 或 sfd 半监督数据。")
        super().__init__(args, is_ssl=True, pfl=False)
        if any(dataset.is_labeled is None for dataset in self.train_sets.values()):
            raise ValueError("fedtest 要求训练数据提供 is_labeled 字段。")
        self.raw_global_protos = torch.zeros(self.num_class, self.feature_dim)
        self.raw_global_valid = torch.zeros(self.num_class, dtype=torch.bool)
        self.previous_target_protos = None
        self.previous_target_valid = None
        self.proto_anchor_weight = args.proto_anchor_weight
        self.proto_sep_weight = args.proto_sep_weight
        self.proto_opt_lr = args.proto_opt_lr
        self.proto_opt_steps = args.proto_opt_steps
        self.mean_protos = torch.zeros(self.num_class, self.feature_dim)
        self.mean_valid = torch.zeros(self.num_class, dtype=torch.bool)
        self.global_protos = torch.zeros(self.num_class, self.feature_dim)
        self.global_valid = torch.zeros(self.num_class, dtype=torch.bool)
        self.global_radii = torch.zeros(self.num_class)
        self.radius_valid = torch.zeros(self.num_class, dtype=torch.bool)
        self.lambda_p = args.lambda_p
        self.lambda_s = args.lambda_s
        self.lambda_u = args.lambda_u
        self.unlabeled_ratio = args.unlabeled_ratio
        self.tau = args.tau
        self.proto_radius_quantile = args.proto_radius_quantile
        self.loss_proto = []
        self.loss_contrast = []
        self.acc_proto_mean = []
        metric_names = (
            "loss_x",
            "loss_calibrate",
            "loss_u",
            "loss_proto",
            "round_time",
            "acc_proto_mean",
            "unlabeled_classifier_acc",
            "unlabeled_proto_acc",
            "seen_classifier_acc",
            "seen_proto_acc",
            "missing_classifier_acc",
            "missing_proto_acc",
            "missing_sample_count",
            "class_support",
            "candidate_empty_ratio",
            "candidate_single_ratio",
            "candidate_multi_ratio",
            "candidate_avg_size",
            "candidate_set_recall",
            "singleton_accuracy",
            "singleton_used_count",
            "singleton_pseudo_label_accuracy",
            "prototype_drift_by_class",
            "prototype_drift_count_by_class",
            "drift_radius_ratio_by_class",
            "drift_radius_ratio_count_by_class",
            "radius_mean",
            "radius_valid_count",
            "radius_count_by_class",
            "nearest_class_by_class",
            "nearest_inter_distance_by_class",
            "nearest_midpoint_distance_by_class",
            "radius_midpoint_ratio_by_class",
            "nearest_inter_distance_mean",
            "radius_midpoint_ratio_mean",
            "radius_midpoint_ratio_max",
            "radius_midpoint_exceed_count",
            "support_overlap_pair_ratio",
            "support_overlap_ratio_mean",
            "support_overlap_ratio_max",
            "support_overlap_pair_count",
            "support_pair_count",
            "candidate_avg_false_size",
            "candidate_false_assignment_ratio",
            "relative_true_distance_mean",
            "relative_wrong_distance_mean",
            "relative_ratio_mean",
            "relative_ratio_quantiles",
            "relative_true_nearest_ratio",
            "relative_ratio_by_class",
            "relative_true_nearest_ratio_by_class",
            "relative_sample_count_by_class",
            "proto_opt_scale",
            "proto_opt_anchor_loss",
            "proto_opt_sep_loss",
            "proto_target_shift_mean",
            "proto_target_shift_max",
            "raw_nearest_inter_distance_mean",
            "target_nearest_inter_distance_mean",
            "proto_inter_distance_gain",
            "proto_tracking_mse",
        )
        for name in metric_names:
            setattr(self, name, [])

    @staticmethod
    def _percent(value, total):
        return 100.0 * value / max(1, total)

    def _aggregate_global_radii(self, results, selected):
        radii = torch.stack([results[cid]["radii"] for cid in selected])
        counts = torch.stack([results[cid]["radius_count"] for cid in selected])
        valid = counts.gt(0)

        global_radii = torch.zeros(self.num_class)

        for class_id in range(self.num_class):
            class_valid = valid[:, class_id]

            if class_valid.any():
                class_radii = radii[class_valid, class_id]
                global_radii[class_id] = torch.quantile(class_radii, 0.90)

        radius_valid = valid.any(dim=0)

        self.global_radii = global_radii
        self.radius_valid = radius_valid

    def _record_radius_diagnostics(self, results, selected):
        valid_radii = self.global_radii[self.radius_valid]
        radius_mean = float(valid_radii.mean()) if valid_radii.numel() else 0.0
        radius_valid_count = int(self.radius_valid.sum())
        radius_count_by_class = torch.stack(
            [results[cid]["radius_count"] for cid in selected]
        ).sum(dim=0)
        self.radius_mean.append(radius_mean)
        self.radius_valid_count.append(radius_valid_count)
        self.radius_count_by_class.append(radius_count_by_class.tolist())

    def _record_prototype_drift(
        self, results, selected, mean_protos, local_protos, local_counts
    ):
        drift_sum = torch.zeros(self.num_class)
        drift_count = torch.zeros(self.num_class, dtype=torch.long)
        ratio_sum = torch.zeros(self.num_class)
        ratio_count = torch.zeros(self.num_class, dtype=torch.long)
        for cid, protos, counts in zip(selected, local_protos, local_counts):
            valid_classes = counts.gt(0)
            for class_id in torch.where(valid_classes)[0].tolist():
                delta = mse_distance(
                    protos[class_id].unsqueeze(0),
                    mean_protos[class_id].unsqueeze(0),
                ).squeeze()
                drift_sum[class_id] += delta
                drift_count[class_id] += 1
                radius = results[cid]["radii"][class_id]
                radius_count_value = results[cid]["radius_count"][class_id]
                if radius_count_value > 0 and radius > 0:
                    ratio_sum[class_id] += delta / radius
                    ratio_count[class_id] += 1
        drift = drift_sum / drift_count.clamp_min(1)
        ratio = ratio_sum / ratio_count.clamp_min(1)
        self.prototype_drift_by_class.append(drift.tolist())
        self.prototype_drift_count_by_class.append(drift_count.tolist())
        self.drift_radius_ratio_by_class.append(ratio.tolist())
        self.drift_radius_ratio_count_by_class.append(ratio_count.tolist())

    @torch.no_grad()
    def _record_interclass_radius_geometry(self):
        valid = (self.global_valid & self.radius_valid).to(self.device).bool()
        valid_indices = torch.where(valid)[0].tolist()
        n_valid = len(valid_indices)

        nearest_class = [-1] * self.num_class
        nearest_inter = [0.0] * self.num_class
        nearest_mid = [0.0] * self.num_class
        mid_ratio = [0.0] * self.num_class

        if n_valid < 2:
            self.nearest_class_by_class.append(nearest_class)
            self.nearest_inter_distance_by_class.append(nearest_inter)
            self.nearest_midpoint_distance_by_class.append(nearest_mid)
            self.radius_midpoint_ratio_by_class.append(mid_ratio)
            self.nearest_inter_distance_mean.append(0.0)
            self.radius_midpoint_ratio_mean.append(0.0)
            self.radius_midpoint_ratio_max.append(0.0)
            self.radius_midpoint_exceed_count.append(0)
            self.support_overlap_pair_ratio.append(0.0)
            self.support_overlap_ratio_mean.append(0.0)
            self.support_overlap_ratio_max.append(0.0)
            self.support_overlap_pair_count.append(0)
            self.support_pair_count.append(0)
            return

        protos = self.global_protos.to(self.device)
        radii = self.global_radii.to(self.device)

        dist_matrix = mse_distance(protos, protos)
        diag_mask = torch.eye(self.num_class, dtype=torch.bool, device=self.device)
        dist_matrix.masked_fill_(
            diag_mask | ~valid.unsqueeze(0) | ~valid.unsqueeze(1), float("inf")
        )

        exceed_count = 0
        inter_list = []
        mid_list = []
        ratio_list = []

        for c in valid_indices:
            min_dist, min_cls = dist_matrix[c].min(dim=0)
            d_inter = float(min_dist.item())
            cls_idx = int(min_cls.item())
            d_mid = d_inter / 4.0
            r_c = float(radii[c].item())
            rho = r_c / max(1e-12, d_mid)

            nearest_class[c] = cls_idx
            nearest_inter[c] = d_inter
            nearest_mid[c] = d_mid
            mid_ratio[c] = rho

            inter_list.append(d_inter)
            mid_list.append(d_mid)
            ratio_list.append(rho)
            if rho > 1.0:
                exceed_count += 1

        self.nearest_class_by_class.append(nearest_class)
        self.nearest_inter_distance_by_class.append(nearest_inter)
        self.nearest_midpoint_distance_by_class.append(nearest_mid)
        self.radius_midpoint_ratio_by_class.append(mid_ratio)
        self.nearest_inter_distance_mean.append(
            sum(inter_list) / max(1, len(inter_list))
        )
        self.radius_midpoint_ratio_mean.append(
            sum(ratio_list) / max(1, len(ratio_list))
        )
        self.radius_midpoint_ratio_max.append(max(ratio_list) if ratio_list else 0.0)
        self.radius_midpoint_exceed_count.append(exceed_count)

        overlap_ratios = []
        overlap_pair_count = 0
        total_pair_count = 0

        for i in range(n_valid):
            c = valid_indices[i]
            r_c_root = torch.sqrt(radii[c].clamp_min(0.0))
            for j in range(i + 1, n_valid):
                k = valid_indices[j]
                r_k_root = torch.sqrt(radii[k].clamp_min(0.0))
                d_ck_root = torch.sqrt(dist_matrix[c, k].clamp_min(1e-12))
                o_ck = float(((r_c_root + r_k_root) / d_ck_root).item())
                overlap_ratios.append(o_ck)
                total_pair_count += 1
                if o_ck > 1.0:
                    overlap_pair_count += 1

        pair_ratio = (
            (overlap_pair_count / max(1, total_pair_count))
            if total_pair_count > 0
            else 0.0
        )
        mean_overlap = (
            (sum(overlap_ratios) / max(1, len(overlap_ratios)))
            if overlap_ratios
            else 0.0
        )
        max_overlap = max(overlap_ratios) if overlap_ratios else 0.0

        self.support_overlap_pair_ratio.append(pair_ratio)
        self.support_overlap_ratio_mean.append(mean_overlap)
        self.support_overlap_ratio_max.append(max_overlap)
        self.support_overlap_pair_count.append(overlap_pair_count)
        self.support_pair_count.append(total_pair_count)

    def _optimize_global_prototypes(self, raw_protos, valid):
        valid_indices = torch.where(valid)[0]
        n_valid = len(valid_indices)

        if n_valid < 2:
            self.proto_opt_scale.append(0.0)
            self.proto_opt_anchor_loss.append(0.0)
            self.proto_opt_sep_loss.append(0.0)
            self.proto_target_shift_mean.append(0.0)
            self.proto_target_shift_max.append(0.0)
            self.raw_nearest_inter_distance_mean.append(0.0)
            self.target_nearest_inter_distance_mean.append(0.0)
            self.proto_inter_distance_gain.append(1.0)
            return raw_protos.clone()

        raw_valid = raw_protos[valid_indices].detach().to(self.device)
        target_valid = torch.nn.Parameter(raw_valid.clone())

        with torch.no_grad():
            raw_dist = mse_distance(raw_valid, raw_valid)
            pair_mask = torch.triu(
                torch.ones_like(raw_dist, dtype=torch.bool), diagonal=1
            )
            pair_dist = raw_dist[pair_mask]
            proto_scale = pair_dist.median().detach().clamp_min(1e-12)
            diag_mask = torch.eye(n_valid, dtype=torch.bool, device=self.device)
            raw_dist_masked = raw_dist.masked_fill(diag_mask, float("inf"))
            raw_nearest_inter = float(raw_dist_masked.min(dim=1).values.mean().item())

        optimizer = torch.optim.Adam([target_valid], lr=self.proto_opt_lr)

        for _ in range(self.proto_opt_steps):
            optimizer.zero_grad()
            anchor_distance = (target_valid - raw_valid).square().mean(dim=1)
            anchor_loss = (anchor_distance / proto_scale).mean()

            target_dist = mse_distance(target_valid, target_valid)
            target_pair_dist = target_dist[pair_mask]
            sep_loss = torch.exp(-target_pair_dist / proto_scale).mean()

            proto_loss = (
                self.proto_anchor_weight * anchor_loss
                + self.proto_sep_weight * sep_loss
            )
            check_losses(proto_loss, locals())
            proto_loss.backward()
            optimizer.step()

        with torch.no_grad():
            final_anchor_dist = (target_valid - raw_valid).square().mean(dim=1)
            final_anchor_loss = float((final_anchor_dist / proto_scale).mean().item())

            target_dist = mse_distance(target_valid, target_valid)
            target_pair_dist = target_dist[pair_mask]
            final_sep_loss = float(
                torch.exp(-target_pair_dist / proto_scale).mean().item()
            )

            target_dist_masked = target_dist.masked_fill(diag_mask, float("inf"))
            target_nearest_inter = float(
                target_dist_masked.min(dim=1).values.mean().item()
            )
            shift_by_class = (target_valid - raw_valid).square().mean(dim=1)
            shift_mean = float(shift_by_class.mean().item())
            shift_max = float(shift_by_class.max().item())
            inter_gain = target_nearest_inter / max(1e-12, raw_nearest_inter)

        self.proto_opt_scale.append(float(proto_scale.item()))
        self.proto_opt_anchor_loss.append(final_anchor_loss)
        self.proto_opt_sep_loss.append(final_sep_loss)
        self.proto_target_shift_mean.append(shift_mean)
        self.proto_target_shift_max.append(shift_max)
        self.raw_nearest_inter_distance_mean.append(raw_nearest_inter)
        self.target_nearest_inter_distance_mean.append(target_nearest_inter)
        self.proto_inter_distance_gain.append(inter_gain)

        optimized = raw_protos.clone().to(self.device)
        optimized[valid_indices] = target_valid.detach()
        return optimized.cpu()

    @torch.no_grad()
    def _record_relative_proto_geometry(self, selected):
        self.model.to(self.device).eval()
        prototypes = self.raw_global_protos.to(self.device)
        valid = self.raw_global_valid.to(self.device).bool()

        if int(valid.sum().item()) < 2:
            self.relative_true_distance_mean.append(0.0)
            self.relative_wrong_distance_mean.append(0.0)
            self.relative_ratio_mean.append(0.0)
            self.relative_ratio_quantiles.append([0.0] * 6)
            self.relative_true_nearest_ratio.append(0.0)
            self.relative_ratio_by_class.append([0.0] * self.num_class)
            self.relative_true_nearest_ratio_by_class.append([0.0] * self.num_class)
            self.relative_sample_count_by_class.append([0] * self.num_class)
            return

        all_true_dist = []
        all_wrong_dist = []
        all_q = []
        all_y = []

        for cid in selected:
            dataset = self.train_sets[cid]
            labeled_indices = torch.where(dataset.is_labeled.bool())[0].tolist()
            if not labeled_indices:
                continue
            loader = DataLoader(
                Subset(dataset, labeled_indices),
                batch_size=128,
                shuffle=False,
            )
            for x, y, *_ in loader:
                x, y = x.to(self.device), y.to(self.device)
                usable = valid[y]
                if not bool(usable.any()):
                    continue
                x_u = prepare_input_batch(x[usable], self.dataset)
                y_u = y[usable]

                feature = self.model.extractor(x_u)
                distance = mse_distance(feature, prototypes)
                distance[:, ~valid] = float("inf")

                true_distance = distance.gather(1, y_u[:, None]).squeeze(1)

                wrong_distance = distance.clone()
                wrong_distance.scatter_(1, y_u[:, None], float("inf"))
                min_wrong_distance = wrong_distance.min(dim=1).values

                relative_ratio = true_distance / min_wrong_distance.clamp_min(1e-12)

                all_true_dist.append(true_distance.cpu())
                all_wrong_dist.append(min_wrong_distance.cpu())
                all_q.append(relative_ratio.cpu())
                all_y.append(y_u.cpu())

        if not all_q:
            self.relative_true_distance_mean.append(0.0)
            self.relative_wrong_distance_mean.append(0.0)
            self.relative_ratio_mean.append(0.0)
            self.relative_ratio_quantiles.append([0.0] * 6)
            self.relative_true_nearest_ratio.append(0.0)
            self.relative_ratio_by_class.append([0.0] * self.num_class)
            self.relative_true_nearest_ratio_by_class.append([0.0] * self.num_class)
            self.relative_sample_count_by_class.append([0] * self.num_class)
            return

        all_true_dist = torch.cat(all_true_dist)
        all_wrong_dist = torch.cat(all_wrong_dist)
        all_q = torch.cat(all_q)
        all_y = torch.cat(all_y)

        probs = torch.tensor([0.10, 0.25, 0.50, 0.75, 0.90, 0.95], dtype=torch.float32)
        q_quantiles = torch.quantile(all_q, probs).tolist()

        self.relative_true_distance_mean.append(float(all_true_dist.mean().item()))
        self.relative_wrong_distance_mean.append(float(all_wrong_dist.mean().item()))
        self.relative_ratio_mean.append(float(all_q.mean().item()))
        self.relative_ratio_quantiles.append(q_quantiles)
        self.relative_true_nearest_ratio.append(
            float((all_q < 1.0).float().mean().item())
        )

        q_by_class = [0.0] * self.num_class
        nearest_by_class = [0.0] * self.num_class
        count_by_class = [0] * self.num_class

        for c in range(self.num_class):
            c_mask = all_y.eq(c)
            c_count = int(c_mask.sum().item())
            count_by_class[c] = c_count
            if c_count > 0:
                q_by_class[c] = float(all_q[c_mask].mean().item())
                nearest_by_class[c] = float((all_q[c_mask] < 1.0).float().mean().item())

        self.relative_ratio_by_class.append(q_by_class)
        self.relative_true_nearest_ratio_by_class.append(nearest_by_class)
        self.relative_sample_count_by_class.append(count_by_class)

    @torch.no_grad()
    def _diagnose_client(self, dataset, labeled_present):
        loader = DataLoader(dataset, batch_size=128, shuffle=False)
        stats = {
            key: 0
            for key in (
                "total",
                "classifier_correct",
                "prototype_correct",
                "seen_total",
                "seen_cls",
                "seen_proto",
                "missing_total",
                "missing_cls",
                "missing_proto",
            )
        }
        self.model.to(self.device).eval()
        prototypes = self.global_protos.to(self.device)
        valid = self.global_valid.to(self.device).bool()
        present = labeled_present.to(self.device).bool()
        for x, y, _, is_labeled in loader:
            mask = ~is_labeled.bool()
            if not bool(mask.any()):
                continue
            x, y = x[mask].to(self.device), y[mask].to(self.device)
            x = prepare_input_batch(x, self.dataset)
            feature = self.model.extractor(x)
            logits = self.model.classifier(feature)
            classifier_pred = logits.argmax(dim=1)
            distance = mse_distance(feature, prototypes)
            distance[:, ~valid] = float("inf")
            proto_pred = distance.argmin(dim=1)
            classifier_ok = classifier_pred == y
            proto_ok = (
                proto_pred == y
                if bool(valid.any())
                else torch.zeros_like(classifier_ok)
            )
            missing, seen = ~present[y], present[y]
            stats["total"] += y.numel()
            stats["classifier_correct"] += int(classifier_ok.sum())
            stats["prototype_correct"] += int(proto_ok.sum())
            stats["seen_total"] += int(seen.sum())
            stats["seen_cls"] += int((seen & classifier_ok).sum())
            stats["seen_proto"] += int((seen & proto_ok).sum())
            stats["missing_total"] += int(missing.sum())
            stats["missing_cls"] += int((missing & classifier_ok).sum())
            stats["missing_proto"] += int((missing & proto_ok).sum())
        return stats

    @torch.no_grad()
    def _evaluate_accuracy(self):
        loader = DataLoader(self.test_set, batch_size=128, shuffle=False)
        model_correct = proto_correct = mean_proto_correct = total = 0
        self.model.to(self.device).eval()
        prototypes = self.global_protos.to(self.device)
        valid = self.global_valid.to(self.device).bool()
        mean_prototypes = self.mean_protos.to(self.device)
        mean_valid = self.mean_valid.to(self.device).bool()
        for x, y, *_ in loader:
            x, y = x.to(self.device), y.to(self.device)
            feature = self.model.extractor(x)
            model_correct += int((self.model.classifier(feature).argmax(1) == y).sum())
            distance = mse_distance(feature, prototypes)
            distance[:, ~valid] = float("inf")
            proto_pred = distance.argmin(1)
            proto_ok = proto_pred == y
            if not bool(valid.any()):
                proto_ok = torch.zeros_like(proto_ok)
            proto_correct += int(proto_ok.sum())
            mean_distance = mse_distance(feature, mean_prototypes)
            mean_distance[:, ~mean_valid] = float("inf")
            mean_proto_pred = mean_distance.argmin(1)
            mean_proto_ok = mean_proto_pred == y
            if not bool(mean_valid.any()):
                mean_proto_ok = torch.zeros_like(mean_proto_ok)
            mean_proto_correct += int(mean_proto_ok.sum())
            total += y.numel()
        self.model.cpu()
        return (
            self._percent(model_correct, total),
            self._percent(mean_proto_correct, total),
            self._percent(proto_correct, total),
        )

    def fit(self):
        num_join = max(1, int(self.num_clients * self.join_ratio))
        for index in range(self.rounds):
            started = time.time()
            selected = sorted(torch.randperm(self.num_clients)[:num_join].tolist())
            print(f"\n--- FedTest 诊断：第 {index + 1}/{self.rounds} 轮 ---")
            params = []
            for base in self.build_base_params(selected):
                params.append(
                    Params(
                        **asdict(base),
                        unlabeled_ratio=self.unlabeled_ratio,
                        global_protos=self.global_protos.clone(),
                        global_valid=self.global_valid.clone(),
                        global_radii=self.global_radii.clone(),
                        radius_valid=self.radius_valid.clone(),
                        lambda_p=self.lambda_p,
                        lambda_s=self.lambda_s,
                        lambda_u=self.lambda_u,
                        tau=self.tau,
                    )
                )
            results = self.run_clients(train, params)

            # 1. 聚合全局模型 (FedAvg)
            states = [results[cid]["state"] for cid in selected]
            weights = [self.weights[cid] for cid in selected]
            weight_sum = sum(weights)
            weights = [weight / weight_sum for weight in weights]
            self.aggregate(states, weights=weights)

            # 2. 客户端使用同一个 G_t 在有标签数据上提取原型
            proto_params = self.build_base_params(selected)
            proto_results = self.run_clients(
                estimate_prototypes_worker,
                proto_params,
            )

            # 3. 聚合得到真正处于 G_t 特征空间中的全局原型
            global_model_protos = [proto_results[cid]["protos"] for cid in selected]
            global_model_counts = [
                proto_results[cid]["class_count"] for cid in selected
            ]

            mean_protos = proto_aggregate(
                global_model_protos,
                global_model_counts,
            )
            total_count = torch.stack(global_model_counts).sum(dim=0)
            mean_valid = total_count.gt(0)
            mean_protos[~mean_valid] = 0.0

            # 严格保留真实统计原型
            self.raw_global_protos = mean_protos.clone()
            self.raw_global_valid = mean_valid.clone()
            self.mean_protos = self.raw_global_protos.clone()
            self.mean_valid = self.raw_global_valid.clone()
            support = total_count

            # 计算上一轮目标原型 -> 本轮统计原型的跟随度 tracking MSE
            if (
                self.previous_target_protos is not None
                and self.previous_target_valid is not None
            ):
                track_valid = self.raw_global_valid & self.previous_target_valid
                if track_valid.any():
                    track_dist = (
                        (
                            self.raw_global_protos[track_valid]
                            - self.previous_target_protos[track_valid]
                        )
                        .square()
                        .mean(dim=1)
                    )
                    self.proto_tracking_mse.append(float(track_dist.mean().item()))
                else:
                    self.proto_tracking_mse.append(0.0)
            else:
                self.proto_tracking_mse.append(0.0)

            # 服务端优化判别目标原型
            self.global_protos = self._optimize_global_prototypes(
                self.raw_global_protos,
                self.raw_global_valid,
            )
            self.global_valid = self.raw_global_valid.clone()
            self.previous_target_protos = self.global_protos.clone()
            self.previous_target_valid = self.global_valid.clone()

            # 4. 客户端使用同一个 G_t 和 p_c^G(t) 计算半径
            radii_params = [
                RadiiParams(
                    **asdict(base),
                    global_protos=self.global_protos.clone(),
                    global_valid=self.global_valid.clone(),
                    proto_radius_quantile=self.proto_radius_quantile,
                )
                for base in self.build_base_params(selected)
            ]
            radii_results = self.run_clients(estimate_radii_worker, radii_params)

            # 5. 服务端用有效客户端半径的 Q_0.90 分位数进行聚合
            self._aggregate_global_radii(radii_results, selected)
            self._record_radius_diagnostics(radii_results, selected)
            self._record_prototype_drift(
                radii_results,
                selected,
                self.global_protos,
                global_model_protos,
                global_model_counts,
            )
            self._record_interclass_radius_geometry()
            self._record_relative_proto_geometry(selected)

            proto_loss = (
                self.proto_anchor_weight * self.proto_opt_anchor_loss[-1]
                + self.proto_sep_weight * self.proto_opt_sep_loss[-1]
            )
            stats = self._empty_stats()
            for cid in selected:
                dataset = self.train_sets[cid]
                labeled = dataset.is_labeled.bool()
                present = (
                    torch.bincount(dataset.y[labeled], minlength=self.num_class) > 0
                )
                current = self._diagnose_client(dataset, present)
                stats = self._merge_stats(stats, current)
            model_acc, mean_proto_acc, proto_acc = self._evaluate_accuracy()
            loss_sum = sum(results[cid]["loss_total_sum"] for cid in selected)
            loss_count = sum(results[cid]["loss_total_count"] for cid in selected)
            round_loss = loss_sum / max(1, loss_count)
            loss_x_sum = sum(results[cid]["loss_x_sum"] for cid in selected)
            loss_x_count = sum(results[cid]["loss_x_count"] for cid in selected)
            supervised_loss = loss_x_sum / max(1, loss_x_count)
            calibrate_sum = sum(results[cid]["loss_calibrate_sum"] for cid in selected)
            calibrate_count = sum(
                results[cid]["loss_calibrate_count"] for cid in selected
            )
            calibrate_loss = calibrate_sum / max(1, calibrate_count)
            loss_u_sum = sum(results[cid]["loss_u_sum"] for cid in selected)
            loss_u_count = sum(results[cid]["loss_u_count"] for cid in selected)
            unsupervised_loss = loss_u_sum / max(1, loss_u_count)
            loss_contrast_sum = sum(
                results[cid]["loss_contrast_sum"] for cid in selected
            )
            loss_contrast_count = sum(
                results[cid]["loss_contrast_count"] for cid in selected
            )
            contrastive_loss = loss_contrast_sum / max(1, loss_contrast_count)
            self._record_candidate_diagnostics(results, selected)
            self._record_and_print(
                stats,
                support,
                model_acc,
                mean_proto_acc,
                proto_acc,
                proto_loss,
                round_loss,
                supervised_loss,
                calibrate_loss,
                unsupervised_loss,
                contrastive_loss,
                time.time() - started,
            )
            self._print_candidate_diagnostics(contrastive_loss)

    @staticmethod
    def _empty_stats():
        return {
            key: 0
            for key in (
                "total",
                "classifier_correct",
                "prototype_correct",
                "seen_total",
                "seen_cls",
                "seen_proto",
                "missing_total",
                "missing_cls",
                "missing_proto",
            )
        }

    @staticmethod
    def _merge_stats(left, right):
        for key in left:
            left[key] += right[key]
        return left

    def _record_and_print(
        self,
        stats,
        support,
        model_acc,
        mean_proto_acc,
        proto_acc,
        proto_loss,
        round_loss,
        supervised_loss,
        calibrate_loss,
        unsupervised_loss,
        contrastive_loss,
        elapsed,
    ):
        p = self._percent
        self.acc.append(model_acc)
        self.acc_proto_mean.append(mean_proto_acc)
        self.acc_proto.append(proto_acc)
        self.loss_proto.append(proto_loss)
        self.loss_x.append(supervised_loss)
        self.loss_calibrate.append(calibrate_loss)
        self.loss_u.append(unsupervised_loss)
        self.loss_contrast.append(contrastive_loss)
        self.loss.append(round_loss)

        self.unlabeled_classifier_acc.append(
            p(stats["classifier_correct"], stats["total"])
        )
        self.unlabeled_proto_acc.append(p(stats["prototype_correct"], stats["total"]))
        self.seen_classifier_acc.append(p(stats["seen_cls"], stats["seen_total"]))
        self.seen_proto_acc.append(p(stats["seen_proto"], stats["seen_total"]))
        self.missing_classifier_acc.append(
            p(stats["missing_cls"], stats["missing_total"])
        )
        self.missing_proto_acc.append(p(stats["missing_proto"], stats["missing_total"]))
        self.missing_sample_count.append(stats["missing_total"])
        self.class_support.append(support.int().tolist())
        self.round_time.append(elapsed)

        print(
            "准确率（本轮 FedAvg 后全局模型与两类全局原型，测试集）：\n"
            f"FedAvg 后全局模型分类准确率={model_acc:.2f}%\n"
            f"统计全局原型分类准确率={mean_proto_acc:.2f}%\n"
            f"判别目标全局原型分类准确率={proto_acc:.2f}%"
        )
        print(
            "无标签样本诊断（真实标签仅用于评估）：\n"
            f"分类器准确率={self.unlabeled_classifier_acc[-1]:.2f}%\n"
            f"目标原型准确率={self.unlabeled_proto_acc[-1]:.2f}%\n"
            f"已标注类别：分类器准确率={self.seen_classifier_acc[-1]:.2f}%，"
            f"原型预测准确率={self.seen_proto_acc[-1]:.2f}% (n={stats['seen_total']})\n"
            f"缺失类别：分类器准确率={self.missing_classifier_acc[-1]:.2f}%，"
            f"原型预测准确率={self.missing_proto_acc[-1]:.2f}% (n={stats['missing_total']})"
        )
        print(f"各客户端有标签类别样本数（按类别）：{self.class_support[-1]}")
        print(f"本轮耗时：{elapsed:.2f} 秒")

    def _print_candidate_diagnostics(self, contrastive_loss):
        if self.lambda_p > 0.0 or self.lambda_u > 0.0:
            print(
                "候选集合诊断（比例/召回率为百分比；平均候选类别数单位为类/样本）：\n"
                f"空候选比例={self.candidate_empty_ratio[-1]:.2%}\n"
                f"单候选比例={self.candidate_single_ratio[-1]:.2%}\n"
                f"多候选比例={self.candidate_multi_ratio[-1]:.2%}\n"
                f"平均候选类别数={self.candidate_avg_size[-1]:.4f}\n"
                f"平均错误候选类别数={self.candidate_avg_false_size[-1]:.4f}\n"
                f"候选成员错误比例={self.candidate_false_assignment_ratio[-1]:.2%}\n"
                f"候选集合召回率（分母=全部无标签样本）={self.candidate_set_recall[-1]:.2%}\n"
                f"实际用于单例 CE 的样本数={self.singleton_used_count[-1]}\n"
                "单例伪标签准确率（真实标签仅用于诊断，分母=实际用于 CE 的单例样本）="
                f"{self.singleton_pseudo_label_accuracy[-1]:.2%}"
            )
        print(
            "本地类别半径统计（半径由有效客户端按类别取分位数）：\n"
            f"有效全局半径均值={self.radius_mean[-1]:.6f} (有效类别数={self.radius_valid_count[-1]})"
        )
        drift_by_class = self.prototype_drift_by_class[-1]
        drift_counts = self.prototype_drift_count_by_class[-1]
        ratio_by_class = self.drift_radius_ratio_by_class[-1]
        ratio_counts = self.drift_radius_ratio_count_by_class[-1]

        valid_drift = [d for d, c in zip(drift_by_class, drift_counts) if c > 0]
        mean_drift = sum(valid_drift) / max(1, len(valid_drift)) if valid_drift else 0.0

        valid_ratio = [r for r, c in zip(ratio_by_class, ratio_counts) if c > 0]
        mean_ratio = sum(valid_ratio) / max(1, len(valid_ratio)) if valid_ratio else 0.0

        print(
            "原型漂移与漂移/半径比值统计 d(p_{k,c}, p_c^G) / r_{k,c}：\n"
            f"有效类别平均原型漂移 MSE={mean_drift:.6f}\n"
            f"有效类别平均漂移半径比值={mean_ratio:.4f}"
        )
        print(
            "类间距离与支持域几何统计：\n"
            f"最近异类原型 MSE 均值={self.nearest_inter_distance_mean[-1]:.6f}\n"
            f"半径/最近类间中点比值均值={self.radius_midpoint_ratio_mean[-1]:.4f}\n"
            f"半径/最近类间中点比值最大值={self.radius_midpoint_ratio_max[-1]:.4f}\n"
            f"发生重叠的类别对比例={self.support_overlap_pair_ratio[-1]:.2%}\n"
            f"两球重叠比均值={self.support_overlap_ratio_mean[-1]:.4f}"
        )
        print(
            "全局判别原型优化统计：\n"
            f"本轮原型距离尺度={self.proto_opt_scale[-1]:.6f}\n"
            f"优化后锚定损失={self.proto_opt_anchor_loss[-1]:.6f}\n"
            f"优化后分离损失={self.proto_opt_sep_loss[-1]:.6f}\n"
            f"统计原型最近异类 MSE 均值={self.raw_nearest_inter_distance_mean[-1]:.6f}\n"
            f"目标原型最近异类 MSE 均值={self.target_nearest_inter_distance_mean[-1]:.6f}\n"
            f"类间距离放大倍数={self.proto_inter_distance_gain[-1]:.4f}\n"
            f"目标原型相对统计原型偏移均值={self.proto_target_shift_mean[-1]:.6f}\n"
            f"目标原型相对统计原型偏移最大值={self.proto_target_shift_max[-1]:.6f}\n"
            f"上一轮目标原型→本轮统计原型跟随 MSE={self.proto_tracking_mse[-1]:.6f}"
        )
        q_q = self.relative_ratio_quantiles[-1]
        detail_rel_strs = [
            f"c{c}: true_nearest={self.relative_true_nearest_ratio_by_class[-1][c]:.2%}"
            for c in range(self.num_class)
            if self.relative_sample_count_by_class[-1][c] > 0
        ]
        print(
            "相对原型距离诊断（仅使用有标签样本）：\n"
            f"真实类别原型 MSE 均值={self.relative_true_distance_mean[-1]:.6f}\n"
            f"最近错误类别原型 MSE 均值={self.relative_wrong_distance_mean[-1]:.6f}\n"
            "真实/最近错误距离比 q：\n"
            f"均值={self.relative_ratio_mean[-1]:.4f}\n"
            f"P25={q_q[1]:.4f} P50={q_q[2]:.4f} P75={q_q[3]:.4f} P90={q_q[4]:.4f}\n"
            f"真实类别为最近原型的样本比例={self.relative_true_nearest_ratio[-1]:.2%}\n"
            f"各类别真实为最近原型比例：{' | '.join(detail_rel_strs)}"
        )
        if self.lambda_p > 0.0 or self.lambda_u > 0.0:
            print(
                "损失统计（均为实际参与该项损失的样本均值）：\n"
                f"监督分类 CE={self.loss_x[-1]:.6f}\n"
                f"监督原型校准 MSE={self.loss_calibrate[-1]:.6f}\n"
                f"单例伪标签 CE={self.loss_u[-1]:.6f}\n"
                f"候选集合对比损失（分母=候选集合非空样本）={contrastive_loss:.6f}\n"
                f"总训练损失={self.loss[-1]:.6f}"
            )
        else:
            print(
                "损失统计（均为实际参与该项损失的样本均值）：\n"
                f"监督分类 CE={self.loss_x[-1]:.6f}\n"
                f"监督原型校准 MSE={self.loss_calibrate[-1]:.6f}\n"
                f"总训练损失={self.loss[-1]:.6f}"
            )

    def _record_candidate_diagnostics(self, results, selected):
        total = sum(results[cid]["candidate_total"] for cid in selected)
        empty = sum(results[cid]["candidate_empty"] for cid in selected)
        single = sum(results[cid]["candidate_single"] for cid in selected)
        multi = sum(results[cid]["candidate_multi"] for cid in selected)
        size_sum = sum(results[cid]["candidate_size_sum"] for cid in selected)
        recall = sum(results[cid]["candidate_recall"] for cid in selected)
        singleton_correct = sum(results[cid]["singleton_correct"] for cid in selected)
        singleton_used = sum(results[cid]["singleton_used_count"] for cid in selected)
        singleton_pseudo_label_correct = sum(
            results[cid]["singleton_pseudo_label_correct"] for cid in selected
        )
        false_candidate_count = max(0, size_sum - recall)
        self.candidate_empty_ratio.append(empty / max(1, total))
        self.candidate_single_ratio.append(single / max(1, total))
        self.candidate_multi_ratio.append(multi / max(1, total))
        self.candidate_avg_size.append(size_sum / max(1, total))
        self.candidate_set_recall.append(recall / max(1, total))
        self.candidate_avg_false_size.append(false_candidate_count / max(1, total))
        self.candidate_false_assignment_ratio.append(
            false_candidate_count / max(1, size_sum)
        )
        self.singleton_accuracy.append(singleton_correct / max(1, single))
        self.singleton_used_count.append(singleton_used)
        self.singleton_pseudo_label_accuracy.append(
            singleton_pseudo_label_correct / max(1, singleton_used)
        )

    def save(self):
        metrics = {
            "acc": self.acc,
            "acc_proto": self.acc_proto,
            "acc_proto_mean": self.acc_proto_mean,
            "loss": self.loss,
            "loss_proto": self.loss_proto,
        }
        for name in (
            "loss_x",
            "loss_calibrate",
            "loss_u",
            "loss_contrast",
            "round_time",
            "unlabeled_classifier_acc",
            "unlabeled_proto_acc",
            "seen_classifier_acc",
            "seen_proto_acc",
            "missing_classifier_acc",
            "missing_proto_acc",
            "missing_sample_count",
            "class_support",
            "candidate_empty_ratio",
            "candidate_single_ratio",
            "candidate_multi_ratio",
            "candidate_avg_size",
            "candidate_set_recall",
            "singleton_accuracy",
            "singleton_used_count",
            "singleton_pseudo_label_accuracy",
            "prototype_drift_by_class",
            "prototype_drift_count_by_class",
            "drift_radius_ratio_by_class",
            "drift_radius_ratio_count_by_class",
            "radius_mean",
            "radius_valid_count",
            "radius_count_by_class",
            "nearest_class_by_class",
            "nearest_inter_distance_by_class",
            "nearest_midpoint_distance_by_class",
            "radius_midpoint_ratio_by_class",
            "nearest_inter_distance_mean",
            "radius_midpoint_ratio_mean",
            "radius_midpoint_ratio_max",
            "radius_midpoint_exceed_count",
            "support_overlap_pair_ratio",
            "support_overlap_ratio_mean",
            "support_overlap_ratio_max",
            "support_overlap_pair_count",
            "support_pair_count",
            "candidate_avg_false_size",
            "candidate_false_assignment_ratio",
            "relative_true_distance_mean",
            "relative_wrong_distance_mean",
            "relative_ratio_mean",
            "relative_ratio_quantiles",
            "relative_true_nearest_ratio",
            "relative_ratio_by_class",
            "relative_true_nearest_ratio_by_class",
            "relative_sample_count_by_class",
            "proto_opt_scale",
            "proto_opt_anchor_loss",
            "proto_opt_sep_loss",
            "proto_target_shift_mean",
            "proto_target_shift_max",
            "raw_nearest_inter_distance_mean",
            "target_nearest_inter_distance_mean",
            "proto_inter_distance_gain",
            "proto_tracking_mse",
        ):
            metrics[name] = getattr(self, name)
        self.deal_save(
            metrics,
            {
                "global": self.model.state_dict(),
                "proto": self.global_protos,
                "proto_mean": self.mean_protos,
                "aux": {
                    "global_radii": self.global_radii,
                    "radius_valid": self.radius_valid,
                },
            },
        )
