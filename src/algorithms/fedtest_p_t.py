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
from .utils.ssl import build_fixmatch_loaders, iterate_fixmatch_batches


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
    proto_q_thresholds: torch.Tensor
    proto_q_valid: torch.Tensor
    lambda_p: float
    lambda_s: float
    lambda_u: float
    tau: float


@dataclass
class ProtoGateParams(BaseParams):
    global_protos: torch.Tensor
    global_valid: torch.Tensor


@dataclass
class RadiiParams(BaseParams):
    global_protos: torch.Tensor
    global_valid: torch.Tensor
    proto_radius_quantile: float = 0.95


@dataclass
class ClientEvalParams(BaseParams):
    global_protos: torch.Tensor
    global_valid: torch.Tensor
    raw_global_protos: torch.Tensor
    raw_global_valid: torch.Tensor
    proto_radius_quantile: float = 0.95


def mse_distance(features, prototypes):
    """返回样本与原型之间逐特征维平均的平方欧氏距离。"""
    return (features[:, None, :] - prototypes[None, :, :]).square().mean(dim=2)


PROTO_Q_THRESHOLDS = [
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
]


def prototype_top2_distance(
    features,
    prototypes,
    valid_mask,
):
    distance = mse_distance(features, prototypes)
    distance = distance.masked_fill(
        ~valid_mask.unsqueeze(0),
        float("inf"),
    )

    top2_distance, top2_index = torch.topk(
        distance,
        k=2,
        dim=1,
        largest=False,
    )

    pred = top2_index[:, 0]
    d1 = top2_distance[:, 0]
    d2 = top2_distance[:, 1]
    q = d1 / d2.clamp_min(1e-12)

    return pred, d1, d2, q


@torch.no_grad()
def estimate_prototypes_worker(p: BaseParams):
    """客户端使用当前轮次最新的全局模型，在有标签数据上提取类别原型。"""
    device = torch.device(p.client_gpu)

    model = get_model(p).to(device)

    model.load_state_dict(p.model_state)
    model.eval()

    labeled_indices = torch.where(p.train_set.is_labeled.bool())[0]
    labeled_x = prepare_input_batch(p.train_set.x[labeled_indices], p.dataset)
    labeled_y = p.train_set.y[labeled_indices]
    loader = DataLoader(
        TensorDataset(labeled_x, labeled_y),
        batch_size=max(p.batch_size, 256),
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
def client_eval_and_diagnose_worker(p: ClientEvalParams):
    """客户端使用当前全局模型与全局原型，并行完成门控统计、类别半径估计、相对几何统计与无标签诊断。

    对有标签样本仅进行一次前向特征提取：
      1. 用优化后的目标原型 global_protos 计算门控 q 与预测正确性；
      2. 用优化后的目标原型 global_protos 提取真实类别距离并按类用 torch.quantile 计算半径；
      3. 用未优化的统计原型 raw_global_protos 计算相对原型几何 (真实距离与最近错误距离比值)。
    对无标签样本进行并行推断，替代服务端串行推断。
    """
    device = torch.device(p.client_gpu)
    model = get_model(p).to(device)
    model.load_state_dict(p.model_state)
    model.eval()

    num_class = p.num_class
    target_protos = p.global_protos.to(device)
    target_valid = p.global_valid.to(device).bool()
    raw_protos = p.raw_global_protos.to(device)
    raw_valid = p.raw_global_valid.to(device).bool()

    n_target_valid = int(target_valid.sum().item())
    n_raw_valid = int(raw_valid.sum().item())

    # 1. 有标签数据处理 (单次前向同时产出门控 q、类别半径、相对几何)
    labeled_mask = p.train_set.is_labeled.bool()
    labeled_indices = torch.where(labeled_mask)[0]

    q_by_class = [[] for _ in range(num_class)]
    correct_by_class = [[] for _ in range(num_class)]
    distance_by_class = [[] for _ in range(num_class)]

    rel_true_dist_list = []
    rel_wrong_dist_list = []
    rel_q_list = []
    rel_y_list = []

    radii = torch.zeros(num_class, device=device)
    radius_count = torch.zeros(num_class, dtype=torch.long, device=device)

    if len(labeled_indices) > 0:
        labeled_x = p.train_set.x[labeled_indices]
        labeled_y = p.train_set.y[labeled_indices]
        eval_bs = 256
        n_labeled = labeled_x.size(0)

        for i in range(0, n_labeled, eval_bs):
            bx = prepare_input_batch(labeled_x[i : i + eval_bs].to(device), p.dataset)
            by = labeled_y[i : i + eval_bs].to(device)

            features = model.extractor(bx)

            # (1) 目标原型计算：门控与类别半径
            dist_target = mse_distance(features, target_protos)
            if n_target_valid >= 2:
                dist_gate = dist_target.masked_fill(
                    ~target_valid.unsqueeze(0), float("inf")
                )
                top2_dist, top2_idx = torch.topk(dist_gate, k=2, dim=1, largest=False)
                pred = top2_idx[:, 0]
                q = top2_dist[:, 0] / top2_dist[:, 1].clamp_min(1e-12)
                correct = pred.eq(by)

                for c in range(num_class):
                    mask_c = pred == c
                    if mask_c.any():
                        q_by_class[c].append(q[mask_c])
                        correct_by_class[c].append(correct[mask_c])

            valid_by_target = target_valid[by]
            if valid_by_target.any():
                true_dist_target = dist_target.gather(1, by.unsqueeze(1)).squeeze(1)
                for c in range(num_class):
                    if bool(target_valid[c]):
                        c_mask = (by == c) & valid_by_target
                        if c_mask.any():
                            distance_by_class[c].append(true_dist_target[c_mask])

            # (2) 统计原型计算：相对几何 (真实类别距离 vs 最近错误类别距离)
            valid_by_raw = raw_valid[by]
            if n_raw_valid >= 2 and valid_by_raw.any():
                dist_raw = mse_distance(features[valid_by_raw], raw_protos)
                y_raw = by[valid_by_raw]
                t_dist = dist_raw.gather(1, y_raw.unsqueeze(1)).squeeze(1)

                w_dist = dist_raw.clone()
                w_dist.scatter_(1, y_raw.unsqueeze(1), float("inf"))
                w_dist.masked_fill_(~raw_valid.unsqueeze(0), float("inf"))
                min_w_dist = w_dist.min(dim=1).values

                rel_ratio = t_dist / min_w_dist.clamp_min(1e-12)

                rel_true_dist_list.append(t_dist)
                rel_wrong_dist_list.append(min_w_dist)
                rel_q_list.append(rel_ratio)
                rel_y_list.append(y_raw)

        # 在 GPU 上按类别计算分位数半径
        for c in range(num_class):
            if distance_by_class[c]:
                c_dists = torch.cat(distance_by_class[c])
                radii[c] = torch.quantile(c_dists, p.proto_radius_quantile)
                radius_count[c] = c_dists.numel()

    # 2. 无标签数据诊断 (全 GPU 并行推断，消除服务端串行推断)
    unlabeled_mask = ~p.train_set.is_labeled.bool()
    unlabeled_indices = torch.where(unlabeled_mask)[0]

    labeled_present = (
        torch.bincount(p.train_set.y[labeled_mask], minlength=num_class) > 0
    ).to(device)

    client_stats = {
        "total": 0,
        "classifier_correct": 0,
        "prototype_correct": 0,
        "seen_total": 0,
        "seen_cls": 0,
        "seen_proto": 0,
        "missing_total": 0,
        "missing_cls": 0,
        "missing_proto": 0,
        "proto_q_all": [],
        "proto_q_pred_correct": [],
        "proto_q_missing_mask": [],
    }

    if len(unlabeled_indices) > 0:
        unlabeled_x = p.train_set.x[unlabeled_indices]
        unlabeled_y = p.train_set.y[unlabeled_indices]
        eval_bs = 256
        n_unlabeled = unlabeled_x.size(0)

        stats_counts = torch.zeros(9, dtype=torch.long, device=device)
        diag_q_list = []
        diag_corr_list = []
        diag_miss_list = []

        for i in range(0, n_unlabeled, eval_bs):
            bx = prepare_input_batch(unlabeled_x[i : i + eval_bs].to(device), p.dataset)
            by = unlabeled_y[i : i + eval_bs].to(device)

            features = model.extractor(bx)
            logits = model.classifier(features)
            classifier_pred = logits.argmax(dim=1)

            if n_target_valid >= 2:
                proto_pred, _, _, proto_q = prototype_top2_distance(
                    features, target_protos, target_valid
                )
            else:
                dist = mse_distance(features, target_protos)
                dist[:, ~target_valid] = float("inf")
                proto_pred = dist.argmin(dim=1)
                proto_q = None

            classifier_ok = classifier_pred == by
            proto_ok = (
                proto_pred == by
                if target_valid.any()
                else torch.zeros_like(classifier_ok)
            )
            missing = ~labeled_present[by]
            seen = labeled_present[by]

            stats_counts[0] += by.numel()
            stats_counts[1] += classifier_ok.sum()
            stats_counts[2] += proto_ok.sum()
            stats_counts[3] += seen.sum()
            stats_counts[4] += (seen & classifier_ok).sum()
            stats_counts[5] += (seen & proto_ok).sum()
            stats_counts[6] += missing.sum()
            stats_counts[7] += (missing & classifier_ok).sum()
            stats_counts[8] += (missing & proto_ok).sum()

            if proto_q is not None:
                diag_q_list.append(proto_q)
                diag_corr_list.append(proto_ok)
                diag_miss_list.append(missing)

        counts_cpu = stats_counts.cpu().tolist()
        client_stats["total"] = counts_cpu[0]
        client_stats["classifier_correct"] = counts_cpu[1]
        client_stats["prototype_correct"] = counts_cpu[2]
        client_stats["seen_total"] = counts_cpu[3]
        client_stats["seen_cls"] = counts_cpu[4]
        client_stats["seen_proto"] = counts_cpu[5]
        client_stats["missing_total"] = counts_cpu[6]
        client_stats["missing_cls"] = counts_cpu[7]
        client_stats["missing_proto"] = counts_cpu[8]
        if diag_q_list:
            client_stats["proto_q_all"] = [
                torch.cat(diag_q_list).cpu().detach().clone()
            ]
            client_stats["proto_q_pred_correct"] = [
                torch.cat(diag_corr_list).cpu().detach().clone()
            ]
            client_stats["proto_q_missing_mask"] = [
                torch.cat(diag_miss_list).cpu().detach().clone()
            ]

    gate_q = [
        torch.cat(ql).cpu().detach().clone() if ql else torch.empty(0)
        for ql in q_by_class
    ]
    gate_corr = [
        torch.cat(cl).cpu().detach().clone() if cl else torch.empty(0, dtype=torch.bool)
        for cl in correct_by_class
    ]

    rel_geom = {
        "true_dist": (
            torch.cat(rel_true_dist_list).cpu().detach().clone()
            if rel_true_dist_list
            else torch.empty(0)
        ),
        "wrong_dist": (
            torch.cat(rel_wrong_dist_list).cpu().detach().clone()
            if rel_wrong_dist_list
            else torch.empty(0)
        ),
        "q": (
            torch.cat(rel_q_list).cpu().detach().clone()
            if rel_q_list
            else torch.empty(0)
        ),
        "y": (
            torch.cat(rel_y_list).cpu().detach().clone()
            if rel_y_list
            else torch.empty(0, dtype=torch.long)
        ),
    }

    return {
        "radii": radii.cpu().detach().clone(),
        "radius_count": radius_count.cpu().detach().clone(),
        "q_by_class": gate_q,
        "correct_by_class": gate_corr,
        "rel_geom": rel_geom,
        "diagnose_stats": client_stats,
    }


@torch.no_grad()
def estimate_proto_gate_worker(p: ProtoGateParams):
    """客户端使用当前全局模型与判别目标原型，在有标签数据上按预测类别统计 q 与预测正误。"""
    device = torch.device(p.client_gpu)
    model = get_model(p).to(device)
    model.load_state_dict(p.model_state)
    model.eval()

    prototypes = p.global_protos.to(device)
    valid = p.global_valid.to(device).bool()

    labeled_indices = torch.where(p.train_set.is_labeled.bool())[0].tolist()
    loader = DataLoader(
        Subset(p.train_set, labeled_indices),
        batch_size=p.batch_size,
        shuffle=False,
    )

    q_by_class = [[] for _ in range(p.num_class)]
    correct_by_class = [[] for _ in range(p.num_class)]

    if int(valid.sum().item()) >= 2:
        for x, y, *_ in loader:
            x, y = x.to(device), y.to(device)
            x = prepare_input_batch(x, p.dataset)
            features = model.extractor(x)
            pred, _, _, q = prototype_top2_distance(features, prototypes, valid)
            correct = pred.eq(y)

            for c in range(p.num_class):
                mask_c = pred == c
                if bool(mask_c.any()):
                    q_by_class[c].append(q[mask_c].cpu())
                    correct_by_class[c].append(correct[mask_c].cpu())

    return {
        "q_by_class": [
            torch.cat(q_list) if q_list else torch.empty(0) for q_list in q_by_class
        ],
        "correct_by_class": [
            torch.cat(corr_list) if corr_list else torch.empty(0, dtype=torch.bool)
            for corr_list in correct_by_class
        ],
    }


@torch.no_grad()
def estimate_radii_worker(p: RadiiParams):
    """客户端使用当前轮次最新的全局模型与全局原型，重估类别半径。"""
    device = torch.device(p.client_gpu)
    model = get_model(p).to(device)
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


def train(p: Params):
    device = torch.device(p.client_gpu)
    model_l = get_model(p).to(device)
    model_l.load_state_dict(p.model_state)
    has_unlabeled_loss = p.lambda_u > 0.0
    if has_unlabeled_loss:
        model_g = get_model(p).to(
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

    labeled_mask = p.train_set.is_labeled.bool()
    labeled_present = (
        torch.bincount(
            p.train_set.y[labeled_mask],
            minlength=p.num_class,
        )
        > 0
    ).to(device)

    global_protos = p.global_protos.to(device)
    global_valid = p.global_valid.to(device).bool()
    proto_q_thresholds = p.proto_q_thresholds.to(device)
    proto_q_valid = p.proto_q_valid.to(device).bool()

    optimizer = torch.optim.SGD(
        model_l.parameters(), lr=p.lr, momentum=p.momentum, weight_decay=p.weight_decay
    )
    can_pseudo_label = has_unlabeled_loss and (int(global_valid.sum().item()) >= 2)

    # 在 GPU 上维护 float64 统计缓冲区，彻底消除 batch 内部的 .item() / int() 强同步
    # 0: loss_x_sum, 1: sample_count, 2: calibrate_loss_sum, 3: calibrate_loss_count
    # 4: loss_u_sum, 5: loss_u_count, 6: total_loss_sum, 7: total_loss_count
    # 8: proto_unlabeled_total, 9: proto_missing_total, 10: proto_selected_count
    # 11: proto_selected_correct, 12: proto_selected_missing_count
    # 13: proto_selected_missing_correct, 14: proto_q_sum_selected
    train_stats = torch.zeros(15, dtype=torch.float64, device=device)

    model_l.train()
    for _ in range(p.epochs):
        for labeled_batch, unlabeled_batch in iterate_fixmatch_batches(loaders):
            x_l_raw, y_l = labeled_batch
            x_u_raw = unlabeled_batch[0]
            x_l_raw, y_l = x_l_raw.to(device), y_l.to(device)
            x_u_raw = x_u_raw.to(device)

            # 仅反向传播的有标签分支使用弱增强。
            x_l_weak = weak_augment(x_l_raw, p.dataset)
            features_l = model_l.extractor(x_l_weak)
            loss_x = F.cross_entropy(model_l.classifier(features_l), y_l)
            loss = loss_x

            # 无分支且防除以 0 的校准损失计算
            valid_labeled = global_valid[y_l]
            v_count = valid_labeled.sum().double()
            diff_per_sample = (features_l - global_protos[y_l]).square().mean(dim=1)
            loss_cal = (
                diff_per_sample * valid_labeled.float()
            ).sum() / v_count.clamp_min(1.0).float()
            loss = loss + p.lambda_s * loss_cal
            train_stats[2] += (
                diff_per_sample.detach().double() * valid_labeled.double()
            ).sum()
            train_stats[3] += v_count

            loss_u = loss_x * 0.0
            selected_u = torch.zeros(x_u_raw.size(0), dtype=torch.bool, device=device)
            pseudo_labels = torch.zeros(
                x_u_raw.size(0), dtype=torch.long, device=device
            )
            proto_q_values = torch.zeros(x_u_raw.size(0), device=device)

            if can_pseudo_label:
                with torch.no_grad():
                    teacher_features = model_g.extractor(
                        prepare_input_batch(x_u_raw, p.dataset)
                    )
                    pseudo_labels, _, _, proto_q_values = prototype_top2_distance(
                        teacher_features,
                        global_protos,
                        global_valid,
                    )
                    selected_u = proto_q_valid[pseudo_labels] & (
                        proto_q_values <= proto_q_thresholds[pseudo_labels]
                    )

                x_u_strong = strong_augment(x_u_raw, p.dataset)
                features_u = model_l.extractor(x_u_strong)
                logits_u = model_l.classifier(features_u)

                loss_per_sample = F.cross_entropy(
                    logits_u,
                    pseudo_labels,
                    reduction="none",
                )
                loss_u = (loss_per_sample * selected_u.float()).mean()
                loss = loss + p.lambda_u * loss_u

            unlabeled_count = float(x_u_raw.size(0))
            train_stats[4] += loss_u.detach().double() * unlabeled_count
            train_stats[5] += unlabeled_count

            # 真实 y_u 仅在所有训练损失构造完成后的诊断分支中读取 (完全无分支累加)
            with torch.no_grad():
                y_u = unlabeled_batch[1].to(device)
                missing_u = ~labeled_present[y_u]
                selected_missing = selected_u & missing_u

                train_stats[8] += float(y_u.numel())
                train_stats[9] += missing_u.sum().double()
                train_stats[10] += selected_u.sum().double()
                train_stats[11] += (selected_u & pseudo_labels.eq(y_u)).sum().double()
                train_stats[12] += selected_missing.sum().double()
                train_stats[13] += (
                    (selected_missing & pseudo_labels.eq(y_u)).sum().double()
                )
                train_stats[14] += (proto_q_values * selected_u.float()).sum().double()

            optimizer.zero_grad()
            check_losses(loss, locals())
            loss.backward()
            optimizer.step()

            bs_l = float(y_l.size(0))
            train_stats[0] += loss_x.detach().double() * bs_l
            train_stats[1] += bs_l
            train_stats[6] += loss.detach().double() * bs_l
            train_stats[7] += bs_l

    stats_cpu = train_stats.cpu().tolist()
    loss_sum = stats_cpu[0]
    sample_count = int(stats_cpu[1])
    calibrate_loss_sum = stats_cpu[2]
    calibrate_loss_count = int(stats_cpu[3])
    loss_u_sum = stats_cpu[4]
    loss_u_count = int(stats_cpu[5])
    total_loss_sum = stats_cpu[6]
    total_loss_count = int(stats_cpu[7])

    proto_diag = {
        "proto_unlabeled_total": int(stats_cpu[8]),
        "proto_missing_total": int(stats_cpu[9]),
        "proto_selected_count": int(stats_cpu[10]),
        "proto_selected_correct": int(stats_cpu[11]),
        "proto_selected_missing_count": int(stats_cpu[12]),
        "proto_selected_missing_correct": int(stats_cpu[13]),
        "proto_q_sum_selected": float(stats_cpu[14]),
    }

    return {
        "state": clone_cpu_state(model_l.state_dict()),
        "loss": total_loss_sum / max(1, total_loss_count),
        "loss_x_sum": loss_sum,
        "loss_x_count": sample_count,
        "loss_calibrate_sum": calibrate_loss_sum,
        "loss_calibrate_count": calibrate_loss_count,
        "loss_u_sum": loss_u_sum,
        "loss_u_count": loss_u_count,
        "loss_total_sum": total_loss_sum,
        "loss_total_count": total_loss_count,
        **proto_diag,
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
        self.proto_q_target_precision = args.proto_q_target_precision
        self.proto_q_min_samples = args.proto_q_min_samples
        self.proto_q_max_threshold = args.proto_q_max_threshold
        self.proto_q_history_rounds = max(1, int(args.proto_q_history_rounds))
        self.proto_q_history = []
        self.mean_protos = torch.zeros(self.num_class, self.feature_dim)
        self.mean_valid = torch.zeros(self.num_class, dtype=torch.bool)
        self.global_protos = torch.zeros(self.num_class, self.feature_dim)
        self.global_valid = torch.zeros(self.num_class, dtype=torch.bool)
        self.global_radii = torch.zeros(self.num_class)
        self.radius_valid = torch.zeros(self.num_class, dtype=torch.bool)
        self.proto_q_thresholds = torch.zeros(self.num_class)
        self.proto_q_valid = torch.zeros(self.num_class, dtype=torch.bool)
        self.proto_q_gate_details = []
        self.lambda_p = args.lambda_p
        self.lambda_s = args.lambda_s
        self.lambda_u = args.lambda_u
        self.unlabeled_ratio = args.unlabeled_ratio
        self.tau = args.tau
        self.proto_radius_quantile = args.proto_radius_quantile
        self.loss_proto = []
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
            "proto_q_thresholds_by_round",
            "proto_q_valid_by_round",
            "proto_selected_coverage",
            "proto_selected_accuracy",
            "proto_selected_count",
            "proto_selected_mean_q",
            "proto_selected_missing_count",
            "proto_selected_missing_accuracy",
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
            "proto_q_correct_mean",
            "proto_q_wrong_mean",
            "proto_q_correct_quantiles",
            "proto_q_wrong_quantiles",
            "proto_q_threshold_coverage",
            "proto_q_threshold_accuracy",
            "proto_q_missing_threshold_coverage",
            "proto_q_missing_threshold_accuracy",
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

    def _build_proto_q_thresholds(self, gate_results, selected):
        current_round_gate = []
        for c in range(self.num_class):
            q_list = [
                gate_results[cid]["q_by_class"][c].cpu().detach().clone()
                for cid in selected
                if gate_results[cid]["q_by_class"][c].numel() > 0
            ]
            corr_list = [
                gate_results[cid]["correct_by_class"][c].cpu().detach().clone()
                for cid in selected
                if gate_results[cid]["correct_by_class"][c].numel() > 0
            ]
            q_c = torch.cat(q_list) if q_list else torch.empty(0)
            corr_c = (
                torch.cat(corr_list) if corr_list else torch.empty(0, dtype=torch.bool)
            )
            current_round_gate.append({"q": q_c, "correct": corr_c})

        self.proto_q_history.append(current_round_gate)
        if len(self.proto_q_history) > self.proto_q_history_rounds:
            self.proto_q_history = self.proto_q_history[-self.proto_q_history_rounds :]

        history_len = len(self.proto_q_history)
        thresholds = torch.zeros(self.num_class)
        valid = torch.zeros(self.num_class, dtype=torch.bool)
        gate_details = []
        diag_thresholds = [0.5, 0.6, 0.65, 0.7]

        for c in range(self.num_class):
            q_history_c = [
                round_gate[c]["q"]
                for round_gate in self.proto_q_history
                if round_gate[c]["q"].numel() > 0
            ]
            corr_history_c = [
                round_gate[c]["correct"]
                for round_gate in self.proto_q_history
                if round_gate[c]["correct"].numel() > 0
            ]
            if not q_history_c:
                gate_details.append(
                    {
                        "history_rounds": history_len,
                        "n_pred": 0,
                        "n_q60": 0,
                        "acc_q60": 0.0,
                        "n_tau": 0,
                        "acc_tau": 0.0,
                        "tau": None,
                        "valid": False,
                        "threshold_stats": {
                            t: {"count": 0, "accuracy": 0.0} for t in diag_thresholds
                        },
                    }
                )
                continue

            q_all = torch.cat(q_history_c)
            corr_all = torch.cat(corr_history_c)

            threshold_stats = {}
            for threshold in diag_thresholds:
                mask = q_all <= threshold
                n = int(mask.sum().item())
                acc = (
                    float(corr_all[mask].float().mean().item()) * 100.0
                    if n > 0
                    else 0.0
                )
                threshold_stats[threshold] = {
                    "count": n,
                    "accuracy": acc,
                }

            order = torch.argsort(q_all)
            q_sorted = q_all[order]
            corr_sorted = corr_all[order]

            cum_correct = corr_sorted.float().cumsum(0)
            count = torch.arange(1, len(corr_sorted) + 1, dtype=torch.float32)
            precision = cum_correct / count

            q60_mask = q_sorted <= self.proto_q_max_threshold
            n_q60 = int(q60_mask.sum().item())
            acc_q60 = (
                float(corr_sorted[q60_mask].float().mean().item()) * 100.0
                if n_q60 > 0
                else 0.0
            )

            eligible = (
                (precision >= self.proto_q_target_precision)
                & (count >= self.proto_q_min_samples)
                & q60_mask
            )

            indices = torch.where(eligible)[0]
            if indices.numel() > 0:
                k = int(indices[-1].item())
                tau_val = float(q_sorted[k].item())
                thresholds[c] = tau_val
                valid[c] = True
                gate_details.append(
                    {
                        "history_rounds": history_len,
                        "n_pred": int(q_all.numel()),
                        "n_q60": n_q60,
                        "acc_q60": acc_q60,
                        "n_tau": k + 1,
                        "acc_tau": float(precision[k].item()) * 100.0,
                        "tau": tau_val,
                        "valid": True,
                        "threshold_stats": threshold_stats,
                    }
                )
            else:
                gate_details.append(
                    {
                        "history_rounds": history_len,
                        "n_pred": int(q_all.numel()),
                        "n_q60": n_q60,
                        "acc_q60": acc_q60,
                        "n_tau": 0,
                        "acc_tau": 0.0,
                        "tau": None,
                        "valid": False,
                        "threshold_stats": threshold_stats,
                    }
                )

        self.proto_q_thresholds = thresholds
        self.proto_q_valid = valid
        self.proto_q_gate_details = gate_details

    @torch.no_grad()
    def _record_relative_proto_geometry(self, eval_results, selected):
        valid = self.raw_global_valid.bool()
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

        all_true_dist = [
            eval_results[cid]["rel_geom"]["true_dist"]
            for cid in selected
            if eval_results[cid]["rel_geom"]["true_dist"].numel() > 0
        ]
        all_wrong_dist = [
            eval_results[cid]["rel_geom"]["wrong_dist"]
            for cid in selected
            if eval_results[cid]["rel_geom"]["wrong_dist"].numel() > 0
        ]
        all_q = [
            eval_results[cid]["rel_geom"]["q"]
            for cid in selected
            if eval_results[cid]["rel_geom"]["q"].numel() > 0
        ]
        all_y = [
            eval_results[cid]["rel_geom"]["y"]
            for cid in selected
            if eval_results[cid]["rel_geom"]["y"].numel() > 0
        ]

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
        loader = DataLoader(dataset, batch_size=256, shuffle=False)
        stats = self._empty_stats()
        self.model.to(self.device).eval()
        prototypes = self.global_protos.to(self.device)
        valid = self.global_valid.to(self.device).bool()
        present = labeled_present.to(self.device).bool()
        n_valid = int(valid.sum().item())
        for x, y, _, is_labeled in loader:
            mask = ~is_labeled.bool()
            if not bool(mask.any()):
                continue
            x, y = x[mask].to(self.device), y[mask].to(self.device)
            x = prepare_input_batch(x, self.dataset)
            feature = self.model.extractor(x)
            logits = self.model.classifier(feature)
            classifier_pred = logits.argmax(dim=1)
            if n_valid >= 2:
                proto_pred, _, _, proto_q = prototype_top2_distance(
                    feature,
                    prototypes,
                    valid,
                )
            else:
                distance = mse_distance(feature, prototypes)
                distance[:, ~valid] = float("inf")
                proto_pred = distance.argmin(dim=1)
                proto_q = None

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

            if proto_q is not None:
                stats["proto_q_all"].append(proto_q.detach().cpu())
                stats["proto_q_pred_correct"].append(proto_ok.detach().cpu())
                stats["proto_q_missing_mask"].append(missing.detach().cpu())
        return stats

    @torch.no_grad()
    def _evaluate_accuracy(self):
        loader = DataLoader(self.test_set, batch_size=256, shuffle=False)
        self.model.to(self.device).eval()
        prototypes = self.global_protos.to(self.device)
        valid = self.global_valid.to(self.device).bool()
        mean_prototypes = self.mean_protos.to(self.device)
        mean_valid = self.mean_valid.to(self.device).bool()

        counts = torch.zeros(4, dtype=torch.long, device=self.device)
        # 0: model_correct, 1: mean_proto_correct, 2: proto_correct, 3: total

        for x, y, *_ in loader:
            x, y = x.to(self.device), y.to(self.device)
            feature = self.model.extractor(x)
            counts[0] += (self.model.classifier(feature).argmax(1) == y).sum()

            if bool(mean_valid.any()):
                mean_distance = mse_distance(feature, mean_prototypes)
                mean_distance[:, ~mean_valid] = float("inf")
                mean_proto_pred = mean_distance.argmin(1)
                counts[1] += (mean_proto_pred == y).sum()

            if bool(valid.any()):
                distance = mse_distance(feature, prototypes)
                distance[:, ~valid] = float("inf")
                proto_pred = distance.argmin(1)
                counts[2] += (proto_pred == y).sum()

            counts[3] += y.numel()

        self.model.cpu()
        counts_cpu = counts.cpu().tolist()
        total = max(1, counts_cpu[3])
        return (
            self._percent(counts_cpu[0], total),
            self._percent(counts_cpu[1], total),
            self._percent(counts_cpu[2], total),
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
                        proto_q_thresholds=self.proto_q_thresholds.clone(),
                        proto_q_valid=self.proto_q_valid.clone(),
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

            # 4. 客户端全并发运行融合评估任务 (门控 q + 半径 + 相对几何 + 无标签诊断)
            eval_params = [
                ClientEvalParams(
                    **asdict(base),
                    global_protos=self.global_protos.clone(),
                    global_valid=self.global_valid.clone(),
                    raw_global_protos=self.raw_global_protos.clone(),
                    raw_global_valid=self.raw_global_valid.clone(),
                    proto_radius_quantile=self.proto_radius_quantile,
                )
                for base in self.build_base_params(selected)
            ]
            eval_results = self.run_clients(
                client_eval_and_diagnose_worker,
                eval_params,
            )

            # 服务端直接汇总门控、半径、相对几何与无标签诊断 (消除服务端串行推断)
            self._build_proto_q_thresholds(
                eval_results,
                selected,
            )
            self.proto_q_thresholds_by_round.append(self.proto_q_thresholds.tolist())
            self.proto_q_valid_by_round.append(self.proto_q_valid.tolist())

            self._aggregate_global_radii(eval_results, selected)
            self._record_radius_diagnostics(eval_results, selected)
            self._record_prototype_drift(
                eval_results,
                selected,
                self.global_protos,
                global_model_protos,
                global_model_counts,
            )
            self._record_interclass_radius_geometry()
            self._record_relative_proto_geometry(eval_results, selected)

            proto_loss = (
                self.proto_anchor_weight * self.proto_opt_anchor_loss[-1]
                + self.proto_sep_weight * self.proto_opt_sep_loss[-1]
            )
            stats = self._empty_stats()
            for cid in selected:
                stats = self._merge_stats(stats, eval_results[cid]["diagnose_stats"])
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

            # 统计客户端本轮训练中 q 门控的实际执行效果
            proto_train_unlabeled = sum(
                results[cid]["proto_unlabeled_total"] for cid in selected
            )
            proto_train_selected = sum(
                results[cid]["proto_selected_count"] for cid in selected
            )
            proto_train_correct = sum(
                results[cid]["proto_selected_correct"] for cid in selected
            )
            proto_train_missing_selected = sum(
                results[cid]["proto_selected_missing_count"] for cid in selected
            )
            proto_train_missing_correct = sum(
                results[cid]["proto_selected_missing_correct"] for cid in selected
            )
            proto_train_q_sum = sum(
                results[cid]["proto_q_sum_selected"] for cid in selected
            )

            train_cov = 100.0 * proto_train_selected / max(1, proto_train_unlabeled)
            train_acc = (
                100.0 * proto_train_correct / max(1, proto_train_selected)
                if proto_train_selected > 0
                else 0.0
            )
            train_mean_q = (
                proto_train_q_sum / max(1, proto_train_selected)
                if proto_train_selected > 0
                else 0.0
            )
            train_missing_acc = (
                100.0
                * proto_train_missing_correct
                / max(1, proto_train_missing_selected)
                if proto_train_missing_selected > 0
                else 0.0
            )

            self.proto_selected_coverage.append(train_cov)
            self.proto_selected_accuracy.append(train_acc)
            self.proto_selected_count.append(proto_train_selected)
            self.proto_selected_mean_q.append(train_mean_q)
            self.proto_selected_missing_count.append(proto_train_missing_selected)
            self.proto_selected_missing_accuracy.append(train_missing_acc)

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
                time.time() - started,
            )
            self._print_candidate_diagnostics()

    @staticmethod
    def _empty_stats():
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
        stats.update(
            {
                "proto_q_all": [],
                "proto_q_pred_correct": [],
                "proto_q_missing_mask": [],
            }
        )
        return stats

    @staticmethod
    def _merge_stats(left, right):
        for key in left:
            if isinstance(left[key], list):
                left[key].extend(right[key])
            else:
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

        # 目标原型 q 可靠性诊断统计
        has_q_data = bool(stats["proto_q_all"])
        if not has_q_data:
            self.proto_q_correct_mean.append(0.0)
            self.proto_q_wrong_mean.append(0.0)
            self.proto_q_correct_quantiles.append([0.0] * 4)
            self.proto_q_wrong_quantiles.append([0.0] * 4)
            self.proto_q_threshold_coverage.append([0.0] * len(PROTO_Q_THRESHOLDS))
            self.proto_q_threshold_accuracy.append([0.0] * len(PROTO_Q_THRESHOLDS))
            self.proto_q_missing_threshold_coverage.append(
                [0.0] * len(PROTO_Q_THRESHOLDS)
            )
            self.proto_q_missing_threshold_accuracy.append(
                [0.0] * len(PROTO_Q_THRESHOLDS)
            )
        else:
            q_all = torch.cat(stats["proto_q_all"])
            pred_correct = torch.cat(stats["proto_q_pred_correct"])
            missing_mask = torch.cat(stats["proto_q_missing_mask"])

            q_correct = q_all[pred_correct]
            q_wrong = q_all[~pred_correct]

            self.proto_q_correct_mean.append(
                float(q_correct.mean().item()) if q_correct.numel() > 0 else 0.0
            )
            self.proto_q_wrong_mean.append(
                float(q_wrong.mean().item()) if q_wrong.numel() > 0 else 0.0
            )

            q_probs = torch.tensor([0.25, 0.50, 0.75, 0.90])
            correct_q_list = (
                torch.quantile(q_correct, q_probs).tolist()
                if q_correct.numel() > 0
                else [0.0] * 4
            )
            wrong_q_list = (
                torch.quantile(q_wrong, q_probs).tolist()
                if q_wrong.numel() > 0
                else [0.0] * 4
            )
            self.proto_q_correct_quantiles.append(correct_q_list)
            self.proto_q_wrong_quantiles.append(wrong_q_list)

            cov_all, acc_all = [], []
            cov_missing, acc_missing = [], []

            missing_q = q_all[missing_mask]
            missing_correct = pred_correct[missing_mask]

            for threshold in PROTO_Q_THRESHOLDS:
                selected = q_all <= threshold
                coverage = float(selected.float().mean().item()) * 100.0
                accuracy = (
                    float(pred_correct[selected].float().mean().item()) * 100.0
                    if selected.any()
                    else 0.0
                )
                cov_all.append(coverage)
                acc_all.append(accuracy)

                if missing_q.numel() > 0:
                    selected_m = missing_q <= threshold
                    coverage_m = float(selected_m.float().mean().item()) * 100.0
                    accuracy_m = (
                        float(missing_correct[selected_m].float().mean().item()) * 100.0
                        if selected_m.any()
                        else 0.0
                    )
                else:
                    coverage_m, accuracy_m = 0.0, 0.0
                cov_missing.append(coverage_m)
                acc_missing.append(accuracy_m)

            self.proto_q_threshold_coverage.append(cov_all)
            self.proto_q_threshold_accuracy.append(acc_all)
            self.proto_q_missing_threshold_coverage.append(cov_missing)
            self.proto_q_missing_threshold_accuracy.append(acc_missing)

        print(
            "准确率（本轮服务器更新完成后）：\n"
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

        # 客户端实际训练执行的 q 门控统计
        print(
            "客户端训练 q 门控执行统计（本轮训练）：\n"
            f"实际参与 Lu 样本数={self.proto_selected_count[-1]} "
            f"(覆盖率={self.proto_selected_coverage[-1]:.2f}%)\n"
            f"实际伪标签准确率={self.proto_selected_accuracy[-1]:.2f}%\n"
            f"选中样本平均 q={self.proto_selected_mean_q[-1]:.4f}\n"
            f"缺失类别：选中样本数={self.proto_selected_missing_count[-1]}，"
            f"缺失类别伪标签准确率={self.proto_selected_missing_accuracy[-1]:.2f}%"
        )

        if has_q_data:
            cov_lines = [
                f"q<={t:.1f}：覆盖率={c:.2f}%，伪标签准确率={a:.2f}%"
                for t, c, a in zip(
                    PROTO_Q_THRESHOLDS,
                    self.proto_q_threshold_coverage[-1],
                    self.proto_q_threshold_accuracy[-1],
                )
            ]
            missing_cov_lines = [
                f"q<={t:.1f}：覆盖率={c:.2f}%，伪标签准确率={a:.2f}%"
                for t, c, a in zip(
                    PROTO_Q_THRESHOLDS,
                    self.proto_q_missing_threshold_coverage[-1],
                    self.proto_q_missing_threshold_accuracy[-1],
                )
            ]
            c_q = self.proto_q_correct_quantiles[-1]
            w_q = self.proto_q_wrong_quantiles[-1]
            print(
                "目标原型 q 可靠性诊断：\n"
                f"正确预测样本 q 均值={self.proto_q_correct_mean[-1]:.4f}\n"
                f"错误预测样本 q 均值={self.proto_q_wrong_mean[-1]:.4f}\n\n"
                "正确预测 q：\n"
                f"P25={c_q[0]:.4f} P50={c_q[1]:.4f} P75={c_q[2]:.4f} P90={c_q[3]:.4f}\n\n"
                "错误预测 q：\n"
                f"P25={w_q[0]:.4f} P50={w_q[1]:.4f} P75={w_q[2]:.4f} P90={w_q[3]:.4f}\n\n"
                "q 阈值筛选：\n"
                + "\n".join(cov_lines)
                + "\n\n缺失类别 q 阈值筛选：\n"
                + "\n".join(missing_cov_lines)
            )

        # 打印服务器刚估计出的类别自适应门控阈值（供下一轮客户端使用）
        gate_strs = []
        valid_taus = []
        for c in range(self.num_class):
            detail = (
                self.proto_q_gate_details[c]
                if c < len(self.proto_q_gate_details)
                else {
                    "n_pred": 0,
                    "n_q60": 0,
                    "acc_q60": 0.0,
                    "n_tau": 0,
                    "acc_tau": 0.0,
                    "tau": None,
                    "valid": False,
                }
            )
            tau_str = f"{detail['tau']:.4f}" if detail["tau"] is not None else "--"
            valid_str = "True" if detail["valid"] else "False"
            if detail["valid"]:
                valid_taus.append(detail["tau"])
            history_rounds = detail.get("history_rounds", len(self.proto_q_history))
            threshold_stats = detail.get("threshold_stats", {})
            diag_str = " | ".join(
                [
                    f"q<={threshold:.2f}: n={stat['count']}, acc={stat['accuracy']:.2f}%"
                    for threshold, stat in threshold_stats.items()
                ]
            )
            gate_strs.append(
                f"c{c}: tau={tau_str}, valid={valid_str} | "
                f"近{history_rounds}轮tau处样本数={detail['n_tau']}, "
                f"tau处准确率={detail['acc_tau']:.2f}% | "
                f"{diag_str}"
            )
        avg_tau = sum(valid_taus) / max(1, len(valid_taus)) if valid_taus else 0.0
        print(
            "目标原型 q 门控阈值（供下一轮客户端训练使用）：\n"
            + "\n".join(gate_strs)
            + f"\n有效门控类别数={len(valid_taus)}/{self.num_class}，平均有效 q 阈值={avg_tau:.4f}"
        )

        print(f"各客户端有标签类别样本数（按类别）：{self.class_support[-1]}")
        print(f"本轮耗时：{elapsed:.2f} 秒")

    def _print_candidate_diagnostics(self):
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
        print(
            "损失统计：\n"
            f"监督分类 CE={self.loss_x[-1]:.6f}\n"
            f"监督原型校准 MSE={self.loss_calibrate[-1]:.6f}\n"
            f"无标签门控 CE（按全部无标签样本归一化）={self.loss_u[-1]:.6f}\n"
            f"总训练损失={self.loss[-1]:.6f}"
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
            "proto_q_thresholds_by_round",
            "proto_q_valid_by_round",
            "proto_selected_coverage",
            "proto_selected_accuracy",
            "proto_selected_count",
            "proto_selected_mean_q",
            "proto_selected_missing_count",
            "proto_selected_missing_accuracy",
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
            "proto_q_correct_mean",
            "proto_q_wrong_mean",
            "proto_q_correct_quantiles",
            "proto_q_wrong_quantiles",
            "proto_q_threshold_coverage",
            "proto_q_threshold_accuracy",
            "proto_q_missing_threshold_coverage",
            "proto_q_missing_threshold_accuracy",
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
                    "proto_q_thresholds": self.proto_q_thresholds,
                    "proto_q_valid": self.proto_q_valid,
                    "proto_q_history": self.proto_q_history,
                },
            },
        )
