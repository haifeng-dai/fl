import os
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .utils import (
    BaseParams,
    BaseServer,
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
        f"{args.common_name}_{fmt_num(args.lambda_s)}_{fmt_num(args.lambda_u)}"
    )
    return os.path.join(args.log_path, f"{args.file_name}_{args.cur_time}.log")


@dataclass
class Params(BaseParams):
    unlabeled_ratio: int
    global_protos: torch.Tensor
    global_valid: torch.Tensor
    proto_scale: float
    lambda_s: float
    lambda_u: float


@dataclass
class ClientCalibrateParams(BaseParams):
    global_protos: torch.Tensor
    global_valid: torch.Tensor
    proto_scale: float


@dataclass
class ClientDiagnoseParams(BaseParams):
    global_protos: torch.Tensor
    global_valid: torch.Tensor
    proto_scale: float
    radius_q10: torch.Tensor
    radius_q50: torch.Tensor
    radius_valid: torch.Tensor


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


def mse_distance(features, prototypes):
    diff = features.unsqueeze(1) - prototypes.unsqueeze(0)
    return diff.square().mean(dim=-1)



def train(p: Params):
    device = torch.device(p.client_gpu)

    model_l = get_model(
        p.model_name,
        p.dataset,
        p.num_class,
        p.feature_dim,
    ).to(device)
    model_l.load_state_dict(p.model_state)

    loaders = build_fixmatch_loaders(
        p.train_set,
        p.batch_size,
        p.unlabeled_ratio,
    )
    if loaders.labeled_loader is None:
        raise ValueError("fedtest 客户端缺少有标签样本")

    global_protos = p.global_protos.to(device)
    global_valid = p.global_valid.to(device).bool()
    optimizer = torch.optim.SGD(
        model_l.parameters(),
        lr=p.lr,
        momentum=p.momentum,
        weight_decay=p.weight_decay,
    )

    stats = torch.zeros(6, dtype=torch.float64, device=device)
    # 0/1: loss_x sum/count
    # 2/3: loss_cal sum/count
    # 4/5: loss_total sum/count

    model_l.train()

    for _ in range(p.epochs):
        for labeled_batch, unlabeled_batch in iterate_ssl_batches(loaders):
            assert labeled_batch is not None

            x_l_raw, y_l = labeled_batch
            x_l_raw = x_l_raw.to(device)
            y_l = y_l.to(device)

            x_l_weak = weak_augment(x_l_raw, p.dataset)

            features_l = model_l.extractor(x_l_weak)
            logits_l = model_l.classifier(features_l)

            loss_x = F.cross_entropy(logits_l, y_l)

            diff = (
                mse_distance(
                    features_l,
                    global_protos,
                )
                .gather(
                    1,
                    y_l.unsqueeze(1),
                )
                .squeeze(1)
            )

            valid_labeled = global_valid[y_l]
            valid_count = valid_labeled.sum()

            loss_cal = (diff * valid_labeled.float()).sum() / valid_count.clamp_min(
                1
            ).float()

            loss = loss_x + p.lambda_s * loss_cal

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = float(y_l.size(0))

            stats[0] += loss_x.detach().double() * bs
            stats[1] += bs

            stats[2] += (diff.detach().double() * valid_labeled.double()).sum()
            stats[3] += valid_count

            stats[4] += loss.detach().double() * bs
            stats[5] += bs

    stats = stats.cpu().tolist()

    return {
        "state": clone_cpu_state(model_l.state_dict()),
        "loss_x_sum": stats[0],
        "loss_x_count": int(stats[1]),
        "loss_calibrate_sum": stats[2],
        "loss_calibrate_count": int(stats[3]),
        "loss_total_sum": stats[4],
        "loss_total_count": int(stats[5]),
    }


@torch.no_grad()
def client_calibrate_worker(p: ClientCalibrateParams):
    """收集各真实类别样本到自身目标原型的弱增强距离。"""
    device = torch.device(p.client_gpu)

    model = get_model(
        p.model_name,
        p.dataset,
        p.num_class,
        p.feature_dim,
    ).to(device)

    model.load_state_dict(p.model_state)
    model.eval()

    prototypes = p.global_protos.to(device)
    valid = p.global_valid.to(device).bool()

    class_distances = [[] for _ in range(p.num_class)]

    labeled_mask = p.train_set.is_labeled.bool()
    labeled_indices = torch.where(labeled_mask)[0]

    if labeled_indices.numel() > 0:
        labeled_x = p.train_set.x[labeled_indices]
        loader_l = DataLoader(
            TensorDataset(labeled_x, p.train_set.y[labeled_indices]),
            batch_size=max(p.batch_size, 256),
            shuffle=False,
        )
        for bx_raw, by in loader_l:
            bx_raw = bx_raw.to(device)
            bx_weak = weak_augment(bx_raw, p.dataset)
            features = model.extractor(bx_weak)
            dist = mse_distance(features, prototypes)
            dist[:, ~valid] = float("inf")
            own_dist = dist.gather(1, by.to(device)[:, None]).squeeze(1)
            proto_pred = dist.argmin(dim=1)
            proto_correct = proto_pred == by.to(device)
            for c in range(p.num_class):
                mask = (by == c) & valid[c].cpu() & proto_correct.cpu()
                if mask.any():
                    class_distances[c].append(own_dist[mask.to(device)].cpu())

    return {
        "class_distances": [
            torch.cat(v) if v else torch.empty(0) for v in class_distances
        ],
    }


@torch.no_grad()
def client_diagnose_worker(p: ClientDiagnoseParams):
    """阶段2：在无标签数据上评估基础分类与同空间下的候选集合质量。"""
    device = torch.device(p.client_gpu)

    model = get_model(
        p.model_name,
        p.dataset,
        p.num_class,
        p.feature_dim,
    ).to(device)

    model.load_state_dict(p.model_state)
    model.eval()

    prototypes = p.global_protos.to(device)
    valid = p.global_valid.to(device).bool()
    radius_q10 = p.radius_q10.to(device)
    radius_q50 = p.radius_q50.to(device)
    radius_valid = p.radius_valid.to(device).bool()

    num_class = p.num_class

    labeled_mask = p.train_set.is_labeled.bool()
    labeled_present = (
        torch.bincount(
            p.train_set.y[labeled_mask],
            minlength=num_class,
        )
        > 0
    ).to(device)

    unlabeled_indices = torch.where(~labeled_mask)[0]

    # counts 统计量:
    # 0: total
    # 1: cls_correct
    # 2: proto_correct
    # 3: seen_total
    # 4: seen_cls_correct
    # 5: seen_proto_correct
    # 6: missing_total
    # 7: missing_cls_correct
    # 8: missing_proto_correct
    # 9: cand_total (有候选评估的样本数)
    # 10: cand_cover (真实类别在候选集中的样本数)
    # 11: cand_size_sum (候选集大小总和)
    # 12: singleton_count (候选集大小为 1 的样本数)
    # 13: singleton_correct (单候选且预测正确的样本数)
    # 14: empty_count (空候选集样本数)
    # 15: mis_cand_total
    # 16: mis_cand_cover
    # 17: mis_cand_size_sum
    # 18: mis_singleton_count
    # 19: mis_singleton_correct
    # 20: mis_empty_count
    diag_counts = torch.zeros(35, dtype=torch.float64, device=device)

    has_thresh = bool(radius_valid.any())

    if unlabeled_indices.numel() > 0:
        raw_x = p.train_set.x[unlabeled_indices]
        raw_y = p.train_set.y[unlabeled_indices].to(device)
        loader_u = DataLoader(
            TensorDataset(raw_x, raw_y),
            batch_size=max(p.batch_size, 256),
            shuffle=False,
        )
        for bx_raw, by in loader_u:
            bx_raw = bx_raw.to(device)
            bx_prep = prepare_input_batch(bx_raw, p.dataset)
            feat_u = model.extractor(bx_prep)
            logits_u = model.classifier(feat_u)
            dist_u = mse_distance(feat_u, prototypes)
            dist_u[:, ~valid] = float("inf")
            prob_u = torch.softmax(-dist_u / max(float(p.proto_scale), 1e-8), dim=1)
            prob_max, prob_pred = prob_u.max(dim=1)
            prob90 = prob_max > 0.9
            diag_counts[26] += prob90.double().sum()
            diag_counts[27] += (prob90 & prob_pred.eq(by)).double().sum()

            cls_pred = logits_u.argmax(dim=1)
            cls_prob, cls_prob_pred = torch.softmax(logits_u, dim=1).max(dim=1)
            cls95 = cls_prob > 0.95
            proto_pred = dist_u.argmin(dim=1)

            cls_ok = cls_pred == by
            proto_ok = proto_pred == by
            is_seen = labeled_present[by]
            is_mis = ~is_seen
            diag_counts[30] += cls95.double().sum()
            diag_counts[31] += (cls95 & cls_prob_pred.eq(by)).double().sum()
            diag_counts[32] += (cls95 & is_mis).double().sum()
            diag_counts[33] += (cls95 & cls_prob_pred.eq(by) & is_mis).double().sum()
            diag_counts[28] += (prob90 & is_mis).double().sum()
            diag_counts[29] += (prob90 & prob_pred.eq(by) & is_mis).double().sum()

            n_batch = float(by.size(0))
            diag_counts[0] += n_batch
            diag_counts[1] += cls_ok.double().sum()
            diag_counts[2] += proto_ok.double().sum()

            diag_counts[3] += is_seen.double().sum()
            diag_counts[4] += (cls_ok & is_seen).double().sum()
            diag_counts[5] += (proto_ok & is_seen).double().sum()

            diag_counts[6] += is_mis.double().sum()
            diag_counts[7] += (cls_ok & is_mis).double().sum()
            diag_counts[8] += (proto_ok & is_mis).double().sum()

            if has_thresh:
                bx_weak = weak_augment(bx_raw, p.dataset)
                feat_u_weak = model.extractor(bx_weak)
                dist_u_weak = mse_distance(feat_u_weak, prototypes)
                high_mask = (dist_u_weak <= radius_q10.unsqueeze(0)) & radius_valid.unsqueeze(0)
                high_size = high_mask.sum(dim=1)
                high_true = high_mask.gather(1, by.unsqueeze(1)).squeeze(1)
                high_single = high_size == 1
                diag_counts[21] += (high_size > 0).double().sum()
                diag_counts[22] += high_true.double().sum()
                diag_counts[23] += high_single.double().sum()
                diag_counts[24] += (high_single & high_true).double().sum()
                diag_counts[25] += (high_size > 1).double().sum()
                cand_mask = (
                    dist_u_weak <= radius_q50.unsqueeze(0)
                ) & radius_valid.unsqueeze(0)
                c_size = cand_mask.sum(dim=1)
                true_in_cand = cand_mask.gather(1, by.unsqueeze(1)).squeeze(1)
                is_single = c_size == 1
                single_ok = is_single & true_in_cand
                is_empty = c_size == 0

                diag_counts[9] += n_batch
                diag_counts[10] += true_in_cand.double().sum()
                diag_counts[11] += c_size.double().sum()
                diag_counts[12] += is_single.double().sum()
                diag_counts[13] += single_ok.double().sum()
                diag_counts[14] += is_empty.double().sum()

                if is_mis.any():
                    diag_counts[15] += is_mis.double().sum()
                    diag_counts[16] += (true_in_cand & is_mis).double().sum()
                    diag_counts[17] += (c_size.float() * is_mis.float()).double().sum()
                    diag_counts[18] += (is_single & is_mis).double().sum()
                    diag_counts[19] += (single_ok & is_mis).double().sum()
                    diag_counts[20] += (is_empty & is_mis).double().sum()

    return {
        "diag_counts": diag_counts.cpu(),
    }


class Server(BaseServer):
    def __init__(self, args):
        if args.ssl in ("none", "client"):
            raise ValueError("fedtest 要求 sample、double 或 sfd 半监督数据。")
        super().__init__(args, is_ssl=True, pfl=False)
        if any(dataset.is_labeled is None for dataset in self.train_sets.values()):
            raise ValueError("fedtest 要求训练数据提供 is_labeled 字段。")

        self.mean_protos = torch.zeros(self.num_class, self.feature_dim)
        self.mean_valid = torch.zeros(self.num_class, dtype=torch.bool)
        self.global_protos = torch.zeros(self.num_class, self.feature_dim)
        self.global_valid = torch.zeros(self.num_class, dtype=torch.bool)

        self.proto_anchor_weight = args.proto_anchor_weight
        self.proto_sep_weight = args.proto_sep_weight
        self.proto_opt_lr = args.proto_opt_lr
        self.proto_opt_steps = args.proto_opt_steps

        self.lambda_s = args.lambda_s
        self.lambda_u = args.lambda_u
        self.unlabeled_ratio = args.unlabeled_ratio
        self.proto_scale = 1.0

        # 当前轮距离分位数半径
        self.radius_q10 = torch.zeros(self.num_class)
        self.radius_q50 = torch.zeros(self.num_class)
        self.radius_valid = torch.zeros(self.num_class, dtype=torch.bool)

        # 核心指标序列
        self.acc = []
        self.acc_proto = []
        self.acc_proto_mean = []
        self.loss = []

        metric_names = (
            "loss_x",
            "loss_calibrate",
            "round_time",
            "unlabeled_classifier_acc",
            "unlabeled_proto_acc",
            "seen_classifier_acc",
            "seen_proto_acc",
            "missing_classifier_acc",
            "missing_proto_acc",
            "cand_cov",
            "avg_cand_size",
            "singleton_ratio",
            "singleton_acc",
            "empty_ratio",
            "missing_cand_cov",
            "missing_avg_cand_size",
            "missing_singleton_ratio",
            "missing_singleton_acc",
            "missing_empty_ratio",
        )
        for name in metric_names:
            setattr(self, name, [])

    @staticmethod
    def _percent(value, total):
        return 100.0 * value / max(1, total)

    def _optimize_global_prototypes(self, raw_protos, valid):
        valid_indices = torch.where(valid)[0]

        if len(valid_indices) < 2:
            self.proto_scale = 1.0
            return raw_protos.clone()

        raw_valid = raw_protos[valid_indices].detach().to(self.device)
        target_valid = torch.nn.Parameter(raw_valid.clone())

        with torch.no_grad():
            raw_dist = mse_distance(raw_valid, raw_valid)
            pair_mask = torch.triu(
                torch.ones_like(raw_dist, dtype=torch.bool),
                diagonal=1,
            )
            proto_scale = raw_dist[pair_mask].median().detach().clamp_min(1e-12)

        optimizer = torch.optim.Adam([target_valid], lr=self.proto_opt_lr)

        for _ in range(self.proto_opt_steps):
            optimizer.zero_grad()

            anchor_distance = (target_valid - raw_valid).square().mean(dim=1)
            anchor_loss = (anchor_distance / proto_scale).mean()

            target_dist = mse_distance(target_valid, target_valid)
            sep_loss = torch.exp(-target_dist[pair_mask] / proto_scale).mean()

            loss = (
                self.proto_anchor_weight * anchor_loss
                + self.proto_sep_weight * sep_loss
            )

            loss.backward()
            optimizer.step()

        optimized = raw_protos.clone().to(self.device)
        optimized[valid_indices] = target_valid.detach()

        with torch.no_grad():
            optimized_valid = optimized[valid_indices]
            optimized_dist = mse_distance(optimized_valid, optimized_valid)
            probability_scale = (
                optimized_dist[pair_mask].median().detach().clamp_min(1e-12)
            )

        self.proto_scale = float(probability_scale.item())

        return optimized.cpu()

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
        for round_idx in range(self.rounds):
            round_start = time.time()
            selected = sorted(torch.randperm(self.num_clients)[:num_join].tolist())

            # 1. 客户端本地训练 (Teacher 候选集合生成 + Student 偏标签优化)
            train_params = [
                Params(
                    **asdict(base),
                    unlabeled_ratio=self.unlabeled_ratio,
                    global_protos=self.global_protos.clone(),
                    global_valid=self.global_valid.clone(),
                    proto_scale=self.proto_scale,
                    lambda_s=self.lambda_s,
                    lambda_u=self.lambda_u,
                )
                for base in self.build_base_params(selected)
            ]

            train_results = self.run_clients(train, train_params)

            # 统计损失
            lx_sum = sum(train_results[cid]["loss_x_sum"] for cid in selected)
            lx_cnt = sum(train_results[cid]["loss_x_count"] for cid in selected)
            lcal_sum = sum(train_results[cid]["loss_calibrate_sum"] for cid in selected)
            lcal_cnt = sum(
                train_results[cid]["loss_calibrate_count"] for cid in selected
            )
            ltot_sum = sum(train_results[cid]["loss_total_sum"] for cid in selected)
            ltot_cnt = sum(train_results[cid]["loss_total_count"] for cid in selected)

            self.loss_x.append(lx_sum / max(1, lx_cnt))
            self.loss_calibrate.append(lcal_sum / max(1, lcal_cnt))
            self.loss.append(ltot_sum / max(1, ltot_cnt))

            # 2. FedAvg 聚合模型参数
            states = [train_results[cid]["state"] for cid in selected]
            weights = [self.weights[cid] for cid in selected]
            weight_sum = sum(weights)
            weights = [weight / weight_sum for weight in weights]
            self.aggregate(states, weights=weights)

            # 3. 提取特征并聚合统计原型
            proto_results = self.run_clients(
                estimate_prototypes_worker,
                self.build_base_params(selected),
            )
            global_model_protos = [proto_results[cid]["protos"] for cid in selected]
            global_model_counts = [
                proto_results[cid]["class_count"] for cid in selected
            ]
            mean_protos = proto_aggregate(
                global_model_protos,
                global_model_counts,
            )
            total_count = torch.stack(global_model_counts).sum(dim=0)
            self.mean_protos = mean_protos.cpu()
            self.mean_valid = total_count.gt(0).cpu()

            # 4. 优化目标原型并获取空间内在尺度
            self.global_protos = self._optimize_global_prototypes(
                self.mean_protos,
                self.mean_valid,
            )
            self.global_valid = self.mean_valid.clone()

            # 5. 距离分位数校准 (使用当前轮最新的全局模型与目标原型)
            cal_params = [
                ClientCalibrateParams(
                    **asdict(base),
                    global_protos=self.global_protos.clone(),
                    global_valid=self.global_valid.clone(),
                    proto_scale=self.proto_scale,
                )
                for base in self.build_base_params(selected)
            ]

            cal_results = self.run_clients(
                client_calibrate_worker,
                cal_params,
            )

            # 6. 服务器精确聚合各类 Q10、Q50 距离半径
            radius_q10 = torch.zeros(self.num_class)
            radius_q50 = torch.zeros(self.num_class)
            radius_valid = torch.zeros(self.num_class, dtype=torch.bool)
            for c in range(self.num_class):
                parts = [
                    cal_results[cid]["class_distances"][c]
                    for cid in selected
                    if cal_results[cid]["class_distances"][c].numel() > 0
                ]
                if parts:
                    d = torch.cat(parts)
                    radius_q10[c] = torch.quantile(d, 0.10)
                    radius_q50[c] = torch.quantile(d, 0.50)
                    radius_valid[c] = True
            self.radius_q10 = radius_q10
            self.radius_q50 = radius_q50
            self.radius_valid = radius_valid

            valid_idx = torch.where(radius_valid & self.global_valid)[0]
            pair_overlap10 = pair_overlap50 = 0.0
            pair_total = 0
            midpoint10 = []
            midpoint50 = []
            if valid_idx.numel() >= 2:
                proto_dist = mse_distance(
                    self.global_protos[valid_idx], self.global_protos[valid_idx]
                )
                for i in range(valid_idx.numel()):
                    for j in range(i + 1, valid_idx.numel()):
                        ci, cj = valid_idx[i], valid_idx[j]
                        dij = float(proto_dist[i, j])
                        if dij <= 0:
                            continue
                        pair_total += 1
                        if float(torch.sqrt(radius_q10[ci]) + torch.sqrt(radius_q10[cj])) >= dij**0.5:
                            pair_overlap10 += 1
                        if float(torch.sqrt(radius_q50[ci]) + torch.sqrt(radius_q50[cj])) >= dij**0.5:
                            pair_overlap50 += 1
                for i in range(valid_idx.numel()):
                    ci = valid_idx[i]
                    other = proto_dist[i].clone()
                    other[i] = float("inf")
                    nearest = other.min().clamp_min(1e-12)
                    midpoint10.append(float(4 * radius_q10[ci] / nearest))
                    midpoint50.append(float(4 * radius_q50[ci] / nearest))
            pair_overlap10 = 100.0 * pair_overlap10 / max(1, pair_total)
            # 5. 距离分位数校准 (使用当前轮最新的全局模型与目标原型)
            midpoint10_mean = sum(midpoint10) / max(1, len(midpoint10))
            midpoint50_mean = sum(midpoint50) / max(1, len(midpoint50))

            distance_summary = []
            for c in range(self.num_class):
                parts = [
                    cal_results[cid]["class_distances"][c]
                    for cid in selected
                    if cal_results[cid]["class_distances"][c].numel() > 0
                ]
                if parts:
                    d = torch.cat(parts)
                    q = torch.quantile(
                        d, torch.tensor([0.01, 0.10, 0.50, 0.90, 0.99, 1.0])
                    )
                    distance_summary.append((c, int(d.numel()), [float(v) for v in q]))
            inter_distance_summary = []
            nearest_inter_summary = []
            if valid_idx.numel() >= 2:
                inter_values = proto_dist[torch.triu(
                    torch.ones_like(proto_dist, dtype=torch.bool), diagonal=1
                )]
                inter_values = inter_values[torch.isfinite(inter_values) & (inter_values > 0)]
                if inter_values.numel() > 0:
                    iq = torch.quantile(
                        inter_values,
                        torch.tensor([0.01, 0.10, 0.50, 0.90, 0.99]),
                    )
                    inter_distance_summary = [float(v) for v in iq]
                for i, c in enumerate(valid_idx.tolist()):
                    other = proto_dist[i].clone()
                    other[i] = float("inf")
                    nearest_inter_summary.append((c, float(other.min())))

            # 6. 服务器精确聚合各类 Q10、Q50 距离半径
            diag_params = [
                ClientDiagnoseParams(
                    **asdict(base),
                    global_protos=self.global_protos.clone(),
                    global_valid=self.global_valid.clone(),
                    proto_scale=self.proto_scale,
                    radius_q10=self.radius_q10.clone(),
                    radius_q50=self.radius_q50.clone(),
                    radius_valid=self.radius_valid.clone(),
                )
                for base in self.build_base_params(selected)
            ]

            diag_results = self.run_clients(
                client_diagnose_worker,
                diag_params,
            )

            diag = torch.stack(
                [diag_results[cid]["diag_counts"] for cid in selected]
            ).sum(dim=0)

            total = int(diag[0])
            unlabeled_cls_acc = 100.0 * float(diag[1]) / max(1, total)
            unlabeled_prt_acc = 100.0 * float(diag[2]) / max(1, total)

            seen_total = int(diag[3])
            seen_cls_acc = 100.0 * float(diag[4]) / max(1, seen_total)
            seen_prt_acc = 100.0 * float(diag[5]) / max(1, seen_total)

            mis_total = int(diag[6])
            mis_cls_acc = 100.0 * float(diag[7]) / max(1, mis_total)
            mis_prt_acc = 100.0 * float(diag[8]) / max(1, mis_total)

            self.unlabeled_classifier_acc.append(unlabeled_cls_acc)
            self.unlabeled_proto_acc.append(unlabeled_prt_acc)
            self.seen_classifier_acc.append(seen_cls_acc)
            self.seen_proto_acc.append(seen_prt_acc)
            self.missing_classifier_acc.append(mis_cls_acc)
            self.missing_proto_acc.append(mis_prt_acc)

            cand_total = int(diag[9])
            if cand_total > 0:
                cand_cov = 100.0 * float(diag[10]) / cand_total
                avg_cand_size = float(diag[11]) / cand_total
                singleton_ratio = 100.0 * float(diag[12]) / cand_total
                singleton_acc = 100.0 * float(diag[13]) / max(1.0, float(diag[12]))
                empty_ratio = 100.0 * float(diag[14]) / cand_total
            else:
                cand_cov = avg_cand_size = singleton_ratio = singleton_acc = (
                    empty_ratio
                ) = 0.0

            self.cand_cov.append(cand_cov)
            self.avg_cand_size.append(avg_cand_size)
            self.singleton_ratio.append(singleton_ratio)
            self.singleton_acc.append(singleton_acc)
            self.empty_ratio.append(empty_ratio)

            mis_cand_total = int(diag[15])
            if mis_cand_total > 0:
                mis_cand_cov = 100.0 * float(diag[16]) / mis_cand_total
                mis_avg_cand_size = float(diag[17]) / mis_cand_total
                mis_singleton_ratio = 100.0 * float(diag[18]) / mis_cand_total
                mis_singleton_acc = 100.0 * float(diag[19]) / max(1.0, float(diag[18]))
                mis_empty_ratio = 100.0 * float(diag[20]) / mis_cand_total
            else:
                mis_cand_cov = mis_avg_cand_size = mis_singleton_ratio = (
                    mis_singleton_acc
                ) = mis_empty_ratio = 0.0

            self.missing_cand_cov.append(mis_cand_cov)
            self.missing_avg_cand_size.append(mis_avg_cand_size)
            self.missing_singleton_ratio.append(mis_singleton_ratio)
            self.missing_singleton_acc.append(mis_singleton_acc)
            self.missing_empty_ratio.append(mis_empty_ratio)
            high_hit = 100.0 * float(diag[21]) / max(1, total)
            high_cover = 100.0 * float(diag[22]) / max(1, total)
            high_hit_acc = 100.0 * float(diag[22]) / max(1.0, float(diag[21]))
            high_single = 100.0 * float(diag[23]) / max(1, total)
            high_single_acc = 100.0 * float(diag[24]) / max(1.0, float(diag[23]))
            high_multi = 100.0 * float(diag[25]) / max(1, total)
            high_multi_hit = 100.0 * float(diag[25]) / max(1.0, float(diag[21]))
            prob90_rate = 100.0 * float(diag[26]) / max(1, total)
            prob90_acc = (
                f"{100.0 * float(diag[27]) / float(diag[26]):.2f}%"
                if diag[26] > 0 else "N/A"
            )
            prob90_missing_rate = 100.0 * float(diag[28]) / max(1, mis_total)
            prob90_missing_acc = (
                f"{100.0 * float(diag[29]) / float(diag[28]):.2f}%"
                if diag[28] > 0 else "N/A"
            )
            cls95_rate = 100.0 * float(diag[30]) / max(1, total)
            cls95_acc = (
                f"{100.0 * float(diag[31]) / float(diag[30]):.2f}%"
                if diag[30] > 0 else "N/A"
            )
            cls95_missing_rate = 100.0 * float(diag[32]) / max(1, mis_total)
            cls95_missing_acc = (
                f"{100.0 * float(diag[33]) / float(diag[32]):.2f}%"
                if diag[32] > 0 else "N/A"
            )
            q50_text = (
                f"覆盖={cand_cov:.2f}% | 平均候选数={avg_cand_size:.2f} | "
                f"空集率={empty_ratio:.2f}% | 单类率={singleton_ratio:.2f}% | "
                f"单类准确率={singleton_acc:.2f}%"
            )
            distance_text = "\n".join(
                f"  c{c} (n={n}): q01={q[0]:.4g} | q10={q[1]:.4g} | "
                f"q50={q[2]:.4g} | q90={q[3]:.4g} | q99={q[4]:.4g} | q100={q[5]:.4g}"
                for c, n, q in distance_summary
            )
            inter_text = (
                "q01={:.4g},q10={:.4g},q50={:.4g},q90={:.4g},q99={:.4g}".format(
                    *inter_distance_summary
                )
                if inter_distance_summary
                else "无有效类间距离"
            )
            nearest_inter_text = "; ".join(
                f"c{c}:{d:.4g}" for c, d in nearest_inter_summary
            )
            intra_inter_ratio_text = ""
            nearest_map = dict(nearest_inter_summary)
            if nearest_map:
                intra_inter_ratio_text = "; ".join(
                    f"c{c}:{q[2] / nearest_map[c]:.3f}"
                    for c, _, q in distance_summary
                    if c in nearest_map and nearest_map[c] > 0
                )

            # 8. 测试集评估
            acc, acc_proto_mean, acc_proto = self._evaluate_accuracy()
            self.acc.append(acc)
            self.acc_proto_mean.append(acc_proto_mean)
            self.acc_proto.append(acc_proto)

            self.round_time.append(time.time() - round_start)

            # 9. 格式化控制台输出
            print(
                f"\n--- FedTest 第 {round_idx + 1}/{self.rounds} 轮 ---\n"
                f"测试准确率: 分类器={self.acc[-1]:.2f}% | 目标原型={self.acc_proto[-1]:.2f}% | 统计原型={self.acc_proto_mean[-1]:.2f}%\n"
                f"训练损失: CE={self.loss_x[-1]:.4f} | 原型校准={self.loss_calibrate[-1]:.4f} | 总损失={self.loss[-1]:.4f}\n"
                f"无标签分类: 分类器={unlabeled_cls_acc:.2f}% (常见={seen_cls_acc:.2f}%, 缺失={mis_cls_acc:.2f}%) | "
                f"原型={unlabeled_prt_acc:.2f}% (常见={seen_prt_acc:.2f}%, 缺失={mis_prt_acc:.2f}%)\n"
                f"无标签样本数: 总数={total} | 常见类={seen_total} | 缺失类={mis_total}\n"
                f"候选集质量: 真实覆盖率={cand_cov:.2f}% | 平均候选数={avg_cand_size:.2f} | 空集率={empty_ratio:.2f}% | "
                f"单候选率={singleton_ratio:.2f}% (准确率={singleton_acc:.2f}%)\n"
                f"缺失类候选: 真实覆盖率={mis_cand_cov:.2f}% | 平均候选数={mis_avg_cand_size:.2f} | 空集率={mis_empty_ratio:.2f}% | "
                f"单候选率={mis_singleton_ratio:.2f}% (准确率={mis_singleton_acc:.2f}%)\n"
                f"Q10: 命中率={high_hit:.2f}% | 命中样本真实准确率={high_hit_acc:.2f}% | "
                f"全体真实覆盖={high_cover:.2f}% | 单类率={high_single:.2f}% | "
                f"单类准确率={high_single_acc:.2f}% | 多类命中率(全体)={high_multi:.2f}% | "
                f"多类比例(Q10命中)={high_multi_hit:.2f}%\n"
                f"原型概率>90%: 样本率={prob90_rate:.2f}% | 准确率={prob90_acc} | "
                f"缺失类样本率={prob90_missing_rate:.2f}% | 缺失类准确率={prob90_missing_acc}\n"
                f"分类器概率>95%: 样本率={cls95_rate:.2f}% | 准确率={cls95_acc} | "
                f"缺失类样本率={cls95_missing_rate:.2f}% | 缺失类准确率={cls95_missing_acc}\n"
                f"Q50: {q50_text}\n"
                f"几何重叠: Q10={pair_overlap10:.2f}% | Q50={pair_overlap50:.2f}% | 中点比例均值(Q10/Q50)={midpoint10_mean:.4f}/{midpoint50_mean:.4f}\n"
                f"原型预测正确样本类内距离分布:\n{distance_text}\n"
                f"类间原型距离分布: {inter_text}\n"
                f"各类最近异类原型距离: {nearest_inter_text}\n"
                f"正确样本类内q50/最近类间距离: {intra_inter_ratio_text}\n"
                f"半径: Q10有效={int(self.radius_valid.sum())} | Q50有效={int(self.radius_valid.sum())}\n"
                f"本轮耗时: {self.round_time[-1]:.2f} 秒"
            )

    def save(self):
        metrics = {
            "acc": self.acc,
            "acc_proto": self.acc_proto,
            "acc_proto_mean": self.acc_proto_mean,
            "loss": self.loss,
            "loss_x": self.loss_x,
            "loss_calibrate": self.loss_calibrate,
            "round_time": self.round_time,
            "unlabeled_classifier_acc": self.unlabeled_classifier_acc,
            "unlabeled_proto_acc": self.unlabeled_proto_acc,
            "seen_classifier_acc": self.seen_classifier_acc,
            "seen_proto_acc": self.seen_proto_acc,
            "missing_classifier_acc": self.missing_classifier_acc,
            "missing_proto_acc": self.missing_proto_acc,
            "cand_cov": self.cand_cov,
            "avg_cand_size": self.avg_cand_size,
            "singleton_ratio": self.singleton_ratio,
            "singleton_acc": self.singleton_acc,
            "empty_ratio": self.empty_ratio,
            "missing_cand_cov": self.missing_cand_cov,
            "missing_avg_cand_size": self.missing_avg_cand_size,
            "missing_singleton_ratio": self.missing_singleton_ratio,
            "missing_singleton_acc": self.missing_singleton_acc,
            "missing_empty_ratio": self.missing_empty_ratio,
        }

        self.deal_save(
            metrics,
            {
                "global": self.model.state_dict(),
                "proto": self.global_protos,
                "proto_mean": self.mean_protos,
                "radius_q10": self.radius_q10,
                "radius_q50": self.radius_q50,
                "radius_valid": self.radius_valid,
            },
        )
