"""基于可靠原型的联邦半监督学习实验实现。

核心设计：
1. 分类器继续承担有监督训练与高置信伪标签训练；
2. 客户端从真实标签与可靠伪标签中提取类别原型，并估计逐类可靠度；
3. 服务器以可靠度加权中心为锚点，通过小型残差网络学习全局类别原型；
4. 全局原型将低置信样本划分为候选类别、未决类别和排除类别，
   通过集合概率监督利用传统 FixMatch 会丢弃的样本。

本文件只依赖项目已有的数据划分、增强、模型、Ray 客户端调度与参数聚合接口。
新增超参数全部提供默认值，因此第一版不要求修改 configs/algorithms.yaml。
"""

import os
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from .utils import BaseParams, BaseServer, clone_cpu_state, fmt_num, get_model
from .utils.augment import sage_strong_augment, sage_weak_augment
from .utils.ssl import build_fixmatch_loaders, iterate_ssl_batches


def _get(args, name, default):
    return getattr(args, name, default)


def get_path(args):
    proto_temperature = _get(args, "proto_temperature", 0.1)
    candidate_threshold = _get(args, "candidate_threshold", 0.5)
    exclude_threshold = _get(args, "exclude_threshold", 0.1)
    lambda_ambiguous = _get(args, "lambda_ambiguous", 1.0)
    args.file_name = (
        f"{args.common_name}"
        f"_pt{fmt_num(proto_temperature)}"
        f"_cp{fmt_num(candidate_threshold)}"
        f"_cn{fmt_num(exclude_threshold)}"
        f"_la{fmt_num(lambda_ambiguous)}"
    )
    return os.path.join(args.log_path, f"{args.file_name}_{args.cur_time}.log")


@dataclass
class Params(BaseParams):
    unlabeled_ratio: int
    lambda_u: float
    confidence: float

    round_index: int
    use_proto_guidance: bool
    ambiguous_weight: float

    global_protos: torch.Tensor | None
    global_valid: torch.Tensor | None
    previous_local_protos: torch.Tensor | None
    previous_local_valid: torch.Tensor | None

    proto_temperature: float
    candidate_threshold: float
    exclude_threshold: float
    lambda_ambiguous: float
    lambda_negative: float

    pseudo_proto_weight: float
    support_scale: float
    stability_temperature: float


class PrototypeResidualLearner(nn.Module):
    """在可靠度加权原型中心上学习残差修正，而不是从随机类别嵌入直接生成。"""

    def __init__(self, feature_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim),
        )
        # 初始时严格退化为可靠度加权中心，降低服务器训练初期的不稳定性。
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, base_protos):
        return F.normalize(base_protos + self.net(base_protos), dim=1)


def _prototype_probs(features, global_protos, global_valid, temperature):
    """计算样本与有效全局原型之间的归一化关系分布。"""
    if global_protos is None or global_valid is None:
        return None
    if not bool(global_valid.any().item()):
        return None

    protos = global_protos.to(features.device)
    valid = global_valid.to(features.device).bool()
    features = F.normalize(features, dim=1)
    protos = F.normalize(protos, dim=1)
    logits = features @ protos.t() / temperature
    logits = logits.masked_fill(~valid.unsqueeze(0), -1e4)
    probs = torch.softmax(logits, dim=1)
    probs = probs * valid.unsqueeze(0).float()
    return probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-12)


@torch.no_grad()
def _build_category_sets(
    proto_probs,
    global_valid,
    candidate_threshold,
    exclude_threshold,
):
    """根据相对原型关系构建候选、未决和排除类别集合。"""
    valid = global_valid.to(proto_probs.device).bool()
    max_prob = proto_probs.max(dim=1, keepdim=True).values
    relative = proto_probs / max_prob.clamp_min(1e-12)

    positive = (relative >= candidate_threshold) & valid.unsqueeze(0)
    negative = (relative <= exclude_threshold) & valid.unsqueeze(0)
    negative = negative & ~positive
    uncertain = valid.unsqueeze(0) & ~positive & ~negative
    return positive, uncertain, negative


@torch.no_grad()
def _select_high_confidence(
    classifier_probs,
    confidence_threshold,
    positive_mask=None,
    global_valid=None,
):
    """高置信分类器伪标签在已有原型类别上还需通过全局原型校验。"""
    confidence, pseudo_targets = classifier_probs.max(dim=1)
    selected = confidence.ge(confidence_threshold)

    if positive_mask is not None and global_valid is not None:
        row = torch.arange(classifier_probs.size(0), device=classifier_probs.device)
        valid = global_valid.to(classifier_probs.device).bool()
        predicted_proto_exists = valid[pseudo_targets]
        proto_agree = positive_mask[row, pseudo_targets]
        # 对尚未建立全局原型的类别不实施否决，避免训练早期误伤。
        selected = selected & (~predicted_proto_exists | proto_agree)

    return selected, pseudo_targets, confidence


def _ambiguous_set_loss(
    strong_features,
    global_protos,
    global_valid,
    positive_mask,
    negative_mask,
    ambiguous_mask,
    temperature,
    lambda_negative,
):
    """候选集合概率监督 + 高可信排除监督。"""
    zero = strong_features.sum() * 0.0
    if not bool(ambiguous_mask.any().item()):
        return zero, zero.detach(), zero.detach()

    proto_probs = _prototype_probs(
        strong_features,
        global_protos,
        global_valid,
        temperature,
    )
    if proto_probs is None:
        return zero, zero.detach(), zero.detach()

    probs = proto_probs[ambiguous_mask]
    positive = positive_mask[ambiguous_mask]
    negative = negative_mask[ambiguous_mask]

    positive_mass = (probs * positive.float()).sum(dim=1)
    valid_positive = positive.any(dim=1)
    if bool(valid_positive.any().item()):
        positive_loss = -torch.log(
            positive_mass[valid_positive].clamp_min(1e-12)
        ).mean()
    else:
        positive_loss = zero

    negative_count = negative.sum(dim=1)
    valid_negative = negative_count > 0
    if bool(valid_negative.any().item()):
        negative_term = (
            -torch.log((1.0 - probs).clamp_min(1e-12)) * negative.float()
        ).sum(dim=1)
        negative_loss = (
            negative_term[valid_negative]
            / negative_count[valid_negative].float()
        ).mean()
    else:
        negative_loss = zero

    total = positive_loss + lambda_negative * negative_loss
    return total, positive_loss.detach(), negative_loss.detach()


def _train_batch(
    model,
    x_l,
    y_l,
    x_u,
    y_u,
    p,
    device,
):
    """单个 FixMatch 式批次的前向、监督划分和损失计算。"""
    x_l = sage_weak_augment(x_l, p.dataset).to(device)
    y_l = y_l.to(device)
    y_u = y_u.to(device)
    x_u_w = sage_weak_augment(x_u, p.dataset).to(device)
    x_u_s = sage_strong_augment(x_u, p.dataset).to(device)

    batch_l = x_l.size(0)
    batch_u = x_u_w.size(0)
    inputs = torch.cat((x_l, x_u_w, x_u_s), dim=0)
    features = model.extractor(inputs)
    logits = model.classifier(features)

    logits_l = logits[:batch_l]
    logits_u_w = logits[batch_l : batch_l + batch_u]
    logits_u_s = logits[batch_l + batch_u :]
    feat_u_w = features[batch_l : batch_l + batch_u]
    feat_u_s = features[batch_l + batch_u :]

    loss_x = F.cross_entropy(logits_l, y_l)

    with torch.no_grad():
        classifier_probs = torch.softmax(logits_u_w.detach(), dim=1)

        positive_mask = None
        uncertain_mask = None
        negative_mask = None
        proto_probs = None

        if p.use_proto_guidance:
            proto_probs = _prototype_probs(
                feat_u_w.detach(),
                p.global_protos,
                p.global_valid,
                p.proto_temperature,
            )
            if proto_probs is not None:
                positive_mask, uncertain_mask, negative_mask = _build_category_sets(
                    proto_probs,
                    p.global_valid,
                    p.candidate_threshold,
                    p.exclude_threshold,
                )

        high_mask, pseudo_targets, pseudo_confidence = _select_high_confidence(
            classifier_probs,
            p.confidence,
            positive_mask,
            p.global_valid if positive_mask is not None else None,
        )

    loss_h_each = F.cross_entropy(logits_u_s, pseudo_targets, reduction="none")
    loss_h = (loss_h_each * high_mask.float()).mean()

    zero = logits_u_s.sum() * 0.0
    loss_amb = zero
    loss_pos = zero.detach()
    loss_neg = zero.detach()
    ambiguous_mask = ~high_mask

    if positive_mask is not None:
        loss_amb, loss_pos, loss_neg = _ambiguous_set_loss(
            feat_u_s,
            p.global_protos,
            p.global_valid,
            positive_mask,
            negative_mask,
            ambiguous_mask,
            p.proto_temperature,
            p.lambda_negative,
        )

    loss = (
        loss_x
        + p.lambda_u * loss_h
        + p.lambda_ambiguous * p.ambiguous_weight * loss_amb
    )

    # 以下均只用于实验诊断，y_u 不进入任何训练目标。
    with torch.no_grad():
        pseudo_correct = int(((pseudo_targets == y_u) & high_mask).sum().item())
        classifier_high = pseudo_confidence.ge(p.confidence)
        proto_rejected = int((classifier_high & ~high_mask).sum().item())

        candidate_eval = 0
        candidate_cover = 0
        candidate_size_sum = 0.0
        negative_false = 0
        negative_size_sum = 0.0
        if positive_mask is not None:
            valid = p.global_valid.to(device).bool()
            true_class_available = valid[y_u]
            row = torch.arange(batch_u, device=device)
            candidate_eval = int(true_class_available.sum().item())
            if candidate_eval > 0:
                candidate_cover = int(
                    (positive_mask[row, y_u] & true_class_available).sum().item()
                )
                negative_false = int(
                    (negative_mask[row, y_u] & true_class_available).sum().item()
                )
            candidate_size_sum = positive_mask.sum(dim=1).float().sum().item()
            negative_size_sum = negative_mask.sum(dim=1).float().sum().item()

    return loss, {
        "loss_x": loss_x.detach(),
        "loss_h": loss_h.detach(),
        "loss_amb": loss_amb.detach(),
        "loss_pos": loss_pos,
        "loss_neg": loss_neg,
        "pseudo_selected": int(high_mask.sum().item()),
        "pseudo_total": int(high_mask.numel()),
        "pseudo_confidence_sum": (pseudo_confidence * high_mask.float()).sum().item(),
        "pseudo_correct": pseudo_correct,
        "proto_rejected": proto_rejected,
        "candidate_eval": candidate_eval,
        "candidate_cover": candidate_cover,
        "candidate_size_sum": candidate_size_sum,
        "negative_false": negative_false,
        "negative_size_sum": negative_size_sum,
    }


@torch.no_grad()
def _extract_local_prototypes(model, p, device):
    """在本地训练结束后重新提取最终特征空间中的可靠类别原型。"""
    loader = DataLoader(p.train_set, batch_size=p.batch_size, shuffle=False)
    num_class = p.num_class
    feature_dim = p.feature_dim

    vector_sum = torch.zeros(num_class, feature_dim, device=device)
    weight_sum = torch.zeros(num_class, device=device)
    labeled_count = torch.zeros(num_class, device=device)
    pseudo_count = torch.zeros(num_class, device=device)
    pseudo_conf_sum = torch.zeros(num_class, device=device)

    pseudo_selected = 0
    pseudo_correct = 0
    true_unlabeled_total = 0
    offset = 0

    model.eval()
    for batch in loader:
        x, y, *_ = batch
        n = x.size(0)
        y = y.to(device)
        labeled_flag = p.train_set.is_labeled[offset : offset + n].to(device).bool()
        offset += n

        x_w = sage_weak_augment(x, p.dataset).to(device)
        features = model.extractor(x_w)
        logits = model.classifier(features)
        features = F.normalize(features, dim=1)

        # 真实标签原型作为可靠语义锚点。
        if bool(labeled_flag.any().item()):
            labels_l = y[labeled_flag]
            feat_l = features[labeled_flag]
            ones = torch.ones(labels_l.size(0), device=device)
            vector_sum.index_add_(0, labels_l, feat_l)
            weight_sum.index_add_(0, labels_l, ones)
            labeled_count += torch.bincount(labels_l, minlength=num_class).float()

        # 真实无标签样本仅在高置信且通过原型校验时参与局部原型。
        unlabeled_flag = ~labeled_flag
        true_unlabeled_total += int(unlabeled_flag.sum().item())
        if not bool(unlabeled_flag.any().item()):
            continue

        feat_u = features[unlabeled_flag]
        logits_u = logits[unlabeled_flag]
        y_u = y[unlabeled_flag]
        classifier_probs = torch.softmax(logits_u, dim=1)

        positive_mask = None
        if p.use_proto_guidance:
            proto_probs = _prototype_probs(
                feat_u,
                p.global_protos,
                p.global_valid,
                p.proto_temperature,
            )
            if proto_probs is not None:
                positive_mask, _, _ = _build_category_sets(
                    proto_probs,
                    p.global_valid,
                    p.candidate_threshold,
                    p.exclude_threshold,
                )

        high_mask, pseudo_targets, confidence = _select_high_confidence(
            classifier_probs,
            p.confidence,
            positive_mask,
            p.global_valid if positive_mask is not None else None,
        )

        if not bool(high_mask.any().item()):
            continue

        selected_feat = feat_u[high_mask]
        selected_targets = pseudo_targets[high_mask]
        selected_conf = confidence[high_mask]
        selected_weight = p.pseudo_proto_weight * selected_conf

        vector_sum.index_add_(
            0,
            selected_targets,
            selected_feat * selected_weight.unsqueeze(1),
        )
        weight_sum.index_add_(0, selected_targets, selected_weight)
        pseudo_count += torch.bincount(
            selected_targets, minlength=num_class
        ).float()
        pseudo_conf_sum.index_add_(0, selected_targets, selected_conf)

        pseudo_selected += int(high_mask.sum().item())
        pseudo_correct += int(
            (selected_targets == y_u[high_mask]).sum().item()
        )

    valid = weight_sum > 0
    protos = torch.zeros_like(vector_sum)
    protos[valid] = F.normalize(vector_sum[valid], dim=1)

    # 对单位特征而言，归一化前加权和的长度直接反映类内方向一致性。
    compactness = torch.zeros(num_class, device=device)
    compactness[valid] = (
        vector_sum[valid].norm(dim=1) / weight_sum[valid].clamp_min(1e-12)
    ).clamp(0.0, 1.0)

    support = 1.0 - torch.exp(-weight_sum / p.support_scale)

    pseudo_quality = torch.ones(num_class, device=device)
    has_pseudo = pseudo_count > 0
    pseudo_quality[has_pseudo] = (
        pseudo_conf_sum[has_pseudo] / pseudo_count[has_pseudo]
    ).clamp(0.0, 1.0)

    stability = torch.ones(num_class, device=device)
    if p.previous_local_protos is not None and p.previous_local_valid is not None:
        previous_protos = p.previous_local_protos.to(device)
        previous_valid = p.previous_local_valid.to(device).bool()
        common = valid & previous_valid
        if bool(common.any().item()):
            similarity = F.cosine_similarity(
                protos[common], previous_protos[common], dim=1
            )
            stability[common] = torch.exp(
                -(1.0 - similarity) / p.stability_temperature
            ).clamp(0.0, 1.0)

    reliability = (
        support * compactness * pseudo_quality * stability
    ).clamp_min(1e-12).pow(0.25)
    reliability[~valid] = 0.0

    return {
        "protos": protos.cpu(),
        "valid": valid.cpu(),
        "reliability": reliability.cpu(),
        "labeled_count": labeled_count.cpu(),
        "pseudo_count": pseudo_count.cpu(),
        "compactness": compactness.cpu(),
        "support": support.cpu(),
        "pseudo_quality": pseudo_quality.cpu(),
        "stability": stability.cpu(),
        "proto_pseudo_selected": pseudo_selected,
        "proto_pseudo_correct": pseudo_correct,
        "proto_unlabeled_total": true_unlabeled_total,
    }


def train(p: Params):
    device = torch.device(p.client_gpu)
    model = get_model(p.model_name, p.dataset, p.num_class, p.feature_dim).to(device)
    model.load_state_dict(p.model_state)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=p.lr,
        momentum=p.momentum,
        weight_decay=p.weight_decay,
    )
    loaders = build_fixmatch_loaders(
        p.train_set,
        p.batch_size,
        p.unlabeled_ratio,
    )

    loss_sum = 0.0
    loss_x_sum = 0.0
    loss_h_sum = 0.0
    loss_amb_sum = 0.0
    loss_pos_sum = 0.0
    loss_neg_sum = 0.0
    loss_x_count = 0
    loss_u_count = 0

    pseudo_selected = 0
    pseudo_total = 0
    pseudo_correct = 0
    pseudo_confidence_sum = 0.0
    proto_rejected = 0

    candidate_eval = 0
    candidate_cover = 0
    candidate_size_sum = 0.0
    negative_false = 0
    negative_size_sum = 0.0
    candidate_sample_count = 0
    steps = 0

    model.train()
    for _ in range(p.epochs):
        for labeled, (x_u, y_u) in iterate_ssl_batches(loaders):
            if labeled is None:
                raise ValueError(
                    "fedproto_ssl 当前版本要求每个客户端至少具有少量有标签数据。"
                )
            x_l, y_l = labeled

            loss, info = _train_batch(
                model,
                x_l,
                y_l,
                x_u,
                y_u,
                p,
                device,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_l = x_l.size(0)
            batch_u = x_u.size(0)
            loss_sum += loss.item()
            loss_x_sum += info["loss_x"].item() * batch_l
            loss_h_sum += info["loss_h"].item() * batch_u
            loss_amb_sum += info["loss_amb"].item() * batch_u
            loss_pos_sum += info["loss_pos"].item() * batch_u
            loss_neg_sum += info["loss_neg"].item() * batch_u
            loss_x_count += batch_l
            loss_u_count += batch_u

            pseudo_selected += info["pseudo_selected"]
            pseudo_total += info["pseudo_total"]
            pseudo_correct += info["pseudo_correct"]
            pseudo_confidence_sum += info["pseudo_confidence_sum"]
            proto_rejected += info["proto_rejected"]

            candidate_eval += info["candidate_eval"]
            candidate_cover += info["candidate_cover"]
            candidate_size_sum += info["candidate_size_sum"]
            negative_false += info["negative_false"]
            negative_size_sum += info["negative_size_sum"]
            if p.use_proto_guidance:
                candidate_sample_count += batch_u
            steps += 1

    local_proto = _extract_local_prototypes(model, p, device)

    return {
        "state": clone_cpu_state(model.state_dict()),
        "loss": loss_sum / max(1, steps),
        "loss_x_sum": loss_x_sum,
        "loss_h_sum": loss_h_sum,
        "loss_amb_sum": loss_amb_sum,
        "loss_pos_sum": loss_pos_sum,
        "loss_neg_sum": loss_neg_sum,
        "loss_x_count": loss_x_count,
        "loss_u_count": loss_u_count,
        "pseudo_selected": pseudo_selected,
        "pseudo_total": pseudo_total,
        "pseudo_correct": pseudo_correct,
        "pseudo_confidence_sum": pseudo_confidence_sum,
        "proto_rejected": proto_rejected,
        "candidate_eval": candidate_eval,
        "candidate_cover": candidate_cover,
        "candidate_size_sum": candidate_size_sum,
        "negative_false": negative_false,
        "negative_size_sum": negative_size_sum,
        "candidate_sample_count": candidate_sample_count,
        **local_proto,
    }


class Server(BaseServer):
    def __init__(self, args):
        if args.ssl in ("none", "client"):
            raise ValueError(
                "fedproto_ssl 面向每客户端同时具有监督与无监督数据的场景，"
                "请使用 sample、double 或 sfd。"
            )
        super().__init__(args, is_ssl=True, pfl=False)

        self.unlabeled_ratio = args.unlabeled_ratio

        # 客户端原型监督参数。
        self.proto_temperature = float(_get(args, "proto_temperature", 0.1))
        self.candidate_threshold = float(_get(args, "candidate_threshold", 0.5))
        self.exclude_threshold = float(_get(args, "exclude_threshold", 0.1))
        self.lambda_ambiguous = float(_get(args, "lambda_ambiguous", 1.0))
        self.lambda_negative = float(_get(args, "lambda_negative", 0.5))
        self.pseudo_proto_weight = float(_get(args, "pseudo_proto_weight", 0.5))
        self.support_scale = float(_get(args, "proto_support_scale", 10.0))
        self.stability_temperature = float(
            _get(args, "proto_stability_temperature", 0.1)
        )

        # 原型指导先预热，再线性开启低置信集合监督。
        self.proto_warmup_rounds = int(_get(args, "proto_warmup_rounds", 5))
        self.proto_ramp_rounds = int(_get(args, "proto_ramp_rounds", 5))

        # 服务器可学习原型参数。
        self.server_epochs = int(_get(args, "server_epochs", 30))
        self.server_lr = float(_get(args, "server_lr", 0.01))
        self.server_separation_weight = float(
            _get(args, "server_separation_weight", 0.1)
        )
        self.server_margin = float(_get(args, "server_margin", 0.2))
        self.server_max_momentum = float(_get(args, "server_max_momentum", 0.5))
        self.server_support_scale = float(
            _get(args, "server_proto_support_scale", 5.0)
        )

        self.proto_learner = PrototypeResidualLearner(self.feature_dim).to(self.device)
        self.proto_optimizer = torch.optim.SGD(
            self.proto_learner.parameters(), lr=self.server_lr
        )

        self.global_protos = torch.zeros(self.num_class, self.feature_dim)
        self.global_valid = torch.zeros(self.num_class, dtype=torch.bool)
        self.previous_local_protos = [None for _ in range(self.num_clients)]
        self.previous_local_valid = [None for _ in range(self.num_clients)]

        # 训练与诊断指标。
        self.loss_x = []
        self.loss_h = []
        self.loss_amb = []
        self.loss_pos = []
        self.loss_neg = []
        self.loss_server = []
        self.loss_server_align = []
        self.loss_server_sep = []

        self.pseudo_coverage = []
        self.pseudo_accuracy = []
        self.pseudo_confidence = []
        self.proto_reject_ratio = []
        self.candidate_coverage = []
        self.candidate_size = []
        self.negative_false_rate = []
        self.negative_size = []
        self.valid_proto_count = []
        self.mean_proto_reliability = []
        self.proto_build_pseudo_acc = []
        self.round_time = []

        for dataset in self.train_sets.values():
            if dataset.is_labeled is None:
                raise ValueError("fedproto_ssl 需要半监督数据中的 is_labeled 字段。")

    def _guidance_state(self, round_index):
        use_proto = (
            round_index >= self.proto_warmup_rounds
            and bool(self.global_valid.any().item())
        )
        if not use_proto:
            return False, 0.0
        if self.proto_ramp_rounds <= 0:
            return True, 1.0
        progress = (round_index - self.proto_warmup_rounds + 1) / self.proto_ramp_rounds
        return True, min(1.0, max(0.0, progress))

    @staticmethod
    def _normalize_selected_weights(weights):
        total = sum(weights)
        return [w / total for w in weights]

    def _server_proto_update(self, results, selected):
        """可靠度加权中心 -> 残差学习 -> 可靠支持量控制的跨轮记忆。"""
        weighted_sum = torch.zeros(
            self.num_class, self.feature_dim, device=self.device
        )
        total_weight = torch.zeros(self.num_class, device=self.device)

        client_data = []
        for cid in selected:
            protos = results[cid]["protos"].to(self.device)
            reliability = results[cid]["reliability"].to(self.device)
            valid = results[cid]["valid"].to(self.device).bool()
            weight = reliability * valid.float()
            weighted_sum += protos * weight.unsqueeze(1)
            total_weight += weight
            client_data.append((protos, reliability, valid))

        current_valid = total_weight > 0
        if not bool(current_valid.any().item()):
            self.loss_server.append(0.0)
            self.loss_server_align.append(0.0)
            self.loss_server_sep.append(0.0)
            return

        base = torch.zeros_like(weighted_sum)
        base[current_valid] = F.normalize(
            weighted_sum[current_valid]
            / total_weight[current_valid].unsqueeze(1).clamp_min(1e-12),
            dim=1,
        )

        last_loss = 0.0
        last_align = 0.0
        last_sep = 0.0
        self.proto_learner.train()
        for _ in range(self.server_epochs):
            generated = self.proto_learner(base)

            align_num = generated.sum() * 0.0
            align_den = torch.tensor(0.0, device=self.device)
            for local_protos, reliability, valid in client_data:
                mask = valid & current_valid
                if not bool(mask.any().item()):
                    continue
                similarity = F.cosine_similarity(
                    generated[mask], local_protos[mask], dim=1
                )
                weight = reliability[mask]
                align_num = align_num + (weight * (1.0 - similarity)).sum()
                align_den = align_den + weight.sum()
            loss_align = align_num / align_den.clamp_min(1e-12)

            valid_generated = generated[current_valid]
            if valid_generated.size(0) > 1:
                sim_matrix = valid_generated @ valid_generated.t()
                off_diag = ~torch.eye(
                    valid_generated.size(0),
                    dtype=torch.bool,
                    device=self.device,
                )
                loss_sep = F.relu(
                    sim_matrix[off_diag] - self.server_margin
                ).mean()
            else:
                loss_sep = generated.sum() * 0.0

            loss = loss_align + self.server_separation_weight * loss_sep
            self.proto_optimizer.zero_grad()
            loss.backward()
            self.proto_optimizer.step()

            last_loss = loss.item()
            last_align = loss_align.item()
            last_sep = loss_sep.item()

        self.proto_learner.eval()
        with torch.no_grad():
            learned = self.proto_learner(base).cpu()
            total_weight_cpu = total_weight.cpu()
            current_valid_cpu = current_valid.cpu()

            for c in range(self.num_class):
                if not bool(current_valid_cpu[c].item()):
                    continue
                if not bool(self.global_valid[c].item()):
                    self.global_protos[c] = learned[c]
                    self.global_valid[c] = True
                    continue

                support_factor = min(
                    1.0,
                    total_weight_cpu[c].item() / self.server_support_scale,
                )
                update_ratio = self.server_max_momentum * support_factor
                updated = (
                    (1.0 - update_ratio) * self.global_protos[c]
                    + update_ratio * learned[c]
                )
                self.global_protos[c] = F.normalize(updated.unsqueeze(0), dim=1)[0]

        self.loss_server.append(last_loss)
        self.loss_server_align.append(last_align)
        self.loss_server_sep.append(last_sep)

    def fit(self):
        num_join = max(1, int(self.num_clients * self.join_ratio))

        for round_index in range(self.rounds):
            started = time.time()
            selected = sorted(torch.randperm(self.num_clients)[:num_join].tolist())
            use_proto, ambiguous_weight = self._guidance_state(round_index)

            print(f"\n--- FedProto-SSL Round {round_index + 1}/{self.rounds} ---")
            print(
                f" selected_clients={selected} | proto_guidance={use_proto} | "
                f"ambiguous_weight={ambiguous_weight:.3f}"
            )

            parameters = []
            for base in self.build_base_params(selected):
                cid = base.client_id
                parameters.append(
                    Params(
                        **asdict(base),
                        unlabeled_ratio=self.unlabeled_ratio,
                        lambda_u=self.lam,
                        confidence=self.confidence,
                        round_index=round_index,
                        use_proto_guidance=use_proto,
                        ambiguous_weight=ambiguous_weight,
                        global_protos=self.global_protos.clone()
                        if bool(self.global_valid.any().item())
                        else None,
                        global_valid=self.global_valid.clone()
                        if bool(self.global_valid.any().item())
                        else None,
                        previous_local_protos=self.previous_local_protos[cid],
                        previous_local_valid=self.previous_local_valid[cid],
                        proto_temperature=self.proto_temperature,
                        candidate_threshold=self.candidate_threshold,
                        exclude_threshold=self.exclude_threshold,
                        lambda_ambiguous=self.lambda_ambiguous,
                        lambda_negative=self.lambda_negative,
                        pseudo_proto_weight=self.pseudo_proto_weight,
                        support_scale=self.support_scale,
                        stability_temperature=self.stability_temperature,
                    )
                )

            results = self.run_clients(train, parameters)

            states = [results[cid]["state"] for cid in selected]
            weights = [self.weights[cid] for cid in selected]
            self.aggregate(states, weights=self._normalize_selected_weights(weights))

            for cid in selected:
                self.previous_local_protos[cid] = results[cid]["protos"].clone()
                self.previous_local_valid[cid] = results[cid]["valid"].clone()

            self._server_proto_update(results, selected)

            x_count = sum(results[cid]["loss_x_count"] for cid in selected)
            u_count = sum(results[cid]["loss_u_count"] for cid in selected)
            self.loss_x.append(
                sum(results[cid]["loss_x_sum"] for cid in selected) / max(1, x_count)
            )
            self.loss_h.append(
                sum(results[cid]["loss_h_sum"] for cid in selected) / max(1, u_count)
            )
            self.loss_amb.append(
                sum(results[cid]["loss_amb_sum"] for cid in selected) / max(1, u_count)
            )
            self.loss_pos.append(
                sum(results[cid]["loss_pos_sum"] for cid in selected) / max(1, u_count)
            )
            self.loss_neg.append(
                sum(results[cid]["loss_neg_sum"] for cid in selected) / max(1, u_count)
            )
            self.loss.append(
                sum(results[cid]["loss"] for cid in selected) / len(selected)
            )

            pseudo_selected = sum(results[cid]["pseudo_selected"] for cid in selected)
            pseudo_total = sum(results[cid]["pseudo_total"] for cid in selected)
            pseudo_correct = sum(results[cid]["pseudo_correct"] for cid in selected)
            pseudo_conf_sum = sum(
                results[cid]["pseudo_confidence_sum"] for cid in selected
            )
            proto_rejected = sum(results[cid]["proto_rejected"] for cid in selected)

            self.pseudo_coverage.append(
                100.0 * pseudo_selected / max(1, pseudo_total)
            )
            self.pseudo_accuracy.append(
                100.0 * pseudo_correct / max(1, pseudo_selected)
            )
            self.pseudo_confidence.append(
                pseudo_conf_sum / max(1, pseudo_selected)
            )
            self.proto_reject_ratio.append(
                100.0 * proto_rejected / max(1, pseudo_total)
            )

            candidate_eval = sum(results[cid]["candidate_eval"] for cid in selected)
            candidate_cover = sum(results[cid]["candidate_cover"] for cid in selected)
            candidate_samples = sum(
                results[cid]["candidate_sample_count"] for cid in selected
            )
            negative_false = sum(results[cid]["negative_false"] for cid in selected)
            self.candidate_coverage.append(
                100.0 * candidate_cover / max(1, candidate_eval)
                if candidate_eval > 0
                else 0.0
            )
            self.candidate_size.append(
                sum(results[cid]["candidate_size_sum"] for cid in selected)
                / max(1, candidate_samples)
            )
            self.negative_false_rate.append(
                100.0 * negative_false / max(1, candidate_eval)
                if candidate_eval > 0
                else 0.0
            )
            self.negative_size.append(
                sum(results[cid]["negative_size_sum"] for cid in selected)
                / max(1, candidate_samples)
            )

            reliabilities = []
            valid_count = 0
            proto_build_selected = 0
            proto_build_correct = 0
            for cid in selected:
                valid = results[cid]["valid"].bool()
                valid_count += int(valid.sum().item())
                if bool(valid.any().item()):
                    reliabilities.append(results[cid]["reliability"][valid])
                proto_build_selected += results[cid]["proto_pseudo_selected"]
                proto_build_correct += results[cid]["proto_pseudo_correct"]

            self.valid_proto_count.append(valid_count)
            self.mean_proto_reliability.append(
                torch.cat(reliabilities).mean().item() if reliabilities else 0.0
            )
            self.proto_build_pseudo_acc.append(
                100.0 * proto_build_correct / max(1, proto_build_selected)
            )

            if bool(self.global_valid.all().item()):
                self.evaluate(protos=self.global_protos)
                proto_acc_text = f" | Proto Acc: {self.acc_proto[-1]:.2f}%"
            else:
                self.evaluate()
                proto_acc_text = ""

            self.round_time.append(time.time() - started)
            print(
                f"Accuracy: {self.acc[-1]:.2f}%{proto_acc_text} | "
                f"Pseudo: coverage={self.pseudo_coverage[-1]:.2f}%, "
                f"acc={self.pseudo_accuracy[-1]:.2f}%"
            )
            if use_proto:
                print(
                    f"Candidate: cover={self.candidate_coverage[-1]:.2f}%, "
                    f"size={self.candidate_size[-1]:.2f} | "
                    f"Negative: false={self.negative_false_rate[-1]:.2f}%, "
                    f"size={self.negative_size[-1]:.2f}"
                )
            print(
                f"Local protos: valid={valid_count}, "
                f"mean reliability={self.mean_proto_reliability[-1]:.3f} | "
                f"Server loss={self.loss_server[-1]:.4f}"
            )
            print(f"Round finished in {self.round_time[-1]:.2f} seconds")

    def save(self):
        metrics = {
            "acc": self.acc,
            "acc_proto": self.acc_proto,
            "loss": self.loss,
            "loss_x": self.loss_x,
            "loss_h": self.loss_h,
            "loss_amb": self.loss_amb,
            "loss_pos": self.loss_pos,
            "loss_neg": self.loss_neg,
            "loss_server": self.loss_server,
            "loss_server_align": self.loss_server_align,
            "loss_server_sep": self.loss_server_sep,
            "pseudo_coverage": self.pseudo_coverage,
            "pseudo_accuracy": self.pseudo_accuracy,
            "pseudo_confidence": self.pseudo_confidence,
            "proto_reject_ratio": self.proto_reject_ratio,
            "candidate_coverage": self.candidate_coverage,
            "candidate_size": self.candidate_size,
            "negative_false_rate": self.negative_false_rate,
            "negative_size": self.negative_size,
            "valid_proto_count": self.valid_proto_count,
            "mean_proto_reliability": self.mean_proto_reliability,
            "proto_build_pseudo_acc": self.proto_build_pseudo_acc,
            "round_time": self.round_time,
        }
        params = {
            "global": self.model.state_dict(),
            "proto": self.global_protos,
            "aux": {
                "global_valid": self.global_valid,
                "proto_learner": self.proto_learner.state_dict(),
            },
        }
        self.deal_save(metrics, params)
