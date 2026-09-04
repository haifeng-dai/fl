"""可靠原型驱动的联邦半监督学习。

训练目标保持简洁：分类器负责监督/高置信伪标签，原型负责跨客户端类别知识与
低置信集合监督。隐藏真实标签只用于诊断，不参与任何训练目标。
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
    pt = _get(args, "proto_temperature", 0.1)
    cp = _get(args, "candidate_threshold", 0.5)
    cn = _get(args, "exclude_threshold", 0.1)
    la = _get(args, "lambda_ambiguous", 1.0)
    pp = _get(args, "pseudo_proto_weight", 0.1)
    args.file_name = (
        f"{args.common_name}_pt{fmt_num(pt)}_cp{fmt_num(cp)}_cn{fmt_num(cn)}"
        f"_la{fmt_num(la)}_pp{fmt_num(pp)}"
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
    """以可靠度加权中心为锚点学习全局原型残差。"""

    def __init__(self, feature_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, base_protos):
        return F.normalize(base_protos + self.net(base_protos), dim=1)


def _prototype_probs(features, global_protos, global_valid, temperature):
    if global_protos is None or global_valid is None:
        return None
    if not bool(global_valid.any().item()):
        return None
    valid = global_valid.to(features.device).bool()
    protos = F.normalize(global_protos.to(features.device), dim=1)
    features = F.normalize(features, dim=1)
    logits = features @ protos.t() / temperature
    logits = logits.masked_fill(~valid.unsqueeze(0), -1e4)
    probs = torch.softmax(logits, dim=1) * valid.unsqueeze(0).float()
    return probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-12)


@torch.no_grad()
def _build_category_sets(proto_probs, global_valid, pos_threshold, neg_threshold):
    valid = global_valid.to(proto_probs.device).bool()
    relative = proto_probs / proto_probs.max(dim=1, keepdim=True).values.clamp_min(
        1e-12
    )
    positive = (relative >= pos_threshold) & valid.unsqueeze(0)
    negative = (relative <= neg_threshold) & valid.unsqueeze(0)
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
    confidence, pseudo_targets = classifier_probs.max(dim=1)
    selected = confidence.ge(confidence_threshold)
    if positive_mask is not None and global_valid is not None:
        row = torch.arange(classifier_probs.size(0), device=classifier_probs.device)
        valid = global_valid.to(classifier_probs.device).bool()
        proto_exists = valid[pseudo_targets]
        proto_agree = positive_mask[row, pseudo_targets]
        selected = selected & (~proto_exists | proto_agree)
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
    zero = strong_features.sum() * 0.0
    if not bool(ambiguous_mask.any().item()):
        return zero, zero.detach(), zero.detach()
    probs = _prototype_probs(
        strong_features, global_protos, global_valid, temperature
    )
    if probs is None:
        return zero, zero.detach(), zero.detach()
    probs = probs[ambiguous_mask]
    positive = positive_mask[ambiguous_mask]
    negative = negative_mask[ambiguous_mask]

    positive_mass = (probs * positive.float()).sum(dim=1)
    valid_positive = positive.any(dim=1)
    positive_loss = zero
    if bool(valid_positive.any().item()):
        positive_loss = -torch.log(
            positive_mass[valid_positive].clamp_min(1e-12)
        ).mean()

    negative_count = negative.sum(dim=1)
    valid_negative = negative_count > 0
    negative_loss = zero
    if bool(valid_negative.any().item()):
        negative_term = (
            -torch.log((1.0 - probs).clamp_min(1e-12)) * negative.float()
        ).sum(dim=1)
        negative_loss = (
            negative_term[valid_negative] / negative_count[valid_negative].float()
        ).mean()
    return (
        positive_loss + lambda_negative * negative_loss,
        positive_loss.detach(),
        negative_loss.detach(),
    )


@torch.no_grad()
def _dynamic_topk_hit(classifier_probs, targets, k_per_sample):
    """使用每个样本的候选集合大小作为 k，得到分类器自身 Top-k 基线。"""
    num_classes = classifier_probs.size(1)
    ranking = classifier_probs.argsort(dim=1, descending=True)
    ranks = torch.empty_like(ranking)
    rank_values = torch.arange(num_classes, device=classifier_probs.device)
    rank_values = rank_values.unsqueeze(0).expand_as(ranking)
    ranks.scatter_(1, ranking, rank_values)
    row = torch.arange(classifier_probs.size(0), device=classifier_probs.device)
    true_rank = ranks[row, targets]
    return true_rank < k_per_sample.clamp(min=1, max=num_classes)


def _train_batch(model, x_l, y_l, x_u, y_u, p, device):
    x_l = sage_weak_augment(x_l, p.dataset).to(device)
    y_l = y_l.to(device)
    y_u = y_u.to(device)
    x_u_w = sage_weak_augment(x_u, p.dataset).to(device)
    x_u_s = sage_strong_augment(x_u, p.dataset).to(device)

    batch_l = x_l.size(0)
    batch_u = x_u_w.size(0)
    features = model.extractor(torch.cat((x_l, x_u_w, x_u_s), dim=0))
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
        negative_mask = None
        if p.use_proto_guidance:
            proto_probs = _prototype_probs(
                feat_u_w.detach(),
                p.global_protos,
                p.global_valid,
                p.proto_temperature,
            )
            if proto_probs is not None:
                positive_mask, _, negative_mask = _build_category_sets(
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

    loss_h = (
        F.cross_entropy(logits_u_s, pseudo_targets, reduction="none")
        * high_mask.float()
    ).mean()
    zero = logits_u_s.sum() * 0.0
    loss_amb = zero
    loss_pos = zero.detach()
    loss_neg = zero.detach()
    if positive_mask is not None:
        loss_amb, loss_pos, loss_neg = _ambiguous_set_loss(
            feat_u_s,
            p.global_protos,
            p.global_valid,
            positive_mask,
            negative_mask,
            ~high_mask,
            p.proto_temperature,
            p.lambda_negative,
        )
    loss = (
        loss_x
        + p.lambda_u * loss_h
        + p.lambda_ambiguous * p.ambiguous_weight * loss_amb
    )

    # 仅诊断：y_u 不参与训练。
    with torch.no_grad():
        classifier_high = pseudo_confidence.ge(p.confidence)
        classifier_low = ~classifier_high
        rejected = classifier_high & ~high_mask
        raw_high_count = int(classifier_high.sum().item())
        raw_high_correct = int(
            ((pseudo_targets == y_u) & classifier_high).sum().item()
        )
        pseudo_correct = int(((pseudo_targets == y_u) & high_mask).sum().item())
        rejected_count = int(rejected.sum().item())
        rejected_wrong = int(((pseudo_targets != y_u) & rejected).sum().item())

        candidate_eval = candidate_cover = negative_false = 0
        candidate_size_sum = negative_size_sum = 0.0
        low_eval = low_cover = low_negative_false = low_topk_cover = 0
        low_candidate_size_sum = low_negative_size_sum = 0.0
        low_sample_count = 0

        if positive_mask is not None:
            valid = p.global_valid.to(device).bool()
            true_available = valid[y_u]
            row = torch.arange(batch_u, device=device)
            pos_sizes = positive_mask.sum(dim=1)
            neg_sizes = negative_mask.sum(dim=1)

            candidate_eval = int(true_available.sum().item())
            candidate_cover = int(
                (positive_mask[row, y_u] & true_available).sum().item()
            )
            negative_false = int(
                (negative_mask[row, y_u] & true_available).sum().item()
            )
            candidate_size_sum = pos_sizes.float().sum().item()
            negative_size_sum = neg_sizes.float().sum().item()

            low_sample_count = int(classifier_low.sum().item())
            low_mask = classifier_low & true_available
            low_eval = int(low_mask.sum().item())
            low_cover = int((positive_mask[row, y_u] & low_mask).sum().item())
            low_negative_false = int(
                (negative_mask[row, y_u] & low_mask).sum().item()
            )
            low_candidate_size_sum = pos_sizes[classifier_low].float().sum().item()
            low_negative_size_sum = neg_sizes[classifier_low].float().sum().item()
            topk_hit = _dynamic_topk_hit(classifier_probs, y_u, pos_sizes)
            low_topk_cover = int((topk_hit & low_mask).sum().item())

    return loss, {
        "loss_x": loss_x.detach(),
        "loss_h": loss_h.detach(),
        "loss_amb": loss_amb.detach(),
        "loss_pos": loss_pos,
        "loss_neg": loss_neg,
        "pseudo_selected": int(high_mask.sum().item()),
        "pseudo_total": int(high_mask.numel()),
        "pseudo_correct": pseudo_correct,
        "pseudo_confidence_sum": (
            pseudo_confidence * high_mask.float()
        ).sum().item(),
        "raw_high_count": raw_high_count,
        "raw_high_correct": raw_high_correct,
        "proto_rejected": rejected_count,
        "proto_rejected_wrong": rejected_wrong,
        "candidate_eval": candidate_eval,
        "candidate_cover": candidate_cover,
        "candidate_size_sum": candidate_size_sum,
        "negative_false": negative_false,
        "negative_size_sum": negative_size_sum,
        "low_candidate_eval": low_eval,
        "low_candidate_cover": low_cover,
        "low_candidate_size_sum": low_candidate_size_sum,
        "low_negative_false": low_negative_false,
        "low_negative_size_sum": low_negative_size_sum,
        "low_topk_cover": low_topk_cover,
        "low_sample_count": low_sample_count,
    }


@torch.no_grad()
def _extract_local_prototypes(model, p, device):
    """提取监督/混合局部原型；隐藏真值原型仅用于可靠度诊断。"""
    loader = DataLoader(p.train_set, batch_size=p.batch_size, shuffle=False)
    c_num, d_num = p.num_class, p.feature_dim
    mixed_sum = torch.zeros(c_num, d_num, device=device)
    mixed_weight = torch.zeros(c_num, device=device)
    supervised_sum = torch.zeros(c_num, d_num, device=device)
    supervised_count = torch.zeros(c_num, device=device)
    oracle_sum = torch.zeros(c_num, d_num, device=device)
    oracle_count = torch.zeros(c_num, device=device)
    pseudo_count = torch.zeros(c_num, device=device)
    pseudo_conf_sum = torch.zeros(c_num, device=device)
    pseudo_correct_by_class = torch.zeros(c_num, device=device)

    full_labeled_mask = p.train_set.is_labeled.bool()
    labeled_targets = p.train_set.y[full_labeled_mask].long()
    labeled_class_present = (
        torch.bincount(labeled_targets, minlength=c_num).to(device) > 0
    )
    pseudo_selected = pseudo_correct = true_unlabeled_total = 0
    missing_samples = missing_classifier_correct = 0
    missing_proto_eval = missing_proto_correct = 0
    offset = 0

    model.eval()
    for x, y, *_ in loader:
        n = x.size(0)
        y = y.to(device)
        labeled_flag = p.train_set.is_labeled[offset : offset + n].to(device).bool()
        offset += n
        x_w = sage_weak_augment(x, p.dataset).to(device)
        raw_features = model.extractor(x_w)
        logits = model.classifier(raw_features)
        features = F.normalize(raw_features, dim=1)

        oracle_sum.index_add_(0, y, features)
        oracle_count += torch.bincount(y, minlength=c_num).float()

        if bool(labeled_flag.any().item()):
            labels_l = y[labeled_flag]
            feat_l = features[labeled_flag]
            ones = torch.ones(labels_l.size(0), device=device)
            supervised_sum.index_add_(0, labels_l, feat_l)
            supervised_count.index_add_(0, labels_l, ones)
            mixed_sum.index_add_(0, labels_l, feat_l)
            mixed_weight.index_add_(0, labels_l, ones)

        unlabeled_flag = ~labeled_flag
        true_unlabeled_total += int(unlabeled_flag.sum().item())
        if not bool(unlabeled_flag.any().item()):
            continue

        feat_u = features[unlabeled_flag]
        logits_u = logits[unlabeled_flag]
        y_u = y[unlabeled_flag]
        classifier_probs = torch.softmax(logits_u, dim=1)
        missing_mask = ~labeled_class_present[y_u]
        missing_samples += int(missing_mask.sum().item())
        if bool(missing_mask.any().item()):
            pred = classifier_probs.argmax(dim=1)
            missing_classifier_correct += int(
                ((pred == y_u) & missing_mask).sum().item()
            )

        positive_mask = None
        proto_probs = None
        if p.use_proto_guidance:
            proto_probs = _prototype_probs(
                feat_u, p.global_protos, p.global_valid, p.proto_temperature
            )
            if proto_probs is not None:
                positive_mask, _, _ = _build_category_sets(
                    proto_probs,
                    p.global_valid,
                    p.candidate_threshold,
                    p.exclude_threshold,
                )
                valid_global = p.global_valid.to(device).bool()
                proto_eval_mask = missing_mask & valid_global[y_u]
                missing_proto_eval += int(proto_eval_mask.sum().item())
                if bool(proto_eval_mask.any().item()):
                    proto_pred = proto_probs.argmax(dim=1)
                    missing_proto_correct += int(
                        ((proto_pred == y_u) & proto_eval_mask).sum().item()
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
        selected_truth = y_u[high_mask]
        selected_weight = p.pseudo_proto_weight * selected_conf
        mixed_sum.index_add_(
            0, selected_targets, selected_feat * selected_weight.unsqueeze(1)
        )
        mixed_weight.index_add_(0, selected_targets, selected_weight)
        pseudo_count += torch.bincount(selected_targets, minlength=c_num).float()
        pseudo_conf_sum.index_add_(0, selected_targets, selected_conf)
        pseudo_correct_by_class.index_add_(
            0, selected_targets, (selected_targets == selected_truth).float()
        )
        pseudo_selected += int(high_mask.sum().item())
        pseudo_correct += int((selected_targets == selected_truth).sum().item())

    valid = mixed_weight > 0
    supervised_valid = supervised_count > 0
    oracle_valid = oracle_count > 0
    protos = torch.zeros_like(mixed_sum)
    supervised_protos = torch.zeros_like(supervised_sum)
    oracle_protos = torch.zeros_like(oracle_sum)
    protos[valid] = F.normalize(
        mixed_sum[valid] / mixed_weight[valid].unsqueeze(1), dim=1
    )
    supervised_protos[supervised_valid] = F.normalize(
        supervised_sum[supervised_valid]
        / supervised_count[supervised_valid].unsqueeze(1),
        dim=1,
    )
    oracle_protos[oracle_valid] = F.normalize(
        oracle_sum[oracle_valid] / oracle_count[oracle_valid].unsqueeze(1), dim=1
    )

    compactness = torch.zeros(c_num, device=device)
    compactness[valid] = (
        mixed_sum[valid].norm(dim=1) / mixed_weight[valid].clamp_min(1e-12)
    ).clamp(0.0, 1.0)
    support = 1.0 - torch.exp(-mixed_weight / p.support_scale)
    pseudo_quality = torch.ones(c_num, device=device)
    has_pseudo = pseudo_count > 0
    pseudo_quality[has_pseudo] = (
        pseudo_conf_sum[has_pseudo] / pseudo_count[has_pseudo]
    ).clamp(0.0, 1.0)

    stability = torch.ones(c_num, device=device)
    if p.previous_local_protos is not None and p.previous_local_valid is not None:
        previous = p.previous_local_protos.to(device)
        previous_valid = p.previous_local_valid.to(device).bool()
        common = valid & previous_valid
        if bool(common.any().item()):
            similarity = F.cosine_similarity(protos[common], previous[common], dim=1)
            stability[common] = torch.exp(
                -(1.0 - similarity) / p.stability_temperature
            ).clamp(0.0, 1.0)

    reliability = (
        support * compactness * pseudo_quality * stability
    ).clamp_min(1e-12).pow(0.25)
    reliability[~valid] = 0.0

    oracle_quality = torch.full((c_num,), -1.0, device=device)
    comparable = valid & oracle_valid
    if bool(comparable.any().item()):
        oracle_quality[comparable] = F.cosine_similarity(
            protos[comparable], oracle_protos[comparable], dim=1
        )
    supervised_quality = torch.full((c_num,), -1.0, device=device)
    sup_comparable = supervised_valid & oracle_valid
    if bool(sup_comparable.any().item()):
        supervised_quality[sup_comparable] = F.cosine_similarity(
            supervised_protos[sup_comparable], oracle_protos[sup_comparable], dim=1
        )
    proto_shift = torch.zeros(c_num, device=device)
    common_sup = valid & supervised_valid
    if bool(common_sup.any().item()):
        proto_shift[common_sup] = 1.0 - F.cosine_similarity(
            protos[common_sup], supervised_protos[common_sup], dim=1
        )
    pseudo_precision = torch.full((c_num,), -1.0, device=device)
    pseudo_precision[has_pseudo] = (
        pseudo_correct_by_class[has_pseudo] / pseudo_count[has_pseudo]
    )

    recovered_missing = valid & ~supervised_valid & oracle_valid
    phantom = valid & ~oracle_valid
    return {
        "protos": protos.cpu(),
        "valid": valid.cpu(),
        "reliability": reliability.cpu(),
        "supervised_protos": supervised_protos.cpu(),
        "supervised_valid": supervised_valid.cpu(),
        "labeled_count": supervised_count.cpu(),
        "pseudo_count": pseudo_count.cpu(),
        "compactness": compactness.cpu(),
        "support": support.cpu(),
        "pseudo_quality": pseudo_quality.cpu(),
        "stability": stability.cpu(),
        "oracle_quality": oracle_quality.cpu(),
        "supervised_quality": supervised_quality.cpu(),
        "proto_shift": proto_shift.cpu(),
        "pseudo_precision": pseudo_precision.cpu(),
        "proto_pseudo_selected": pseudo_selected,
        "proto_pseudo_correct": pseudo_correct,
        "proto_unlabeled_total": true_unlabeled_total,
        "recovered_missing_proto_count": int(recovered_missing.sum().item()),
        "phantom_proto_count": int(phantom.sum().item()),
        "missing_class_samples": missing_samples,
        "missing_class_classifier_correct": missing_classifier_correct,
        "missing_class_proto_eval": missing_proto_eval,
        "missing_class_proto_correct": missing_proto_correct,
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
        p.train_set, p.batch_size, p.unlabeled_ratio
    )

    sums = {k: 0.0 for k in ("loss", "loss_x", "loss_h", "loss_amb", "loss_pos", "loss_neg")}
    counters = {
        k: 0
        for k in (
            "loss_x_count",
            "loss_u_count",
            "pseudo_selected",
            "pseudo_total",
            "pseudo_correct",
            "raw_high_count",
            "raw_high_correct",
            "proto_rejected",
            "proto_rejected_wrong",
            "candidate_eval",
            "candidate_cover",
            "negative_false",
            "low_candidate_eval",
            "low_candidate_cover",
            "low_negative_false",
            "low_topk_cover",
            "low_sample_count",
        )
    }
    for k in (
        "pseudo_confidence_sum",
        "candidate_size_sum",
        "negative_size_sum",
        "low_candidate_size_sum",
        "low_negative_size_sum",
    ):
        counters[k] = 0.0
    steps = 0

    model.train()
    for _ in range(p.epochs):
        for labeled, (x_u, y_u) in iterate_ssl_batches(loaders):
            if labeled is None:
                raise ValueError(
                    "fedproto_ssl 要求每个客户端至少具有少量有标签数据。"
                )
            x_l, y_l = labeled
            loss, info = _train_batch(model, x_l, y_l, x_u, y_u, p, device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_l, batch_u = x_l.size(0), x_u.size(0)
            sums["loss"] += loss.item()
            sums["loss_x"] += info["loss_x"].item() * batch_l
            sums["loss_h"] += info["loss_h"].item() * batch_u
            sums["loss_amb"] += info["loss_amb"].item() * batch_u
            sums["loss_pos"] += info["loss_pos"].item() * batch_u
            sums["loss_neg"] += info["loss_neg"].item() * batch_u
            counters["loss_x_count"] += batch_l
            counters["loss_u_count"] += batch_u
            for key in counters:
                if key in ("loss_x_count", "loss_u_count"):
                    continue
                counters[key] += info[key]
            steps += 1

    local_proto = _extract_local_prototypes(model, p, device)
    return {
        "state": clone_cpu_state(model.state_dict()),
        "loss": sums["loss"] / max(1, steps),
        "loss_x_sum": sums["loss_x"],
        "loss_h_sum": sums["loss_h"],
        "loss_amb_sum": sums["loss_amb"],
        "loss_pos_sum": sums["loss_pos"],
        "loss_neg_sum": sums["loss_neg"],
        **counters,
        **local_proto,
    }


def _pearson_corr(x, y):
    if x.numel() < 2:
        return 0.0
    x, y = x.float(), y.float()
    xc, yc = x - x.mean(), y - y.mean()
    denom = xc.norm() * yc.norm()
    if denom.item() <= 1e-12:
        return 0.0
    return float(((xc * yc).sum() / denom).item())


class Server(BaseServer):
    def __init__(self, args):
        if args.ssl in ("none", "client"):
            raise ValueError(
                "fedproto_ssl 仅支持 sample、double 或 sfd 半监督场景。"
            )
        super().__init__(args, is_ssl=True, pfl=False)
        self.unlabeled_ratio = args.unlabeled_ratio
        self.proto_temperature = float(_get(args, "proto_temperature", 0.1))
        self.candidate_threshold = float(_get(args, "candidate_threshold", 0.5))
        self.exclude_threshold = float(_get(args, "exclude_threshold", 0.1))
        self.lambda_ambiguous = float(_get(args, "lambda_ambiguous", 1.0))
        self.lambda_negative = float(_get(args, "lambda_negative", 0.5))
        self.pseudo_proto_weight = float(_get(args, "pseudo_proto_weight", 0.1))
        self.support_scale = float(_get(args, "proto_support_scale", 10.0))
        self.stability_temperature = float(
            _get(args, "proto_stability_temperature", 0.1)
        )
        self.proto_warmup_rounds = int(_get(args, "proto_warmup_rounds", 2))
        self.proto_ramp_rounds = int(_get(args, "proto_ramp_rounds", 3))
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

        metric_names = (
            "loss_x",
            "loss_h",
            "loss_amb",
            "loss_pos",
            "loss_neg",
            "loss_server",
            "loss_server_align",
            "loss_server_sep",
            "pseudo_coverage",
            "pseudo_accuracy",
            "raw_pseudo_accuracy",
            "pseudo_confidence",
            "proto_reject_ratio",
            "proto_rejected_error_rate",
            "candidate_coverage",
            "candidate_size",
            "negative_false_rate",
            "negative_size",
            "low_candidate_coverage",
            "low_classifier_topk_coverage",
            "low_candidate_size",
            "low_negative_false_rate",
            "low_negative_size",
            "valid_proto_count",
            "mean_proto_reliability",
            "reliability_quality_corr",
            "mean_oracle_quality",
            "mean_supervised_quality",
            "mean_proto_shift",
            "proto_build_pseudo_acc",
            "class_valid_counts",
            "class_mean_reliability",
            "recovered_missing_proto_count",
            "phantom_proto_count",
            "missing_class_classifier_acc",
            "missing_class_proto_acc",
            "missing_class_sample_count",
            "proto_acc_supervised",
            "proto_acc_mean",
            "proto_acc_weighted",
            "proto_acc_global",
            "round_time",
        )
        for name in metric_names:
            setattr(self, name, [])

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
        progress = (
            round_index - self.proto_warmup_rounds + 1
        ) / self.proto_ramp_rounds
        return True, min(1.0, max(0.0, progress))

    @staticmethod
    def _normalize_weights(weights):
        total = sum(weights)
        return [w / total for w in weights]

    @torch.no_grad()
    def _evaluate_proto_variants(self, variants):
        """一次测试集前向同时比较监督均值、普通均值、可靠加权和学习原型。"""
        prepared = {}
        for name, (protos, valid) in variants.items():
            if protos is None or valid is None or not bool(valid.any().item()):
                prepared[name] = None
            else:
                prepared[name] = (
                    F.normalize(protos.to(self.device), dim=1),
                    valid.to(self.device).bool(),
                )
        correct = {name: 0 for name in variants}
        count = 0
        loader = DataLoader(self.test_set, batch_size=128, shuffle=False)
        self.model.to(self.device)
        self.model.eval()
        for data, target, *_ in loader:
            data, target = data.to(self.device), target.to(self.device)
            features = F.normalize(self.model.extractor(data), dim=1)
            count += target.size(0)
            for name, item in prepared.items():
                if item is None:
                    continue
                protos, valid = item
                scores = features @ protos.t()
                scores = scores.masked_fill(~valid.unsqueeze(0), -1e4)
                correct[name] += int((scores.argmax(dim=1) == target).sum().item())
        self.model.cpu()
        return {
            name: (100.0 * correct[name] / max(1, count) if prepared[name] else 0.0)
            for name in variants
        }

    def _server_proto_update(self, results, selected):
        weighted_sum = torch.zeros(
            self.num_class, self.feature_dim, device=self.device
        )
        total_weight = torch.zeros(self.num_class, device=self.device)
        mean_sum = torch.zeros_like(weighted_sum)
        mean_count = torch.zeros(self.num_class, device=self.device)
        supervised_sum = torch.zeros_like(weighted_sum)
        supervised_weight = torch.zeros(self.num_class, device=self.device)
        client_data = []

        for cid in selected:
            protos = results[cid]["protos"].to(self.device)
            reliability = results[cid]["reliability"].to(self.device)
            valid = results[cid]["valid"].to(self.device).bool()
            weight = reliability * valid.float()
            weighted_sum += protos * weight.unsqueeze(1)
            total_weight += weight
            mean_sum += protos * valid.float().unsqueeze(1)
            mean_count += valid.float()

            sup = results[cid]["supervised_protos"].to(self.device)
            sup_valid = results[cid]["supervised_valid"].to(self.device).bool()
            labeled_count = results[cid]["labeled_count"].to(self.device)
            sw = labeled_count * sup_valid.float()
            supervised_sum += sup * sw.unsqueeze(1)
            supervised_weight += sw
            client_data.append((protos, reliability, valid))

        current_valid = total_weight > 0
        mean_valid = mean_count > 0
        supervised_valid = supervised_weight > 0
        mean_proto = torch.zeros_like(mean_sum)
        supervised_proto = torch.zeros_like(supervised_sum)
        if bool(mean_valid.any().item()):
            mean_proto[mean_valid] = F.normalize(
                mean_sum[mean_valid] / mean_count[mean_valid].unsqueeze(1), dim=1
            )
        if bool(supervised_valid.any().item()):
            supervised_proto[supervised_valid] = F.normalize(
                supervised_sum[supervised_valid]
                / supervised_weight[supervised_valid].unsqueeze(1),
                dim=1,
            )

        if not bool(current_valid.any().item()):
            self.loss_server.append(0.0)
            self.loss_server_align.append(0.0)
            self.loss_server_sep.append(0.0)
            return {
                "mean": mean_proto.cpu(),
                "mean_valid": mean_valid.cpu(),
                "weighted": None,
                "weighted_valid": current_valid.cpu(),
                "supervised": supervised_proto.cpu(),
                "supervised_valid": supervised_valid.cpu(),
            }

        base = torch.zeros_like(weighted_sum)
        base[current_valid] = F.normalize(
            weighted_sum[current_valid]
            / total_weight[current_valid].unsqueeze(1).clamp_min(1e-12),
            dim=1,
        )
        last_loss = last_align = last_sep = 0.0
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
                sim = valid_generated @ valid_generated.t()
                off_diag = ~torch.eye(
                    valid_generated.size(0), dtype=torch.bool, device=self.device
                )
                loss_sep = F.relu(sim[off_diag] - self.server_margin).mean()
            else:
                loss_sep = generated.sum() * 0.0
            loss = loss_align + self.server_separation_weight * loss_sep
            self.proto_optimizer.zero_grad()
            loss.backward()
            self.proto_optimizer.step()
            last_loss, last_align, last_sep = (
                loss.item(),
                loss_align.item(),
                loss_sep.item(),
            )

        self.proto_learner.eval()
        with torch.no_grad():
            learned = self.proto_learner(base).cpu()
            tw = total_weight.cpu()
            cv = current_valid.cpu()
            for c in range(self.num_class):
                if not bool(cv[c].item()):
                    continue
                if not bool(self.global_valid[c].item()):
                    self.global_protos[c] = learned[c]
                    self.global_valid[c] = True
                    continue
                support_factor = min(1.0, tw[c].item() / self.server_support_scale)
                ratio = self.server_max_momentum * support_factor
                updated = (1.0 - ratio) * self.global_protos[c] + ratio * learned[c]
                self.global_protos[c] = F.normalize(updated.unsqueeze(0), dim=1)[0]

        self.loss_server.append(last_loss)
        self.loss_server_align.append(last_align)
        self.loss_server_sep.append(last_sep)
        return {
            "mean": mean_proto.cpu(),
            "mean_valid": mean_valid.cpu(),
            "weighted": base.cpu(),
            "weighted_valid": current_valid.cpu(),
            "supervised": supervised_proto.cpu(),
            "supervised_valid": supervised_valid.cpu(),
        }

    def _collect_local_diag(self, results, selected):
        rels, qualities, sup_qualities, shifts = [], [], [], []
        class_valid = torch.zeros(self.num_class)
        class_rel_sum = torch.zeros(self.num_class)
        class_rel_count = torch.zeros(self.num_class)
        build_selected = build_correct = 0
        recovered = phantom = 0
        missing_samples = missing_cls_correct = 0
        missing_proto_eval = missing_proto_correct = 0

        for cid in selected:
            valid = results[cid]["valid"].bool()
            rel = results[cid]["reliability"]
            quality = results[cid]["oracle_quality"]
            class_valid += valid.float()
            class_rel_sum += rel * valid.float()
            class_rel_count += valid.float()
            if bool(valid.any().item()):
                rels.append(rel[valid])
                qualities.append(quality[valid])
            sup_valid = results[cid]["supervised_valid"].bool()
            if bool(sup_valid.any().item()):
                sup_qualities.append(results[cid]["supervised_quality"][sup_valid])
            common = valid & sup_valid
            if bool(common.any().item()):
                shifts.append(results[cid]["proto_shift"][common])
            build_selected += results[cid]["proto_pseudo_selected"]
            build_correct += results[cid]["proto_pseudo_correct"]
            recovered += results[cid]["recovered_missing_proto_count"]
            phantom += results[cid]["phantom_proto_count"]
            missing_samples += results[cid]["missing_class_samples"]
            missing_cls_correct += results[cid]["missing_class_classifier_correct"]
            missing_proto_eval += results[cid]["missing_class_proto_eval"]
            missing_proto_correct += results[cid]["missing_class_proto_correct"]

        if rels:
            rel_all = torch.cat(rels)
            quality_all = torch.cat(qualities)
            corr = _pearson_corr(rel_all, quality_all)
            mean_rel = rel_all.mean().item()
            mean_quality = quality_all.mean().item()
        else:
            corr = mean_rel = mean_quality = 0.0
        mean_sup_quality = (
            torch.cat(sup_qualities).mean().item() if sup_qualities else 0.0
        )
        mean_shift = torch.cat(shifts).mean().item() if shifts else 0.0
        class_mean_rel = torch.zeros(self.num_class)
        mask = class_rel_count > 0
        class_mean_rel[mask] = class_rel_sum[mask] / class_rel_count[mask]
        return {
            "valid_count": int(class_valid.sum().item()),
            "mean_reliability": mean_rel,
            "reliability_quality_corr": corr,
            "mean_oracle_quality": mean_quality,
            "mean_supervised_quality": mean_sup_quality,
            "mean_proto_shift": mean_shift,
            "proto_build_pseudo_acc": 100.0 * build_correct / max(1, build_selected),
            "class_valid_counts": class_valid.int().tolist(),
            "class_mean_reliability": [round(float(v), 4) for v in class_mean_rel],
            "recovered_missing_proto_count": recovered,
            "phantom_proto_count": phantom,
            "missing_class_sample_count": missing_samples,
            "missing_class_classifier_acc": 100.0 * missing_cls_correct / max(1, missing_samples),
            "missing_class_proto_acc": (
                100.0 * missing_proto_correct / max(1, missing_proto_eval)
                if missing_proto_eval > 0
                else 0.0
            ),
        }

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

            params = []
            for base in self.build_base_params(selected):
                cid = base.client_id
                params.append(
                    Params(
                        **asdict(base),
                        unlabeled_ratio=self.unlabeled_ratio,
                        lambda_u=self.lam,
                        confidence=self.confidence,
                        round_index=round_index,
                        use_proto_guidance=use_proto,
                        ambiguous_weight=ambiguous_weight,
                        global_protos=(
                            self.global_protos.clone()
                            if bool(self.global_valid.any().item())
                            else None
                        ),
                        global_valid=(
                            self.global_valid.clone()
                            if bool(self.global_valid.any().item())
                            else None
                        ),
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
            results = self.run_clients(train, params)

            states = [results[cid]["state"] for cid in selected]
            weights = [self.weights[cid] for cid in selected]
            self.aggregate(states, weights=self._normalize_weights(weights))
            for cid in selected:
                self.previous_local_protos[cid] = results[cid]["protos"].clone()
                self.previous_local_valid[cid] = results[cid]["valid"].clone()

            variants = self._server_proto_update(results, selected)
            x_count = sum(results[cid]["loss_x_count"] for cid in selected)
            u_count = sum(results[cid]["loss_u_count"] for cid in selected)
            for name, key, count in (
                ("loss_x", "loss_x_sum", x_count),
                ("loss_h", "loss_h_sum", u_count),
                ("loss_amb", "loss_amb_sum", u_count),
                ("loss_pos", "loss_pos_sum", u_count),
                ("loss_neg", "loss_neg_sum", u_count),
            ):
                getattr(self, name).append(
                    sum(results[cid][key] for cid in selected) / max(1, count)
                )
            self.loss.append(sum(results[cid]["loss"] for cid in selected) / len(selected))

            pseudo_selected = sum(results[cid]["pseudo_selected"] for cid in selected)
            pseudo_total = sum(results[cid]["pseudo_total"] for cid in selected)
            pseudo_correct = sum(results[cid]["pseudo_correct"] for cid in selected)
            raw_high = sum(results[cid]["raw_high_count"] for cid in selected)
            raw_correct = sum(results[cid]["raw_high_correct"] for cid in selected)
            rejected = sum(results[cid]["proto_rejected"] for cid in selected)
            rejected_wrong = sum(
                results[cid]["proto_rejected_wrong"] for cid in selected
            )
            conf_sum = sum(
                results[cid]["pseudo_confidence_sum"] for cid in selected
            )
            self.pseudo_coverage.append(100.0 * pseudo_selected / max(1, pseudo_total))
            self.pseudo_accuracy.append(100.0 * pseudo_correct / max(1, pseudo_selected))
            self.raw_pseudo_accuracy.append(100.0 * raw_correct / max(1, raw_high))
            self.pseudo_confidence.append(conf_sum / max(1, pseudo_selected))
            self.proto_reject_ratio.append(100.0 * rejected / max(1, raw_high))
            self.proto_rejected_error_rate.append(
                100.0 * rejected_wrong / max(1, rejected)
            )

            candidate_eval = sum(results[cid]["candidate_eval"] for cid in selected)
            candidate_cover = sum(results[cid]["candidate_cover"] for cid in selected)
            negative_false = sum(results[cid]["negative_false"] for cid in selected)
            candidate_samples = pseudo_total if use_proto else 0
            self.candidate_coverage.append(
                100.0 * candidate_cover / max(1, candidate_eval) if candidate_eval else 0.0
            )
            self.candidate_size.append(
                sum(results[cid]["candidate_size_sum"] for cid in selected)
                / max(1, candidate_samples)
            )
            self.negative_false_rate.append(
                100.0 * negative_false / max(1, candidate_eval) if candidate_eval else 0.0
            )
            self.negative_size.append(
                sum(results[cid]["negative_size_sum"] for cid in selected)
                / max(1, candidate_samples)
            )

            low_eval = sum(results[cid]["low_candidate_eval"] for cid in selected)
            low_cover = sum(results[cid]["low_candidate_cover"] for cid in selected)
            low_topk = sum(results[cid]["low_topk_cover"] for cid in selected)
            low_false = sum(results[cid]["low_negative_false"] for cid in selected)
            low_samples = sum(results[cid]["low_sample_count"] for cid in selected)
            self.low_candidate_coverage.append(
                100.0 * low_cover / max(1, low_eval) if low_eval else 0.0
            )
            self.low_classifier_topk_coverage.append(
                100.0 * low_topk / max(1, low_eval) if low_eval else 0.0
            )
            self.low_candidate_size.append(
                sum(results[cid]["low_candidate_size_sum"] for cid in selected)
                / max(1, low_samples)
            )
            self.low_negative_false_rate.append(
                100.0 * low_false / max(1, low_eval) if low_eval else 0.0
            )
            self.low_negative_size.append(
                sum(results[cid]["low_negative_size_sum"] for cid in selected)
                / max(1, low_samples)
            )

            diag = self._collect_local_diag(results, selected)
            for key in (
                "valid_proto_count",
                "mean_proto_reliability",
                "reliability_quality_corr",
                "mean_oracle_quality",
                "mean_supervised_quality",
                "mean_proto_shift",
                "proto_build_pseudo_acc",
                "class_valid_counts",
                "class_mean_reliability",
                "recovered_missing_proto_count",
                "phantom_proto_count",
                "missing_class_classifier_acc",
                "missing_class_proto_acc",
                "missing_class_sample_count",
            ):
                source = key
                if key == "valid_proto_count":
                    source = "valid_count"
                getattr(self, key).append(diag[source])

            self.evaluate()
            proto_scores = self._evaluate_proto_variants(
                {
                    "supervised": (variants["supervised"], variants["supervised_valid"]),
                    "mean": (variants["mean"], variants["mean_valid"]),
                    "weighted": (variants["weighted"], variants["weighted_valid"]),
                    "global": (self.global_protos, self.global_valid),
                }
            )
            for metric, score_key in (
                ("proto_acc_supervised", "supervised"),
                ("proto_acc_mean", "mean"),
                ("proto_acc_weighted", "weighted"),
                ("proto_acc_global", "global"),
            ):
                getattr(self, metric).append(proto_scores[score_key])
            self.acc_proto.append(proto_scores["global"])
            self.round_time.append(time.time() - started)

            print(
                f"Accuracy: {self.acc[-1]:.2f}% | Proto(S/M/W/G): "
                f"{proto_scores['supervised']:.2f}/{proto_scores['mean']:.2f}/"
                f"{proto_scores['weighted']:.2f}/{proto_scores['global']:.2f}%"
            )
            print(
                f"Pseudo: raw_acc={self.raw_pseudo_accuracy[-1]:.2f}%, "
                f"filtered_acc={self.pseudo_accuracy[-1]:.2f}%, "
                f"coverage={self.pseudo_coverage[-1]:.2f}%, "
                f"reject={self.proto_reject_ratio[-1]:.2f}%, "
                f"rejected_wrong={self.proto_rejected_error_rate[-1]:.2f}%"
            )
            if use_proto:
                print(
                    f"Low-conf Candidate: cover={self.low_candidate_coverage[-1]:.2f}% "
                    f"(classifier Top-k={self.low_classifier_topk_coverage[-1]:.2f}%), "
                    f"size={self.low_candidate_size[-1]:.2f} | Negative: "
                    f"false={self.low_negative_false_rate[-1]:.2f}%, "
                    f"size={self.low_negative_size[-1]:.2f}"
                )
            print(
                f"Local protos: valid={diag['valid_count']}, "
                f"reliability={diag['mean_reliability']:.3f}, "
                f"rel-quality corr={diag['reliability_quality_corr']:.3f}, "
                f"quality(mixed/sup)={diag['mean_oracle_quality']:.3f}/"
                f"{diag['mean_supervised_quality']:.3f}, "
                f"shift={diag['mean_proto_shift']:.4f}"
            )
            if self.num_class <= 20:
                print(
                    f"Class support={diag['class_valid_counts']} | "
                    f"Class reliability={diag['class_mean_reliability']}"
                )
            print(
                f"Missing-class: samples={diag['missing_class_sample_count']}, "
                f"classifier_acc={diag['missing_class_classifier_acc']:.2f}%, "
                f"proto_acc={diag['missing_class_proto_acc']:.2f}% | "
                f"recovered={diag['recovered_missing_proto_count']}, "
                f"phantom={diag['phantom_proto_count']}"
            )
            print(
                f"Server loss={self.loss_server[-1]:.4f} "
                f"(align={self.loss_server_align[-1]:.4f}, "
                f"sep={self.loss_server_sep[-1]:.4f}) | "
                f"Round time={self.round_time[-1]:.2f}s"
            )

    def save(self):
        metrics = {"acc": self.acc, "acc_proto": self.acc_proto, "loss": self.loss}
        for name in (
            "loss_x",
            "loss_h",
            "loss_amb",
            "loss_pos",
            "loss_neg",
            "loss_server",
            "loss_server_align",
            "loss_server_sep",
            "pseudo_coverage",
            "pseudo_accuracy",
            "raw_pseudo_accuracy",
            "pseudo_confidence",
            "proto_reject_ratio",
            "proto_rejected_error_rate",
            "candidate_coverage",
            "candidate_size",
            "negative_false_rate",
            "negative_size",
            "low_candidate_coverage",
            "low_classifier_topk_coverage",
            "low_candidate_size",
            "low_negative_false_rate",
            "low_negative_size",
            "valid_proto_count",
            "mean_proto_reliability",
            "reliability_quality_corr",
            "mean_oracle_quality",
            "mean_supervised_quality",
            "mean_proto_shift",
            "proto_build_pseudo_acc",
            "class_valid_counts",
            "class_mean_reliability",
            "recovered_missing_proto_count",
            "phantom_proto_count",
            "missing_class_classifier_acc",
            "missing_class_proto_acc",
            "missing_class_sample_count",
            "proto_acc_supervised",
            "proto_acc_mean",
            "proto_acc_weighted",
            "proto_acc_global",
            "round_time",
        ):
            metrics[name] = getattr(self, name)
        params = {
            "global": self.model.state_dict(),
            "proto": self.global_protos,
            "aux": {
                "global_valid": self.global_valid,
                "proto_learner": self.proto_learner.state_dict(),
            },
        }
        self.deal_save(metrics, params)
