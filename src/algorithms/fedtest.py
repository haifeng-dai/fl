import os
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

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
from .utils.augment import weak_augment
from .utils.ssl import build_fixmatch_loaders, iterate_fixmatch_batches


def get_path(args):
    args.file_name = (
        f"{args.common_name}_{fmt_num(args.lambda_s)}_{fmt_num(args.lambda_u)}"
    )
    return os.path.join(args.log_path, f"{args.file_name}_{args.cur_time}.log")


@dataclass
class Params(BaseParams):
    unlabeled_ratio: int
    proto_g: torch.Tensor
    global_valid: torch.Tensor
    proto_scale: float
    lambda_s: float
    lambda_u: float


@dataclass
class ClientDiagnoseParams(BaseParams):
    proto_g: torch.Tensor
    global_valid: torch.Tensor
    proto_scale: float
    proto_conf_threshold: torch.Tensor
    proto_conf_threshold_valid: torch.Tensor


@torch.no_grad()
def estimate_prototypes_worker(p: BaseParams):
    """客户端使用当前轮次最新的全局模型，在有标签数据上提取类别原型。"""

    model = get_model(p).to(p.dev)

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
        p.dev,
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

    model_l = get_model(p).to(p.dev)
    model_l.load_state_dict(p.model_state)

    model_g = get_model(p).to(p.dev)
    model_g.load_state_dict(p.model_state)
    model_g.eval()
    for parameter in model_g.parameters():
        parameter.requires_grad_(False)

    loaders = build_fixmatch_loaders(
        p.train_set,
        p.batch_size,
        p.unlabeled_ratio,
    )
    if loaders.labeled_loader is None:
        raise ValueError("fedtest 客户端缺少有标签样本")

    proto_g = p.proto_g.to(p.dev)
    global_valid = p.global_valid.to(p.dev).bool()
    optimizer = torch.optim.SGD(
        model_l.parameters(),
        lr=p.lr,
        momentum=p.momentum,
        weight_decay=p.weight_decay,
    )

    stats = torch.zeros(13, dtype=torch.float64, device=p.dev)
    # 0/1: loss_x sum/count
    # 2/3: loss_cal sum/count
    # 4/5: loss_total sum/count
    # 6/7: loss_x_u sum/count
    # 8/9: loss_cal_u sum/count
    # 10/11: pseudo-label count / unlabeled sample count
    # 12: selected pseudo-label correct count
    hc_diag = {
        "lo": [0, 0, 0],
        "go": [0, 0, 0],
        "b": [0, 0, 0, 0, 0],
        "n": [0, 0, 0],
        "l": [0, 0, 0],
        "g": [0, 0, 0],
        "u": [0, 0, 0],
    }
    proto_diag = {
        "all": [0, 0, 0, 0, 0],
        "local_high": [0, 0, 0, 0, 0],
    }

    model_l.train()

    for local_epoch in range(p.epochs):
        for labeled_batch, unlabeled_batch in iterate_fixmatch_batches(loaders):
            x_l_raw, y_l = labeled_batch
            x_l_raw = x_l_raw.to(p.dev)
            y_l = y_l.to(p.dev)
            x_u_raw, y_u_raw = unlabeled_batch
            x_u_raw = x_u_raw.to(p.dev)

            x_l_weak = weak_augment(x_l_raw, p.dataset)

            with torch.no_grad():
                x_u_weak = weak_augment(x_u_raw, p.dataset)
                features_u_weak = model_l.extractor(x_u_weak)
                logits_u_pseudo = model_l.classifier(features_u_weak)
                confidence_u, pseudo_y = torch.softmax(logits_u_pseudo, dim=1).max(
                    dim=1
                )
                pseudo_mask = confidence_u > 0.95

                if local_epoch == p.epochs - 1:
                    local_mask = pseudo_mask
                    local_pred = pseudo_y
                    global_logits = model_g(x_u_weak)
                    global_prob, global_pred = torch.softmax(global_logits, dim=1).max(
                        dim=1
                    )
                    global_mask = global_prob > 0.95
                    groups = {
                        "lo": local_mask & ~global_mask,
                        "go": ~local_mask & global_mask,
                        "b": local_mask & global_mask,
                        "n": ~local_mask & ~global_mask,
                        "l": local_mask,
                        "g": global_mask,
                        "u": local_mask | global_mask,
                    }
                    for key, group in groups.items():
                        hc_diag[key][0] += int(group.sum().item())
                        hc_diag[key][1] += int(
                            ((local_pred == y_u_raw.to(p.dev)) & group).sum().item()
                        )
                        hc_diag[key][2] += int(
                            ((global_pred == y_u_raw.to(p.dev)) & group).sum().item()
                        )
                    agree = local_pred == global_pred
                    both = groups["b"]
                    hc_diag["b"][3] += int((agree & both).sum().item())
                    hc_diag["b"][4] += int(
                        (agree & (local_pred == y_u_raw.to(p.dev)) & both).sum().item()
                    )

                    if bool(global_valid.any()):
                        proto_distance = mse_distance(features_u_weak, proto_g)
                        proto_distance[:, ~global_valid] = float("inf")
                        proto_pred = proto_distance.argmin(dim=1)
                        proto_ok = proto_pred == y_u_raw.to(p.dev)
                        classifier_ok = local_pred == y_u_raw.to(p.dev)
                        proto_agree = local_pred == proto_pred
                        for key, group in {
                            "all": torch.ones_like(local_mask),
                            "local_high": local_mask,
                        }.items():
                            proto_diag[key][0] += int(group.sum().item())
                            proto_diag[key][1] += int(
                                (classifier_ok & group).sum().item()
                            )
                            proto_diag[key][2] += int((proto_ok & group).sum().item())
                            proto_diag[key][3] += int(
                                (proto_agree & group).sum().item()
                            )
                            proto_diag[key][4] += int(
                                (proto_agree & classifier_ok & group).sum().item()
                            )

            selected_x_u = x_u_weak[pseudo_mask]
            selected_y_u = pseudo_y[pseudo_mask]

            if selected_y_u.numel() > 0:
                x_train = torch.cat((x_l_weak, selected_x_u), dim=0)
            else:
                x_train = x_l_weak
            features_train = model_l.extractor(x_train)
            logits_train = model_l.classifier(features_train)
            features_l = features_train[: y_l.size(0)]
            logits_l = logits_train[: y_l.size(0)]

            loss_x = F.cross_entropy(logits_l, y_l)

            diff = (
                mse_distance(features_l, proto_g).gather(1, y_l.unsqueeze(1)).squeeze(1)
            )

            valid_labeled = global_valid[y_l]
            valid_count = valid_labeled.sum()

            loss_cal = (diff * valid_labeled.float()).sum() / valid_count.clamp_min(
                1
            ).float()

            if selected_y_u.numel() > 0:
                logits_u = logits_train[y_l.size(0) :]
                loss_x_u = F.cross_entropy(logits_u, selected_y_u)
                diff_u = (
                    mse_distance(features_train[y_l.size(0) :], proto_g)
                    .gather(1, selected_y_u.unsqueeze(1))
                    .squeeze(1)
                )
                valid_u = global_valid[selected_y_u]
                valid_u_count = valid_u.sum()
                loss_cal_u = (diff_u * valid_u.float()).sum() / valid_u_count.clamp_min(
                    1
                ).float()
            else:
                loss_x_u = logits_l.sum() * 0.0
                loss_cal_u = features_l.sum() * 0.0
                diff_u = loss_cal_u.detach().expand(0)
                valid_u = torch.zeros(0, dtype=torch.bool, device=p.dev)
                valid_u_count = torch.zeros((), dtype=torch.long, device=p.dev)

            loss_u = loss_x_u + p.lambda_s * loss_cal_u
            loss = loss_x + p.lambda_s * loss_cal + p.lambda_u * loss_u

            optimizer.zero_grad()
            check_losses(loss, locals())
            loss.backward()
            optimizer.step()

            bs = float(y_l.size(0))

            stats[0] += loss_x.detach().double() * bs
            stats[1] += bs

            stats[2] += (diff.detach().double() * valid_labeled.double()).sum()
            stats[3] += valid_count

            stats[4] += loss.detach().double() * bs
            stats[5] += bs
            pseudo_bs = float(selected_y_u.numel())
            stats[6] += loss_x_u.detach().double() * pseudo_bs
            stats[7] += pseudo_bs
            stats[8] += (diff_u.detach().double() * valid_u.double()).sum()
            stats[9] += valid_u_count
            stats[10] += pseudo_bs
            stats[11] += float(y_u_raw.size(0))
            stats[12] += (selected_y_u == y_u_raw.to(p.dev)[pseudo_mask]).double().sum()

    stats = stats.cpu().tolist()

    return {
        "state": clone_cpu_state(model_l.state_dict()),
        "loss_x_sum": stats[0],
        "loss_x_count": int(stats[1]),
        "loss_calibrate_sum": stats[2],
        "loss_calibrate_count": int(stats[3]),
        "loss_total_sum": stats[4],
        "loss_total_count": int(stats[5]),
        "loss_x_u_sum": stats[6],
        "loss_x_u_count": int(stats[7]),
        "loss_calibrate_u_sum": stats[8],
        "loss_calibrate_u_count": int(stats[9]),
        "pseudo_count": int(stats[10]),
        "unlabeled_count": int(stats[11]),
        "pseudo_correct": int(stats[12]),
        "hc_diag": hc_diag,
        "proto_diag": proto_diag,
    }


@torch.no_grad()
def client_diagnose_worker(p: ClientDiagnoseParams):
    """阶段2：在无标签数据上评估基础分类与同空间下的候选集合质量。"""

    model = get_model(p).to(p.dev)

    model.load_state_dict(p.model_state)
    model.eval()

    prototypes = p.proto_g.to(p.dev)
    valid = p.global_valid.to(p.dev).bool()
    num_class = p.num_class

    labeled_mask = p.train_set.is_labeled.bool()
    labeled_present = (
        torch.bincount(
            p.train_set.y[labeled_mask],
            minlength=num_class,
        )
        > 0
    ).to(p.dev)

    unlabeled_indices = torch.where(~labeled_mask)[0]

    # 0 total, 1 classifier correct, 2 prototype correct
    # 3/4/5 seen total/classifier correct/prototype correct
    # 6/7/8 missing total/classifier correct/prototype correct
    # 0-8: total/classifier/prototype, seen and missing class counts
    # 9-13: classifier confidence <95% total/classifier/prototype/agreement/agreement-correct
    # 14-21: classifier confidence >95% total/correct/missing/missing-correct/
    #        agreement/agreement-correct/prototype-correct/prototype-correct-classifier-correct
    # 22:22+num_class: prototype-correct confidence sums by true class
    # 22+num_class:22+2*num_class: corresponding counts
    diag_counts = torch.zeros(22 + 2 * num_class, dtype=torch.float64, device=p.dev)
    threshold_count = torch.zeros(num_class, dtype=torch.float64, device=p.dev)
    threshold_correct = torch.zeros(num_class, dtype=torch.float64, device=p.dev)
    candidate_counts = torch.zeros(8, dtype=torch.float64, device=p.dev)
    low_candidate_counts = torch.zeros(8, dtype=torch.float64, device=p.dev)
    cand_size_counts = torch.zeros(num_class + 1, dtype=torch.float64, device=p.dev)
    cand_size_correct = torch.zeros(num_class + 1, dtype=torch.float64, device=p.dev)
    low_cand_size_counts = torch.zeros(num_class + 1, dtype=torch.float64, device=p.dev)
    low_cand_size_correct = torch.zeros(
        num_class + 1, dtype=torch.float64, device=p.dev
    )
    threshold = p.proto_conf_threshold.to(p.dev)
    threshold_valid = p.proto_conf_threshold_valid.to(p.dev).bool()

    if unlabeled_indices.numel() > 0:
        raw_x = p.train_set.x[unlabeled_indices]
        raw_y = p.train_set.y[unlabeled_indices].to(p.dev)
        loader_u = DataLoader(
            TensorDataset(raw_x, raw_y),
            batch_size=max(p.batch_size, 256),
            shuffle=False,
        )
        for bx_raw, by in loader_u:
            bx_raw = bx_raw.to(p.dev)
            bx_prep = prepare_input_batch(bx_raw, p.dataset)
            feat_u = model.extractor(bx_prep)
            logits_u = model.classifier(feat_u)
            cls_pred = logits_u.argmax(dim=1)
            cls_prob = torch.softmax(logits_u, dim=1).max(dim=1).values
            cls95 = cls_prob > 0.95
            low_conf = cls_prob < 0.95

            if bool(valid.any()):
                dist_u = mse_distance(feat_u, prototypes)
                dist_u[:, ~valid] = float("inf")
                prob_u = torch.softmax(-dist_u / max(float(p.proto_scale), 1e-8), dim=1)
                prob_max, proto_pred = prob_u.max(dim=1)
                threshold_candidate = threshold_valid[proto_pred] & (
                    prob_max > threshold[proto_pred]
                )
                threshold_count += torch.bincount(
                    proto_pred[threshold_candidate], minlength=num_class
                ).double()
                threshold_correct += torch.bincount(
                    proto_pred[threshold_candidate & (proto_pred == by)],
                    minlength=num_class,
                ).double()
                if bool(threshold_valid.any()):
                    candidate_mask = (
                        prob_u > threshold.unsqueeze(0)
                    ) & threshold_valid.unsqueeze(0)
                    candidate_count = candidate_mask.sum(dim=1)
                    true_in_candidate = candidate_mask.gather(
                        1, by.unsqueeze(1)
                    ).squeeze(1)
                    single = candidate_count == 1
                    multi = candidate_count > 1
                    candidate_counts += torch.stack(
                        [
                            torch.tensor(by.numel(), device=p.dev),
                            (candidate_count == 0).sum(),
                            single.sum(),
                            multi.sum(),
                            candidate_count.sum(),
                            true_in_candidate.sum(),
                            (single & true_in_candidate).sum(),
                            (multi & true_in_candidate).sum(),
                        ]
                    ).double()
                    cand_size_counts += torch.bincount(
                        candidate_count, minlength=num_class + 1
                    ).double()
                    cand_size_correct += torch.bincount(
                        candidate_count[true_in_candidate], minlength=num_class + 1
                    ).double()
                    low_mask = cls_prob < 0.95
                    low_candidate_count = candidate_count[low_mask]
                    low_true_in_candidate = true_in_candidate[low_mask]
                    low_single = low_candidate_count == 1
                    low_multi = low_candidate_count > 1
                    low_candidate_counts += torch.stack(
                        [
                            low_mask.sum(),
                            (low_candidate_count == 0).sum(),
                            low_single.sum(),
                            low_multi.sum(),
                            low_candidate_count.sum(),
                            low_true_in_candidate.sum(),
                            (low_single & low_true_in_candidate).sum(),
                            (low_multi & low_true_in_candidate).sum(),
                        ]
                    ).double()
                    low_cand_size_counts += torch.bincount(
                        low_candidate_count, minlength=num_class + 1
                    ).double()
                    low_cand_size_correct += torch.bincount(
                        low_candidate_count[low_true_in_candidate],
                        minlength=num_class + 1,
                    ).double()
            else:
                prob_max = torch.zeros(by.size(0), device=p.dev)
                proto_pred = torch.full_like(by, -1)

            cls_ok = cls_pred == by
            proto_ok = proto_pred == by
            is_seen = labeled_present[by]
            is_mis = ~is_seen
            agree = cls_pred == proto_pred
            diag_counts[9] += low_conf.double().sum()
            diag_counts[10] += (low_conf & cls_ok).double().sum()
            diag_counts[11] += (low_conf & proto_ok).double().sum()
            diag_counts[12] += (low_conf & agree).double().sum()
            diag_counts[13] += (low_conf & agree & cls_ok).double().sum()
            diag_counts[14] += cls95.double().sum()
            diag_counts[15] += (cls95 & cls_ok).double().sum()
            diag_counts[16] += (cls95 & is_mis).double().sum()
            diag_counts[17] += (cls95 & cls_ok & is_mis).double().sum()
            diag_counts[18] += (cls95 & agree).double().sum()
            diag_counts[19] += (cls95 & agree & cls_ok).double().sum()
            diag_counts[20] += (cls95 & proto_ok).double().sum()
            diag_counts[21] += (cls95 & proto_ok & cls_ok).double().sum()

            class_conf_sum = torch.bincount(
                by[proto_ok],
                weights=prob_max[proto_ok],
                minlength=num_class,
            )
            class_conf_count = torch.bincount(by[proto_ok], minlength=num_class).to(
                torch.float64
            )
            diag_counts[22 : 22 + num_class] += class_conf_sum.double()
            diag_counts[22 + num_class :] += class_conf_count

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

    return {
        "diag_counts": diag_counts.cpu(),
        "proto_conf_threshold_count": threshold_count.cpu(),
        "proto_conf_threshold_correct": threshold_correct.cpu(),
        "proto_candidate_counts": candidate_counts.cpu(),
        "proto_low_candidate_counts": low_candidate_counts.cpu(),
        "proto_candidate_size_counts": cand_size_counts.cpu(),
        "proto_candidate_size_correct": cand_size_correct.cpu(),
        "proto_low_candidate_size_counts": low_cand_size_counts.cpu(),
        "proto_low_candidate_size_correct": low_cand_size_correct.cpu(),
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
        self.proto_g = torch.zeros(self.num_class, self.feature_dim)
        self.global_valid = torch.zeros(self.num_class, dtype=torch.bool)

        self.proto_anchor_weight = args.proto_anchor_weight
        self.proto_sep_weight = args.proto_sep_weight
        self.proto_opt_lr = args.proto_opt_lr
        self.proto_opt_steps = args.proto_opt_steps

        self.lambda_s = args.lambda_s
        self.lambda_u = args.lambda_u
        self.unlabeled_ratio = args.unlabeled_ratio
        self.proto_scale = 1.0

        # 核心指标序列
        self.acc = []
        self.acc_proto = []
        self.acc_proto_mean = []
        self.loss = []

        metric_names = (
            "loss_x",
            "loss_calibrate",
            "loss_x_u",
            "loss_calibrate_u",
            "pseudo_count",
            "pseudo_rate",
            "pseudo_acc",
            "round_time",
            "unlabeled_classifier_acc",
            "unlabeled_proto_acc",
            "seen_classifier_acc",
            "seen_proto_acc",
            "missing_classifier_acc",
            "missing_proto_acc",
            "low_conf_classifier_acc",
            "low_conf_proto_acc",
            "low_conf_agree_rate",
            "low_conf_agree_cls_acc",
            "cls95_agree_rate",
            "cls95_agree_sample_rate",
            "cls95_agree_acc",
            "cls95_proto_ok_rate",
            "cls95_proto_ok_cls_acc",
            "cls95_discard_rate",
            "cls95_discard_acc",
            "cls95_discard_mis_rate",
        )
        for name in metric_names:
            setattr(self, name, [])
        self.proto_correct_conf_mean = []
        self.proto_correct_conf_sum = []
        self.proto_correct_conf_count = []
        self.proto_conf_threshold = []
        self.proto_conf_threshold_valid = []
        self.proto_conf_threshold_count = []
        self.proto_conf_threshold_correct = []
        self.proto_conf_threshold_acc = []
        self.proto_candidate_counts = []
        self.proto_candidate_valid = []
        self.proto_candidate_empty_rate = []
        self.proto_candidate_single_rate = []
        self.proto_candidate_multi_rate = []
        self.proto_candidate_mean_count = []
        self.proto_candidate_true_coverage = []
        self.proto_candidate_single_coverage = []
        self.proto_candidate_multi_coverage = []
        self.proto_candidate_single_accuracy = []
        self.proto_candidate_multi_accuracy = []
        self.proto_candidate_nonempty_accuracy = []
        self.proto_low_candidate_counts = []
        self.proto_low_candidate_valid = []
        self.proto_low_candidate_empty_rate = []
        self.proto_low_candidate_single_rate = []
        self.proto_low_candidate_multi_rate = []
        self.proto_low_candidate_mean_count = []
        self.proto_low_candidate_true_coverage = []
        self.proto_low_candidate_single_coverage = []
        self.proto_low_candidate_multi_coverage = []
        self.proto_low_candidate_single_accuracy = []
        self.proto_low_candidate_multi_accuracy = []
        self.proto_low_candidate_nonempty_accuracy = []
        self.proto_candidate_size_counts = []
        self.proto_candidate_size_correct = []
        self.proto_low_candidate_size_counts = []
        self.proto_low_candidate_size_correct = []
        self.hc_diag = []
        self.proto_diag = []

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

            check_losses(loss, locals())
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
        prototypes = self.proto_g.to(self.device)
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
                    proto_g=self.proto_g.clone(),
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
            lxu_sum = sum(train_results[cid]["loss_x_u_sum"] for cid in selected)
            lxu_cnt = sum(train_results[cid]["loss_x_u_count"] for cid in selected)
            lcalu_sum = sum(
                train_results[cid]["loss_calibrate_u_sum"] for cid in selected
            )
            lcalu_cnt = sum(
                train_results[cid]["loss_calibrate_u_count"] for cid in selected
            )
            pseudo_count = sum(train_results[cid]["pseudo_count"] for cid in selected)
            pseudo_correct = sum(
                train_results[cid]["pseudo_correct"] for cid in selected
            )
            unlabeled_count = sum(
                train_results[cid]["unlabeled_count"] for cid in selected
            )
            hc_diag = {
                "lo": [0, 0, 0],
                "go": [0, 0, 0],
                "b": [0, 0, 0, 0, 0],
                "n": [0, 0, 0],
                "l": [0, 0, 0],
                "g": [0, 0, 0],
                "u": [0, 0, 0],
            }
            proto_diag = {
                "all": [0, 0, 0, 0, 0],
                "local_high": [0, 0, 0, 0, 0],
            }
            for cid in selected:
                for key, values in train_results[cid]["hc_diag"].items():
                    for index, value in enumerate(values):
                        hc_diag[key][index] += value
                for key, values in train_results[cid]["proto_diag"].items():
                    for index, value in enumerate(values):
                        proto_diag[key][index] += value

            self.loss_x.append(lx_sum / max(1, lx_cnt))
            self.loss_calibrate.append(lcal_sum / max(1, lcal_cnt))
            self.loss_x_u.append(lxu_sum / max(1, lxu_cnt))
            self.loss_calibrate_u.append(lcalu_sum / max(1, lcalu_cnt))
            self.pseudo_count.append(pseudo_count)
            self.pseudo_rate.append(100.0 * pseudo_count / max(1, unlabeled_count))
            self.pseudo_acc.append(
                100.0 * pseudo_correct / pseudo_count if pseudo_count else 0.0
            )
            self.loss.append(ltot_sum / max(1, ltot_cnt))
            self.hc_diag.append(hc_diag)
            self.proto_diag.append(proto_diag)

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
            self.proto_g = self._optimize_global_prototypes(
                self.mean_protos,
                self.mean_valid,
            )
            self.global_valid = self.mean_valid.clone()

            # 5. 诊断仅观察无标签分类器与原型预测
            if self.proto_correct_conf_count:
                previous_threshold = torch.tensor(
                    self.proto_correct_conf_mean[-1], dtype=torch.float32
                )
                previous_threshold_valid = torch.tensor(
                    self.proto_correct_conf_count[-1], dtype=torch.float32
                ).gt(0) & torch.isfinite(previous_threshold)
            else:
                previous_threshold = torch.zeros(self.num_class)
                previous_threshold_valid = torch.zeros(self.num_class, dtype=torch.bool)
            self.proto_conf_threshold.append(previous_threshold.tolist())
            self.proto_conf_threshold_valid.append(previous_threshold_valid.tolist())
            diag_params = [
                ClientDiagnoseParams(
                    **asdict(base),
                    proto_g=self.proto_g.clone(),
                    global_valid=self.global_valid.clone(),
                    proto_scale=self.proto_scale,
                    proto_conf_threshold=previous_threshold.clone(),
                    proto_conf_threshold_valid=previous_threshold_valid.clone(),
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
            threshold_count = torch.stack(
                [diag_results[cid]["proto_conf_threshold_count"] for cid in selected]
            ).sum(dim=0)
            threshold_correct = torch.stack(
                [diag_results[cid]["proto_conf_threshold_correct"] for cid in selected]
            ).sum(dim=0)
            candidate_counts = torch.stack(
                [diag_results[cid]["proto_candidate_counts"] for cid in selected]
            ).sum(dim=0)
            low_candidate_counts = torch.stack(
                [diag_results[cid]["proto_low_candidate_counts"] for cid in selected]
            ).sum(dim=0)
            cand_size_counts = torch.stack(
                [diag_results[cid]["proto_candidate_size_counts"] for cid in selected]
            ).sum(dim=0)
            cand_size_correct = torch.stack(
                [diag_results[cid]["proto_candidate_size_correct"] for cid in selected]
            ).sum(dim=0)
            low_cand_size_counts = torch.stack(
                [
                    diag_results[cid]["proto_low_candidate_size_counts"]
                    for cid in selected
                ]
            ).sum(dim=0)
            low_cand_size_correct = torch.stack(
                [
                    diag_results[cid]["proto_low_candidate_size_correct"]
                    for cid in selected
                ]
            ).sum(dim=0)
            candidate_valid = bool(
                previous_threshold_valid.any()
                and self.global_valid.any()
                and candidate_counts[0] > 0
            )
            if candidate_valid:
                candidate_total = candidate_counts[0]
                candidate_rates = {
                    "empty": float(candidate_counts[1] / candidate_total * 100),
                    "single": float(candidate_counts[2] / candidate_total * 100),
                    "multi": float(candidate_counts[3] / candidate_total * 100),
                    "mean": float(candidate_counts[4] / candidate_total),
                    "true": float(candidate_counts[5] / candidate_total * 100),
                    "single_true": (
                        float(candidate_counts[6] / candidate_counts[2] * 100)
                        if candidate_counts[2] > 0
                        else float("nan")
                    ),
                    "multi_true": (
                        float(candidate_counts[7] / candidate_counts[3] * 100)
                        if candidate_counts[3] > 0
                        else float("nan")
                    ),
                    "single_accuracy": float(
                        candidate_counts[6] / candidate_counts[2] * 100
                    )
                    if candidate_counts[2] > 0
                    else float("nan"),
                    "multi_accuracy": float(
                        candidate_counts[7] / candidate_counts[3] * 100
                    )
                    if candidate_counts[3] > 0
                    else float("nan"),
                    "nonempty_accuracy": float(
                        (candidate_counts[6] + candidate_counts[7])
                        / (candidate_counts[2] + candidate_counts[3])
                        * 100
                    )
                    if candidate_counts[2] + candidate_counts[3] > 0
                    else float("nan"),
                }
            else:
                candidate_rates = {
                    key: float("nan")
                    for key in (
                        "empty",
                        "single",
                        "multi",
                        "mean",
                        "true",
                        "single_true",
                        "multi_true",
                        "single_accuracy",
                        "multi_accuracy",
                        "nonempty_accuracy",
                    )
                }
            self.proto_candidate_counts.append(candidate_counts.tolist())
            self.proto_candidate_valid.append(candidate_valid)
            self.proto_candidate_empty_rate.append(candidate_rates["empty"])
            self.proto_candidate_single_rate.append(candidate_rates["single"])
            self.proto_candidate_multi_rate.append(candidate_rates["multi"])
            self.proto_candidate_mean_count.append(candidate_rates["mean"])
            self.proto_candidate_true_coverage.append(candidate_rates["true"])
            self.proto_candidate_single_coverage.append(candidate_rates["single_true"])
            self.proto_candidate_multi_coverage.append(candidate_rates["multi_true"])
            self.proto_candidate_single_accuracy.append(
                candidate_rates["single_accuracy"]
            )
            self.proto_candidate_multi_accuracy.append(
                candidate_rates["multi_accuracy"]
            )
            self.proto_candidate_nonempty_accuracy.append(
                candidate_rates["nonempty_accuracy"]
            )
            low_candidate_valid = bool(candidate_valid and low_candidate_counts[0] > 0)
            if low_candidate_valid:
                low_total = low_candidate_counts[0]
                low_candidate_rates = {
                    "empty": float(low_candidate_counts[1] / low_total * 100),
                    "single": float(low_candidate_counts[2] / low_total * 100),
                    "multi": float(low_candidate_counts[3] / low_total * 100),
                    "mean": float(low_candidate_counts[4] / low_total),
                    "true": float(low_candidate_counts[5] / low_total * 100),
                    "single_true": (
                        float(low_candidate_counts[6] / low_candidate_counts[2] * 100)
                        if low_candidate_counts[2] > 0
                        else float("nan")
                    ),
                    "multi_true": (
                        float(low_candidate_counts[7] / low_candidate_counts[3] * 100)
                        if low_candidate_counts[3] > 0
                        else float("nan")
                    ),
                    "single_accuracy": float(
                        low_candidate_counts[6] / low_candidate_counts[2] * 100
                    )
                    if low_candidate_counts[2] > 0
                    else float("nan"),
                    "multi_accuracy": float(
                        low_candidate_counts[7] / low_candidate_counts[3] * 100
                    )
                    if low_candidate_counts[3] > 0
                    else float("nan"),
                    "nonempty_accuracy": float(
                        (low_candidate_counts[6] + low_candidate_counts[7])
                        / (low_candidate_counts[2] + low_candidate_counts[3])
                        * 100
                    )
                    if low_candidate_counts[2] + low_candidate_counts[3] > 0
                    else float("nan"),
                }
            else:
                low_candidate_rates = {
                    key: float("nan")
                    for key in (
                        "empty",
                        "single",
                        "multi",
                        "mean",
                        "true",
                        "single_true",
                        "multi_true",
                        "single_accuracy",
                        "multi_accuracy",
                        "nonempty_accuracy",
                    )
                }
            self.proto_low_candidate_counts.append(low_candidate_counts.tolist())
            self.proto_low_candidate_valid.append(low_candidate_valid)
            self.proto_low_candidate_empty_rate.append(low_candidate_rates["empty"])
            self.proto_low_candidate_single_rate.append(low_candidate_rates["single"])
            self.proto_low_candidate_multi_rate.append(low_candidate_rates["multi"])
            self.proto_low_candidate_mean_count.append(low_candidate_rates["mean"])
            self.proto_low_candidate_true_coverage.append(low_candidate_rates["true"])
            self.proto_low_candidate_single_coverage.append(
                low_candidate_rates["single_true"]
            )
            self.proto_low_candidate_multi_coverage.append(
                low_candidate_rates["multi_true"]
            )
            self.proto_low_candidate_single_accuracy.append(
                low_candidate_rates["single_accuracy"]
            )
            self.proto_low_candidate_multi_accuracy.append(
                low_candidate_rates["multi_accuracy"]
            )
            self.proto_low_candidate_nonempty_accuracy.append(
                low_candidate_rates["nonempty_accuracy"]
            )
            self.proto_candidate_size_counts.append(cand_size_counts.tolist())
            self.proto_candidate_size_correct.append(cand_size_correct.tolist())
            self.proto_low_candidate_size_counts.append(low_cand_size_counts.tolist())
            self.proto_low_candidate_size_correct.append(low_cand_size_correct.tolist())
            threshold_acc = [
                (float(correct) / float(count) * 100.0 if count > 0 else float("nan"))
                for count, correct in zip(threshold_count, threshold_correct)
            ]
            self.proto_conf_threshold_count.append(threshold_count.tolist())
            self.proto_conf_threshold_correct.append(threshold_correct.tolist())
            self.proto_conf_threshold_acc.append(threshold_acc)

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

            low_total = float(diag[9])
            low_conf_cls_acc = (
                100.0 * float(diag[10]) / low_total if low_total > 0 else 0.0
            )
            low_conf_proto_acc = (
                100.0 * float(diag[11]) / low_total if low_total > 0 else 0.0
            )
            low_conf_agree_rate = (
                100.0 * float(diag[12]) / low_total if low_total > 0 else 0.0
            )
            low_conf_agree_cls_acc = (
                100.0 * float(diag[13]) / float(diag[12]) if diag[12] > 0 else 0.0
            )
            cls95_rate = 100.0 * float(diag[14]) / max(1, total)
            cls95_acc = (
                f"{100.0 * float(diag[15]) / float(diag[14]):.2f}%"
                if diag[14] > 0
                else "N/A"
            )
            cls95_missing_rate = 100.0 * float(diag[16]) / max(1, mis_total)
            cls95_missing_acc = (
                f"{100.0 * float(diag[17]) / float(diag[16]):.2f}%"
                if diag[16] > 0
                else "N/A"
            )
            cls95_agree_rate = (
                100.0 * float(diag[18]) / float(diag[14]) if diag[14] > 0 else 0.0
            )
            cls95_agree_sample_rate = 100.0 * float(diag[18]) / max(1, total)
            cls95_agree_acc = (
                100.0 * float(diag[19]) / float(diag[18]) if diag[18] > 0 else 0.0
            )
            cls95_proto_ok_rate = (
                100.0 * float(diag[20]) / float(diag[14]) if diag[14] > 0 else 0.0
            )
            cls95_proto_ok_cls_acc = (
                100.0 * float(diag[21]) / float(diag[20]) if diag[20] > 0 else 0.0
            )
            discard_cnt = float(diag[14]) - float(diag[18])
            discard_ok_cnt = float(diag[15]) - float(diag[19])
            cls95_discard_rate = (
                100.0 * discard_cnt / float(diag[14]) if diag[14] > 0 else 0.0
            )
            cls95_discard_acc = (
                100.0 * discard_ok_cnt / discard_cnt if discard_cnt > 0 else 0.0
            )
            cls95_discard_mis_rate = (
                100.0 * discard_ok_cnt / float(diag[15]) if diag[15] > 0 else 0.0
            )
            conf_sum = diag[22 : 22 + self.num_class].tolist()
            conf_count = diag[22 + self.num_class :].tolist()
            conf_mean = [
                value / count if count > 0 else float("nan")
                for value, count in zip(conf_sum, conf_count)
            ]
            conf_mean_text = ", ".join(f"{value:.2f}" for value in conf_mean)
            self.low_conf_classifier_acc.append(low_conf_cls_acc)
            self.low_conf_proto_acc.append(low_conf_proto_acc)
            self.low_conf_agree_rate.append(low_conf_agree_rate)
            self.low_conf_agree_cls_acc.append(low_conf_agree_cls_acc)
            self.cls95_agree_rate.append(cls95_agree_rate)
            self.cls95_agree_sample_rate.append(cls95_agree_sample_rate)
            self.cls95_agree_acc.append(cls95_agree_acc)
            self.cls95_proto_ok_rate.append(cls95_proto_ok_rate)
            self.cls95_proto_ok_cls_acc.append(cls95_proto_ok_cls_acc)
            self.cls95_discard_rate.append(cls95_discard_rate)
            self.cls95_discard_acc.append(cls95_discard_acc)
            self.cls95_discard_mis_rate.append(cls95_discard_mis_rate)
            self.proto_correct_conf_mean.append(conf_mean)
            self.proto_correct_conf_sum.append(conf_sum)
            self.proto_correct_conf_count.append(conf_count)
            # 8. 测试集评估
            acc, acc_proto_mean, acc_proto = self._evaluate_accuracy()
            self.acc.append(acc)
            self.acc_proto_mean.append(acc_proto_mean)
            self.acc_proto.append(acc_proto)

            self.round_time.append(time.time() - round_start)

            # 9. 格式化控制台输出
            print(
                f"\n--- FedTest 第 {round_idx + 1}/{self.rounds} 轮 ---\n"
                f"测试准确率:\n"
                f"  分类器={self.acc[-1]:.2f}% | 目标原型={self.acc_proto[-1]:.2f}% | 统计原型={self.acc_proto_mean[-1]:.2f}%\n"
                f"训练损失:\n"
                f"  总损失={self.loss[-1]:.4f}\n"
                f"  CE={self.loss_x[-1]:.4f} | 原型校准={self.loss_calibrate[-1]:.4f}\n"
                f"  无标签CE={self.loss_x_u[-1]:.4f} | 无标签原型校准={self.loss_calibrate_u[-1]:.4f}\n"
                f"高置信伪标签:\n"
                f"  数量={self.pseudo_count[-1]} | 比例={self.pseudo_rate[-1]:.2f}% | 准确率={self.pseudo_acc[-1]:.2f}%\n"
                f"无标签分类:\n"
                f"  分类器={unlabeled_cls_acc:.2f}% (常见={seen_cls_acc:.2f}%, 缺失={mis_cls_acc:.2f}%)\n"
                f"  原型={unlabeled_prt_acc:.2f}% (常见={seen_prt_acc:.2f}%, 缺失={mis_prt_acc:.2f}%)\n"
                f"无标签样本数: 总数={total} | 常见类={seen_total} | 缺失类={mis_total}\n"
                f"分类器置信度>95%（最大 softmax）:\n"
                f"  样本率={cls95_rate:.2f}% | 准确率={cls95_acc}\n"
                f"  缺失类样本率={cls95_missing_rate:.2f}% | 缺失类准确率={cls95_missing_acc}\n"
                f"分类器置信度>95%且与原型预测一致（保留）:\n"
                f"  一致率={cls95_agree_rate:.2f}% | 样本率={cls95_agree_sample_rate:.2f}% | 准确率={cls95_agree_acc:.2f}%\n"
                f"分类器置信度>95%且原型预测正确:\n"
                f"  样本率={cls95_proto_ok_rate:.2f}% | 分类器准确率={cls95_proto_ok_cls_acc:.2f}%\n"
                f"分类器置信度>95%但与原型预测不一致（剔除）:\n"
                f"  剔除率={cls95_discard_rate:.2f}% | 剔除样本准确率={cls95_discard_acc:.2f}% | 正确样本误删率={cls95_discard_mis_rate:.2f}%\n"
                f"分类器置信度<95%（最大 softmax）:\n"
                f"  分类器准确率={low_conf_cls_acc:.2f}% | 原型准确率={low_conf_proto_acc:.2f}%\n"
                f"  一致率={low_conf_agree_rate:.2f}% | 一致样本分类器准确率={low_conf_agree_cls_acc:.2f}%\n"
                f"各类原型预测正确样本的原型预测置信度均值:\n"
                f"  [{conf_mean_text}]\n"
                f"本轮耗时: {self.round_time[-1]:.2f} 秒"
            )
            hc_rate = {
                key: (
                    "N/A"
                    if values[0] == 0
                    else f"{100.0 * values[1] / values[0]:.2f}%/"
                    f"{100.0 * values[2] / values[0]:.2f}%"
                )
                for key, values in hc_diag.items()
            }
            proto_rate = {
                key: (
                    "N/A"
                    if values[0] == 0
                    else f"{100.0 * values[1] / values[0]:.2f}%/"
                    f"{100.0 * values[2] / values[0]:.2f}%/"
                    f"{100.0 * values[3] / values[0]:.2f}%/"
                    f"{100.0 * values[4] / values[0]:.2f}%"
                )
                for key, values in proto_diag.items()
            }
            both_agree_rate = (
                "N/A"
                if hc_diag["b"][0] == 0
                else f"{100.0 * hc_diag['b'][3] / hc_diag['b'][0]:.2f}%"
            )
            both_agree_correct_rate = (
                "N/A"
                if hc_diag["b"][3] == 0
                else f"{100.0 * hc_diag['b'][4] / hc_diag['b'][3]:.2f}%"
            )
            print(
                "训练池 L∪U 分类器高置信（数量/本地准确率/全局准确率）：\n"
                f"  local={hc_diag['l'][0]}/{hc_rate['l']}，"
                f"global={hc_diag['g'][0]}/{hc_rate['g']}，"
                f"union={hc_diag['u'][0]}/{hc_rate['u']}\n"
                "训练池 L∪U 分类器分组（数量/本地准确率/全局准确率）：\n"
                f"  仅本地={hc_diag['lo'][0]}/{hc_rate['lo']}，"
                f"仅全局={hc_diag['go'][0]}/{hc_rate['go']}，"
                f"共同={hc_diag['b'][0]}/{hc_rate['b']}，"
                f"均未选={hc_diag['n'][0]}/{hc_rate['n']}\n"
                f"共同组：预测一致率={both_agree_rate}，"
                f"一致样本准确率={both_agree_correct_rate}\n"
                "训练池 L∪U 原型对照（数量/本地分类器准确率/原型准确率/一致率/一致且正确率）：\n"
                f"  全部={proto_diag['all'][0]}/{proto_rate['all']}\n"
                f"  本地高置信={proto_diag['local_high'][0]}/"
                f"{proto_rate['local_high']}"
            )
            threshold_text = "\n".join(
                (
                    f"  类{class_id}: T={previous_threshold[class_id].item():.4f}，"
                    f"候选={int(threshold_count[class_id])}，"
                    f"正确={int(threshold_correct[class_id])}，准确率="
                    f"{('N/A' if threshold_count[class_id] == 0 else f'{threshold_acc[class_id]:.2f}%')}"
                )
                if previous_threshold_valid[class_id]
                else f"  类{class_id}: T=N/A，候选=N/A，正确=N/A，准确率=N/A"
                for class_id in range(self.num_class)
            )
            print(f"纯U原型置信度阈值评估（使用上一轮T，按类）：\n{threshold_text}")
            if candidate_valid:
                candidate_line1 = (
                    f"空={int(candidate_counts[1])} "
                    f"({candidate_rates['empty']:.2f}%) | "
                    f"单={int(candidate_counts[2])} "
                    f"({candidate_rates['single']:.2f}%) | "
                    f"多={int(candidate_counts[3])} "
                    f"({candidate_rates['multi']:.2f}%) | "
                    f"平均候选数={candidate_rates['mean']:.2f}"
                )
                candidate_line2 = (
                    f"总体覆盖={candidate_rates['true']:.2f}% | "
                    f"单候选准确率={('N/A' if candidate_rates['single_accuracy'] != candidate_rates['single_accuracy'] else f'{candidate_rates["single_accuracy"]:.2f}%')} | "
                    f"多候选覆盖={('N/A' if candidate_rates['multi_true'] != candidate_rates['multi_true'] else f'{candidate_rates["multi_true"]:.2f}%')} | "
                    f"非空合并准确率={('N/A' if candidate_rates['nonempty_accuracy'] != candidate_rates['nonempty_accuracy'] else f'{candidate_rates["nonempty_accuracy"]:.2f}%')}"
                )
                cand_size_parts = [
                    f"{k}类={int(cand_size_counts[k])} "
                    f"({(cand_size_counts[k] / candidate_total * 100.0):.2f}%, "
                    f"准确率={(cand_size_correct[k] / cand_size_counts[k] * 100.0):.2f}%)"
                    for k in range(1, self.num_class + 1)
                    if cand_size_counts[k] > 0
                ]
                candidate_line3 = (
                    " | ".join(cand_size_parts) if cand_size_parts else "无非空候选"
                )
            else:
                candidate_line1 = "N/A"
                candidate_line2 = "N/A"
                candidate_line3 = "N/A"
            print(
                "纯U原型候选集（空/单/多/平均候选数）：\n"
                f"  {candidate_line1}\n"
                "纯U原型候选集（各候选集大小具体数量比例及准确率 [占全体, 准确率]）：\n"
                f"  {candidate_line3}\n"
                "纯U原型候选集（真实类覆盖与准确率）：\n"
                f"  {candidate_line2}"
            )
            if low_candidate_valid:
                low_candidate_line1 = (
                    f"空={int(low_candidate_counts[1])} "
                    f"({low_candidate_rates['empty']:.2f}%) | "
                    f"单={int(low_candidate_counts[2])} "
                    f"({low_candidate_rates['single']:.2f}%) | "
                    f"多={int(low_candidate_counts[3])} "
                    f"({low_candidate_rates['multi']:.2f}%) | "
                    f"平均候选数={low_candidate_rates['mean']:.2f}"
                )
                low_candidate_line2 = (
                    f"总体覆盖={low_candidate_rates['true']:.2f}% | "
                    f"单候选准确率={('N/A' if low_candidate_rates['single_accuracy'] != low_candidate_rates['single_accuracy'] else f'{low_candidate_rates["single_accuracy"]:.2f}%')} | "
                    f"多候选覆盖={('N/A' if low_candidate_rates['multi_true'] != low_candidate_rates['multi_true'] else f'{low_candidate_rates["multi_true"]:.2f}%')} | "
                    f"非空合并准确率={('N/A' if low_candidate_rates['nonempty_accuracy'] != low_candidate_rates['nonempty_accuracy'] else f'{low_candidate_rates["nonempty_accuracy"]:.2f}%')}"
                )
                low_cand_size_parts = [
                    f"{k}类={int(low_cand_size_counts[k])} "
                    f"({(low_cand_size_counts[k] / low_total * 100.0):.2f}%, "
                    f"准确率={(low_cand_size_correct[k] / low_cand_size_counts[k] * 100.0):.2f}%)"
                    for k in range(1, self.num_class + 1)
                    if low_cand_size_counts[k] > 0
                ]
                low_candidate_line3 = (
                    " | ".join(low_cand_size_parts)
                    if low_cand_size_parts
                    else "无非空候选"
                )
            else:
                low_candidate_line1 = "N/A"
                low_candidate_line2 = "N/A"
                low_candidate_line3 = "N/A"
            print(
                "全局分类器低置信纯U原型候选集（空/单/多/平均候选数）：\n"
                f"  {low_candidate_line1}\n"
                "全局分类器低置信纯U原型候选集（各候选集大小具体数量比例及准确率 [占全体, 准确率]）：\n"
                f"  {low_candidate_line3}\n"
                "全局分类器低置信纯U原型候选集（真实类覆盖与准确率）：\n"
                f"  {low_candidate_line2}"
            )

    def save(self):
        metrics = {
            "acc": self.acc,
            "acc_proto": self.acc_proto,
            "acc_proto_mean": self.acc_proto_mean,
            "loss": self.loss,
            "loss_x": self.loss_x,
            "loss_calibrate": self.loss_calibrate,
            "loss_x_u": self.loss_x_u,
            "loss_calibrate_u": self.loss_calibrate_u,
            "pseudo_count": self.pseudo_count,
            "pseudo_rate": self.pseudo_rate,
            "pseudo_acc": self.pseudo_acc,
            "round_time": self.round_time,
            "unlabeled_classifier_acc": self.unlabeled_classifier_acc,
            "unlabeled_proto_acc": self.unlabeled_proto_acc,
            "seen_classifier_acc": self.seen_classifier_acc,
            "seen_proto_acc": self.seen_proto_acc,
            "missing_classifier_acc": self.missing_classifier_acc,
            "missing_proto_acc": self.missing_proto_acc,
            "low_conf_classifier_acc": self.low_conf_classifier_acc,
            "low_conf_proto_acc": self.low_conf_proto_acc,
            "low_conf_agree_rate": self.low_conf_agree_rate,
            "low_conf_agree_cls_acc": self.low_conf_agree_cls_acc,
            "cls95_agree_rate": self.cls95_agree_rate,
            "cls95_agree_sample_rate": self.cls95_agree_sample_rate,
            "cls95_agree_acc": self.cls95_agree_acc,
            "cls95_proto_ok_rate": self.cls95_proto_ok_rate,
            "cls95_proto_ok_cls_acc": self.cls95_proto_ok_cls_acc,
            "cls95_discard_rate": self.cls95_discard_rate,
            "cls95_discard_acc": self.cls95_discard_acc,
            "cls95_discard_mis_rate": self.cls95_discard_mis_rate,
            "proto_correct_conf_mean": self.proto_correct_conf_mean,
            "proto_correct_conf_sum": self.proto_correct_conf_sum,
            "proto_correct_conf_count": self.proto_correct_conf_count,
            "hc_diag": self.hc_diag,
            "proto_diag": self.proto_diag,
            "proto_conf_threshold": self.proto_conf_threshold,
            "proto_conf_threshold_valid": self.proto_conf_threshold_valid,
            "proto_conf_threshold_count": self.proto_conf_threshold_count,
            "proto_conf_threshold_correct": self.proto_conf_threshold_correct,
            "proto_conf_threshold_acc": self.proto_conf_threshold_acc,
            "proto_candidate_counts": self.proto_candidate_counts,
            "proto_candidate_valid": self.proto_candidate_valid,
            "proto_candidate_empty_rate": self.proto_candidate_empty_rate,
            "proto_candidate_single_rate": self.proto_candidate_single_rate,
            "proto_candidate_multi_rate": self.proto_candidate_multi_rate,
            "proto_candidate_mean_count": self.proto_candidate_mean_count,
            "proto_candidate_true_coverage": self.proto_candidate_true_coverage,
            "proto_candidate_single_coverage": self.proto_candidate_single_coverage,
            "proto_candidate_multi_coverage": self.proto_candidate_multi_coverage,
            "proto_candidate_single_accuracy": self.proto_candidate_single_accuracy,
            "proto_candidate_multi_accuracy": self.proto_candidate_multi_accuracy,
            "proto_candidate_nonempty_accuracy": self.proto_candidate_nonempty_accuracy,
            "proto_low_candidate_counts": self.proto_low_candidate_counts,
            "proto_low_candidate_valid": self.proto_low_candidate_valid,
            "proto_low_candidate_empty_rate": self.proto_low_candidate_empty_rate,
            "proto_low_candidate_single_rate": self.proto_low_candidate_single_rate,
            "proto_low_candidate_multi_rate": self.proto_low_candidate_multi_rate,
            "proto_low_candidate_mean_count": self.proto_low_candidate_mean_count,
            "proto_low_candidate_true_coverage": self.proto_low_candidate_true_coverage,
            "proto_low_candidate_single_coverage": self.proto_low_candidate_single_coverage,
            "proto_low_candidate_multi_coverage": self.proto_low_candidate_multi_coverage,
            "proto_low_candidate_single_accuracy": self.proto_low_candidate_single_accuracy,
            "proto_low_candidate_multi_accuracy": self.proto_low_candidate_multi_accuracy,
            "proto_low_candidate_nonempty_accuracy": self.proto_low_candidate_nonempty_accuracy,
            "proto_candidate_size_counts": self.proto_candidate_size_counts,
            "proto_candidate_size_correct": self.proto_candidate_size_correct,
            "proto_low_candidate_size_counts": self.proto_low_candidate_size_counts,
            "proto_low_candidate_size_correct": self.proto_low_candidate_size_correct,
        }

        self.deal_save(
            metrics,
            {
                "global": self.model.state_dict(),
                "proto": self.proto_g,
                "proto_mean": self.mean_protos,
            },
        )
