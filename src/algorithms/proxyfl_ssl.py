"""ProxyFL 的半监督版本。

该实现对应 ``scratch/ProxyFL`` 的训练闭环，但使用项目现有模型和
``MetaDataset`` 接口。ICPL 直接在模型 extractor 输出的特征空间中计算，
不引入源码中的 feat_proj / proxy_proj。
"""

import os
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .utils import BaseParams, BaseServer, fmt_num, get_model, param_aggregate
from .utils.augment import sage_strong_augment, sage_weak_augment
from .utils.ssl import build_fixmatch_loaders, iterate_ssl_batches


def get_path(args):
    args.file_name = (
        f"{args.common_name}_{fmt_num(args.temperature)}"
        f"_{fmt_num(args.gpt_lr)}_{fmt_num(args.gpt_epochs)}"
        f"_{fmt_num(args.gpt_batch_size)}_{fmt_num(args.gpt_threshold)}"
        f"_{fmt_num(args.ema_beta)}"
    )
    return os.path.join(args.log_path, f"{args.file_name}_{args.cur_time}.log")


@dataclass
class Params(BaseParams):
    global_class_dist: torch.Tensor
    confidence: float
    lam: float
    temperature: float
    unlabeled_ratio: int


def icpl_loss_with_proxy(
    features,
    class_sets,
    proxy_weights,
    single_proxy_mask,
    classifier_weight,
    query_mask,
):
    """在显式关系池上计算使用 classifier.weight 的简化 ICPL。"""
    if not query_mask.any():
        return features.sum() * 0.0

    feature = F.normalize(features, p=2, dim=1)
    proxy = F.normalize(classifier_weight, p=2, dim=1)
    candidate_proxy = proxy_weights @ proxy
    single_class = class_sets.float().argmax(dim=1)
    single_proxy = proxy[single_class]
    positive_proxy = torch.where(
        single_proxy_mask.unsqueeze(1), single_proxy, candidate_proxy
    )
    positive = (feature * positive_proxy).sum(dim=1)

    overlap = (class_sets.unsqueeze(1) & class_sets.unsqueeze(0)).any(dim=2)
    negative_mask = ~overlap
    negative_mask.fill_diagonal_(False)
    negative_mask = negative_mask[query_mask]
    pairwise_sim = feature @ feature.T
    query_pairwise_sim = pairwise_sim[query_mask]
    negative_mask &= query_pairwise_sim >= 1e-6
    negative = query_pairwise_sim.masked_fill(~negative_mask, float("-inf"))

    logits = torch.cat((positive[query_mask].unsqueeze(1), negative), dim=1)

    labels = torch.zeros(logits.size(0), dtype=torch.long, device=features.device)
    return F.cross_entropy(logits, labels)


def train(p: Params):
    device = torch.device(p.client_gpu)
    model = get_model(p.model_name, p.dataset, p.num_class, p.feature_dim).to(device)
    model.load_state_dict(p.model_state)

    global_model = get_model(p.model_name, p.dataset, p.num_class, p.feature_dim).to(
        device
    )
    global_model.load_state_dict(p.model_state)
    global_model.eval()
    for parameter in global_model.parameters():
        parameter.requires_grad_(False)

    loaders = build_fixmatch_loaders(
        p.train_set,
        p.batch_size,
        p.unlabeled_ratio,
    )

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=p.lr,
        momentum=p.momentum,
        weight_decay=p.weight_decay,
    )
    class_counts = torch.zeros(p.num_class, device=device)
    pseudo_total = 0
    pseudo_correct = 0
    pseudo_valid = 0
    total_loss = 0.0
    num_batches = 0

    model.train()
    for local_epoch in range(p.epochs):
        for labeled_batch, unlabeled_batch in iterate_ssl_batches(loaders):
            assert labeled_batch is not None
            x_l, y_l = labeled_batch
            x_u, y_u = unlabeled_batch
            x_l, y_l = x_l.to(device), y_l.to(device)
            x_u, y_u = x_u.to(device), y_u.to(device)
            x_l = sage_weak_augment(x_l, p.dataset)
            x_u_w = sage_weak_augment(x_u, p.dataset)
            x_u_s = sage_strong_augment(x_u, p.dataset)

            labeled_batch_size = x_l.size(0)
            unlabeled_batch_size = x_u.size(0)
            inputs = torch.cat((x_l, x_u_w, x_u_s))
            features = model.extractor(inputs)
            logits = model.classifier(features)

            unlabeled_start = labeled_batch_size
            strong_start = labeled_batch_size + unlabeled_batch_size
            logits_l = logits[:unlabeled_start]
            logits_u_w = logits[unlabeled_start:strong_start]
            logits_u_s = logits[strong_start:]
            features_l = features[:unlabeled_start]
            features_u_w = features[unlabeled_start:strong_start]
            features_u_s = features[strong_start:]
            loss_x = F.cross_entropy(logits_l, y_l)

            with torch.no_grad():
                global_logits_u = global_model(x_u_w)
                probs_u_global = torch.softmax(global_logits_u / p.temperature, dim=-1)
                max_probs_global, targets_u_global = probs_u_global.max(dim=1)

            probs_u_local = torch.softmax(logits_u_w.detach() / p.temperature, dim=-1)
            max_probs_local, _ = probs_u_local.max(dim=1)
            mask_valid = torch.maximum(
                max_probs_local.ge(p.confidence).float(),
                max_probs_global.ge(p.confidence).float(),
            )
            targets_global_one_hot = F.one_hot(targets_u_global, p.num_class).float()
            strong_probs = torch.softmax(logits_u_s, dim=-1)
            loss_u = (
                F.kl_div(
                    (strong_probs + 1e-10).log(),
                    targets_global_one_hot + 1e-10,
                    reduction="none",
                ).sum(dim=1)
                * mask_valid
            ).mean()

            icpl_features = torch.cat((features_l, features_u_w, features_u_s), dim=0)
            class_set_l = F.one_hot(y_l, p.num_class).bool()
            proxy_weight_l = class_set_l.float()
            single_proxy_l = torch.ones(len(y_l), dtype=torch.bool, device=device)
            single_proxy_u = mask_valid.bool()
            global_class_dist = p.global_class_dist.to(
                device=probs_u_global.device, dtype=probs_u_global.dtype
            )
            indecisive_set_u = probs_u_global > global_class_dist.unsqueeze(0)
            global_class_u = F.one_hot(targets_u_global, p.num_class).bool()
            class_set_u = torch.where(
                single_proxy_u.unsqueeze(1), global_class_u, indecisive_set_u
            )
            proxy_weight_u = torch.where(
                single_proxy_u.unsqueeze(1),
                global_class_u.float(),
                probs_u_global * indecisive_set_u.float(),
            ).detach()
            u_active = class_set_u.any(dim=1)
            pool_mask = torch.cat(
                (
                    torch.ones(len(y_l), dtype=torch.bool, device=device),
                    u_active,
                    u_active,
                )
            )
            query_mask = torch.cat(
                (
                    torch.zeros(len(y_l), dtype=torch.bool, device=device),
                    u_active,
                    u_active,
                )
            )
            class_sets = torch.cat((class_set_l, class_set_u, class_set_u), dim=0)
            proxy_weights = torch.cat(
                (proxy_weight_l, proxy_weight_u, proxy_weight_u), dim=0
            )
            single_proxy_mask = torch.cat(
                (single_proxy_l, single_proxy_u, single_proxy_u), dim=0
            )
            icpl_features = icpl_features[pool_mask]
            class_sets = class_sets[pool_mask]
            proxy_weights = proxy_weights[pool_mask]
            single_proxy_mask = single_proxy_mask[pool_mask]
            query_mask = query_mask[pool_mask]
            loss_c = icpl_loss_with_proxy(
                icpl_features,
                class_sets,
                proxy_weights,
                single_proxy_mask,
                model.classifier.weight,
                query_mask,
            )
            loss = loss_x + p.lam * loss_u + p.lam * loss_c

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if local_epoch == p.epochs - 1:
                class_counts += F.one_hot(y_l, p.num_class).float().sum(dim=0)
                valid = mask_valid.bool()
                class_counts += (
                    F.one_hot(targets_u_global[valid], p.num_class).float().sum(dim=0)
                )
                pseudo_total += len(targets_u_global)
                pseudo_correct += (targets_u_global == y_u).sum().item()
                pseudo_valid += valid.sum().item()
            total_loss += loss.item()
            num_batches += 1

    state = {k: v.cpu().detach().clone() for k, v in model.state_dict().items()}
    return {
        "loss": total_loss / max(1, num_batches),
        "state": state,
        "num_samples": len(p.train_set.y),
        "class_counts": class_counts.cpu().numpy(),
        "pseudo_total": pseudo_total,
        "pseudo_correct": pseudo_correct,
        "pseudo_valid": pseudo_valid,
    }


class Server(BaseServer):
    def __init__(self, args):
        supported_ssl = ("sample", "double", "sfd")
        if args.ssl not in supported_ssl:
            raise ValueError(
                "proxyfl_ssl 要求 ssl 为 sample、double 或 sfd；"
                "每个客户端必须同时包含有标签和无标签数据"
            )
        super().__init__(args, is_ssl=True, pfl=False)
        self.temperature = args.temperature
        self.unlabeled_ratio = args.unlabeled_ratio
        self.gpt_epochs = args.gpt_epochs
        self.gpt_batch_size = args.gpt_batch_size
        self.gpt_threshold = args.gpt_threshold
        self.ema_beta = args.ema_beta

        self.gpt = nn.Linear(self.feature_dim, self.num_class).to(self.device)
        self.gpt_optimizer = torch.optim.SGD(self.gpt.parameters(), lr=args.gpt_lr)
        self.global_class_dist = torch.full(
            (self.num_class,), 1.0 / self.num_class, dtype=torch.float32
        )
        self.pseudo_acc = []
        self.valid_ratio = []
        self.num_valid = []
        self.fedavg_acc = []
        self.fedavg_pred_dist = []
        self.gpt_pred_dist = []
        self.gpt_loss = []
        self.gpt_min_class_distance = []

    def _build_params(self, selected):
        return [
            Params(
                **asdict(base),
                global_class_dist=self.global_class_dist,
                confidence=self.confidence,
                lam=self.lam,
                temperature=self.temperature,
                unlabeled_ratio=self.unlabeled_ratio,
            )
            for base in self.build_base_params(selected)
        ]

    def _update_global_distribution(self, counts):
        total = torch.from_numpy(np.stack(counts).astype(np.float32)).sum(dim=0)
        if total.sum() > 0:
            current = total / total.sum()
            self.global_class_dist = (
                self.ema_beta * self.global_class_dist + (1.0 - self.ema_beta) * current
            )

    def _diagnose_global_model(self):
        """返回当前全局模型的准确率与预测类别分布，不改变模型参数。"""
        loader = DataLoader(self.test_set, batch_size=128, shuffle=False)
        correct = 0
        total = 0
        pred_counts = torch.zeros(self.num_class, dtype=torch.long)

        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            for x, y, *_ in loader:
                logits = self.model(x.to(self.device))
                prediction = logits.argmax(dim=1)
                target = y.to(self.device)
                correct += prediction.eq(target).sum().item()
                total += target.numel()
                pred_counts += torch.bincount(
                    prediction.cpu(), minlength=self.num_class
                )
        self.model.cpu()

        accuracy = 100.0 * correct / max(1, total)
        distribution = (pred_counts.float() / max(1, total)).numpy()
        return accuracy, distribution

    def _update_gpt(self, states, weights):
        classifier_states = [
            {
                "weight": state["classifier.weight"],
                "bias": state["classifier.bias"],
            }
            for state in states
        ]
        avg_classifier = param_aggregate(classifier_states, weights)
        self.gpt.load_state_dict(avg_classifier)

        proxies = torch.cat([state["classifier.weight"] for state in states], dim=0).to(
            self.device
        )
        labels = torch.arange(self.num_class, device=self.device).repeat(len(states))
        loader = DataLoader(
            TensorDataset(proxies, labels),
            batch_size=self.gpt_batch_size,
            shuffle=True,
        )
        with torch.no_grad():
            avg_weight = avg_classifier["weight"].to(self.device)
            distances = torch.cdist(avg_weight, avg_weight, p=2)
            distances.fill_diagonal_(float("inf"))
            max_dist = distances.min(dim=-1).values.max()

        self.gpt.train()
        total_loss = 0.0
        num_batches = 0
        for _ in range(self.gpt_epochs):
            for proxy, label in loader:
                distance = torch.cdist(proxy, self.gpt.weight, p=2)
                penalty = min(max_dist.item(), self.gpt_threshold)
                distance = (
                    distance
                    + F.one_hot(label, self.num_class).to(distance.dtype) * penalty
                )
                loss = F.cross_entropy(-distance, label)
                self.gpt_optimizer.zero_grad()
                loss.backward()
                self.gpt_optimizer.step()
                total_loss += loss.item()
                num_batches += 1
        self.gpt.eval()

        global_state = self.model.state_dict()
        global_state["classifier.weight"] = self.gpt.weight.detach().cpu().clone()
        global_state["classifier.bias"] = self.gpt.bias.detach().cpu().clone()
        self.model.load_state_dict(global_state)

        with torch.no_grad():
            distances = torch.cdist(self.gpt.weight, self.gpt.weight, p=2)
            distances.fill_diagonal_(float("inf"))
            min_class_distance = distances.min().item()
        return total_loss / max(1, num_batches), min_class_distance

    def fit(self):
        num_join = max(1, int(self.num_clients * self.join_ratio))
        for round_id in range(self.rounds):
            start = time.time()
            selected = sorted(torch.randperm(self.num_clients)[:num_join].tolist())
            results = self.run_clients(train, self._build_params(selected))

            states = [results[cid]["state"] for cid in selected]
            sample_counts = [results[cid]["num_samples"] for cid in selected]
            total_samples = max(1, sum(sample_counts))
            weights = [count / total_samples for count in sample_counts]
            self.model.load_state_dict(param_aggregate(states, weights))
            fedavg_acc, fedavg_pred_dist = self._diagnose_global_model()
            self._update_global_distribution(
                [results[cid]["class_counts"] for cid in selected]
            )
            gpt_loss, min_class_distance = self._update_gpt(states, weights)
            _, gpt_pred_dist = self._diagnose_global_model()

            self.fedavg_acc.append(fedavg_acc)
            self.fedavg_pred_dist.append(fedavg_pred_dist)
            self.gpt_pred_dist.append(gpt_pred_dist)
            self.gpt_loss.append(gpt_loss)
            self.gpt_min_class_distance.append(min_class_distance)

            total_loss = sum(results[cid]["loss"] for cid in selected)
            total_pseudo = sum(results[cid]["pseudo_total"] for cid in selected)
            correct_pseudo = sum(results[cid]["pseudo_correct"] for cid in selected)
            valid_pseudo = sum(results[cid]["pseudo_valid"] for cid in selected)
            self.loss.append(total_loss / num_join)
            self.evaluate()
            pseudo_acc = correct_pseudo / max(1, total_pseudo)
            valid_ratio = valid_pseudo / max(1, total_pseudo)
            self.pseudo_acc.append(pseudo_acc)
            self.valid_ratio.append(valid_ratio)
            self.num_valid.append(valid_pseudo)
            print(f"\n--- ProxyFL-SSL Round {round_id + 1}/{self.rounds} ---")
            print(
                f"Global Accuracy: {self.acc[-1]:.2f}%, "
                f"Avg Loss: {self.loss[-1]:.4f}, "
                f"Pseudo Acc: {pseudo_acc:.4f}, "
                f"Valid Ratio: {valid_ratio:.4f}, "
                f"Time: {time.time() - start:.2f}s"
            )
            print(
                f"[Diagnosis] FedAvg Acc: {fedavg_acc:.2f}%, "
                f"GPT Loss: {gpt_loss:.4f}, "
                f"GPT Min Class Dist: {min_class_distance:.4f}"
            )
            print(
                "[Diagnosis] Pred Dist (FedAvg -> GPT): "
                f"{np.array2string(fedavg_pred_dist, precision=3)} -> "
                f"{np.array2string(gpt_pred_dist, precision=3)}"
            )

    def save(self):
        self.deal_save(
            {
                "acc": self.acc,
                "loss": self.loss,
                "pseudo_acc": self.pseudo_acc,
                "valid_ratio": self.valid_ratio,
                "num_valid": self.num_valid,
                "fedavg_acc": self.fedavg_acc,
                "fedavg_pred_dist": self.fedavg_pred_dist,
                "gpt_pred_dist": self.gpt_pred_dist,
                "gpt_loss": self.gpt_loss,
                "gpt_min_class_distance": self.gpt_min_class_distance,
                "global_class_dist": self.global_class_dist.numpy(),
            },
            {"global_model": self.model.state_dict(), "gpt": self.gpt.state_dict()},
        )
