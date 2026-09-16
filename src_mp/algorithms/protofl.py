"""ProtoFL：以全局原型为低置信样本构建候选标签集的联邦 SSL。"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.algorithms.utils.input import prepare_input_batch
from .core.model import CNN
from .core import (
    BaseClientExecutor,
    BaseServer,
    ClientResult,
    EvalResult,
    clone_state,
    dist_contrastive_loss,
)
from .core.augment import strong_augment, weak_augment


class ProtoFLCNN(CNN):
    """仅供 ProtoFL 使用的 CNN：暴露特征并增加原型投影头。"""

    def __init__(self, input_channels, num_classes, feature_dim, dataset_name):
        super().__init__(input_channels, num_classes, feature_dim, dataset_name)
        self.feat_proj = nn.Linear(feature_dim, feature_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature = self.extractor(x)
        return feature, self.classifier(feature)


def build_protofl_model(args, num_class: int) -> ProtoFLCNN:
    """在本模块构建 ProtoFL 模型，不影响共享模型工厂。"""
    if args.model != "cnn":
        raise ValueError("ProtoFL 当前仅支持 model='cnn'。")
    rgb = {
        "tiny_imagenet",
        "flowers102",
        "cars",
        "gtsrb",
        "cinic10",
        "svhn",
        "pacs",
        "officehome",
        "vlcs",
        "domainnet",
    }
    channels = 3 if "cifar" in args.dataset or args.dataset in rgb else 1
    return ProtoFLCNN(channels, num_class, args.feature_dim, args.dataset)


class Client(BaseClientExecutor):
    """使用全局原型为低置信无标签样本提供候选类的客户端。"""

    def __init__(self, args, device, num_class, **kwargs):
        super().__init__(args, device, num_class, **kwargs)
        self.model = build_protofl_model(args, num_class).to(device)
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
        self.model_g = build_protofl_model(args, num_class).to(device).eval()
        for parameter in self.model_g.parameters():
            parameter.requires_grad_(False)
        self.conf = args.conf
        self.lambda_u = args.lambda_u
        self.lambda_p = args.lambda_p
        self.temperature = args.temperature
        self.unlabeled_ratio = args.unlabeled_ratio

    def train(self):
        payload = cast(dict[str, torch.Tensor], self.payload)
        self.model.train()
        self.model_g.load_state_dict(clone_state(self.model.state_dict()))
        self.model_g.eval()
        self.global_prototypes = payload["global_prototypes"].to(self.device)
        self.prototype_valid = payload["prototype_valid"].to(self.device).bool()
        train_set = self.current_task.train_set
        labeled_mask = train_set.is_labeled.bool()
        if not labeled_mask.any() or not (~labeled_mask).any():
            raise ValueError("ProtoFL 要求每个客户端同时具有标注和无标签样本")
        labeled_loader = DataLoader(
            TensorDataset(train_set.x[labeled_mask], train_set.y[labeled_mask]),
            batch_size=self.batch_size,
            shuffle=True,
        )
        unlabeled_loader = DataLoader(
            TensorDataset(train_set.x[~labeled_mask], train_set.y[~labeled_mask]),
            batch_size=self.batch_size * self.unlabeled_ratio,
            shuffle=True,
        )
        total = 0.0
        batches = 0
        for _ in range(self.epochs):
            total, batches = self.run_epoch(
                total, batches, labeled_loader, unlabeled_loader
            )
        prototypes, prototype_counts = self._local_prototypes()
        return ClientResult(
            self.client_id,
            total / max(1, batches),
            clone_state(self.model.state_dict()),
            {
                "prototypes": prototypes,
                "prototype_counts": prototype_counts,
            },
        )

    def run_epoch(
        self,
        total: float,
        batches: int,
        labeled_loader: DataLoader,
        unlabeled_loader: DataLoader,
    ):
        labeled_iterator = iter(labeled_loader)
        for x_u, _ in unlabeled_loader:
            try:
                x_l, y_l = next(labeled_iterator)
            except StopIteration:
                labeled_iterator = iter(labeled_loader)
                x_l, y_l = next(labeled_iterator)
            x_l, y_l = x_l.to(self.device), y_l.to(self.device)
            x_u = x_u.to(self.device)
            x_l_w = weak_augment(x_l, self.dataset)
            x_u_w = weak_augment(x_u, self.dataset)
            x_u_s = strong_augment(x_u, self.dataset)

            _, logits_l = self.model(x_l_w)
            feats_u_w, logits_u_w = self.model(x_u_w)
            feats_u_s, logits_u_s = self.model(x_u_s)
            loss = F.cross_entropy(logits_l, y_l)
            with torch.no_grad():
                _, logits_u_global = self.model_g(x_u_w)
                probs_u_global = torch.softmax(
                    logits_u_global / self.temperature, dim=1
                )
                probs_u_local = torch.softmax(
                    logits_u_w.detach() / self.temperature, dim=1
                )
                confidence_global, targets_u_global = probs_u_global.max(dim=1)
                confidence_local = probs_u_local.max(dim=1).values
                high_confidence = confidence_local.ge(self.conf) | confidence_global.ge(
                    self.conf
                )

            # 高置信样本完全复用 ProxyFL 的全局硬伪标签监督。
            loss_u = F.cross_entropy(logits_u_s, targets_u_global, reduction="none")
            loss = loss + self.lambda_u * (loss_u * high_confidence).mean()

            # 低置信样本的候选类别只由全局原型决定；候选掩码脱离计算图。
            low_confidence = ~high_confidence
            if low_confidence.any() and self.prototype_valid.any():
                loss_proto = self._candidate_proto_loss(
                    self.model.feat_proj(feats_u_w).detach(),
                    self.model.feat_proj(feats_u_s),
                    low_confidence,
                )
                loss = loss + self.lambda_p * loss_proto

            self.check_nan(loss)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total += loss.item()
            batches += 1
        return total, batches

    def _candidate_proto_loss(
        self,
        weak_projections: torch.Tensor,
        strong_projections: torch.Tensor,
        low_confidence: torch.Tensor,
    ) -> torch.Tensor:
        """以 ``softmax(-L2) >= 1/C`` 展开候选类并复用 L2-CE 对比损失。"""
        valid_ids = torch.where(self.prototype_valid)[0]
        prototypes = self.global_prototypes[valid_ids]
        weak_low = weak_projections[low_confidence]
        strong_low = strong_projections[low_confidence]
        distance_logits = -torch.cdist(weak_low, prototypes, p=2.0)
        prototype_probabilities = torch.softmax(distance_logits, dim=1)
        # 固定全局类别基线，而非有效原型数的倒数。
        candidates = prototype_probabilities.ge(1.0 / self.num_class)
        sample_indices, candidate_indices = torch.where(candidates)
        if sample_indices.numel() == 0:
            return strong_projections.sum() * 0.0
        candidate_features = strong_low[sample_indices]
        candidate_loss = dist_contrastive_loss(
            candidate_features,
            prototypes,
            candidate_indices,
        )
        return candidate_loss

    @torch.no_grad()
    def _local_prototypes(self) -> tuple[torch.Tensor, torch.Tensor]:
        """仅用真实 L，在投影空间计算客户端每类原型及其样本数。"""
        train_set = self.current_task.train_set
        labeled = train_set.is_labeled.bool()
        dataset = TensorDataset(train_set.x[labeled], train_set.y[labeled])
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        sums = torch.zeros(self.num_class, self.model.classifier.in_features)
        counts = torch.zeros(self.num_class, dtype=torch.long)
        self.model.eval()
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            features, _ = self.model(prepare_input_batch(x, self.dataset))
            projections = self.model.feat_proj(features)
            sums.index_add_(0, y.cpu(), projections.cpu())
            counts += torch.bincount(y.cpu(), minlength=self.num_class)
        self.model.train()
        prototypes = torch.zeros_like(sums)
        valid = counts > 0
        prototypes[valid] = sums[valid] / counts[valid].unsqueeze(1)
        return prototypes, counts

    def evaluate(self, task):
        if task.eval_id not in self.eval_loaders:
            self.eval_loaders[task.eval_id] = DataLoader(
                task.test_set, batch_size=128, shuffle=False
            )
        self.model.load_state_dict(task.model_state)
        self.model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y, *_ in self.eval_loaders[task.eval_id]:
                x = x.to(self.device)
                if self.is_ssl:
                    x = prepare_input_batch(x, self.dataset)
                _, logits = self.model(x)
                correct += logits.argmax(dim=1).eq(y.to(self.device)).sum().item()
                total += y.size(0)
        return EvalResult(task.eval_id, correct, total)


class Server(BaseServer):
    client_cls = Client
    supports_ssl = True

    def __init__(self, args, devices):
        if args.ssl in ("none", "client"):
            raise ValueError(
                "ProtoFL 仅支持 sample、double、sfd 等每客户端含 L/U 的 SSL 模式。"
            )
        super().__init__(args, devices)
        self.args = args
        self.model = build_protofl_model(args, self.num_class)
        self.global_prototypes = torch.zeros(self.num_class, args.feature_dim)
        self.prototype_valid = torch.zeros(self.num_class, dtype=torch.bool)

    def train_payloads(self):
        return {
            client_id: {
                "global_prototypes": self.global_prototypes.clone(),
                "prototype_valid": self.prototype_valid.clone(),
            }
            for client_id in self.selected
        }

    def _aggregate_prototypes(self, results):
        prototype_sums = torch.zeros_like(self.global_prototypes)
        prototype_counts = torch.zeros(self.num_class, dtype=torch.long)
        for client_id in self.selected:
            payload = results[client_id].payload
            prototypes = payload["prototypes"]
            counts = payload["prototype_counts"].long()
            prototype_sums += prototypes * counts.unsqueeze(1)
            prototype_counts += counts
        self.global_prototypes[valid] = prototype_sums[valid] / prototype_counts[
            valid
        ].unsqueeze(1)
        )
        self.prototype_valid |= valid

    def apply_result(self, results):
        self.aggregate_model(results)
        self._aggregate_prototypes(results)

    def run_round(self):
        results = self.train_clients()
        self.apply_result(results)
        loss = sum(results[client_id].loss for client_id in self.selected) / len(
            self.selected
        )
        self.record_round(loss, self.evaluate())
