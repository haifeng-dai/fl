"""ProxyFL：基于项目 CNN 的代理分类器联邦半监督学习实现。

模型和训练逻辑均限定在本模块：``ProxyFLCNN`` 继承项目的 CNN 主干，
仅为 ProxyFL 增加投影头和特征输出，不改变共享模型工厂及其他算法。
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.algorithms.utils.input import prepare_input_batch
from src.models import CNN

from .core import BaseClientExecutor, BaseServer, ClientResult, EvalResult, clone_state
from .core.augment import strong_augment, weak_augment


class ProxyFLCNN(CNN):
    """项目 CNN 的 ProxyFL 变体，额外暴露特征和两个投影头。"""

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        feature_dim: int,
        dataset_name: str,
    ):
        super().__init__(input_channels, num_classes, feature_dim, dataset_name)
        self.feat_proj = nn.Linear(feature_dim, feature_dim)
        self.proxy_proj = nn.Linear(feature_dim, feature_dim, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature = self.extractor(x)
        return feature, self.classifier(feature)


def build_proxyfl_model(args, num_class: int) -> ProxyFLCNN:
    """在本文件构建 ProxyFL 专用 CNN，避免污染通用模型工厂。"""
    if args.model != "cnn":
        raise ValueError("ProxyFL 当前仅支持 model='cnn'。")
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
    return ProxyFLCNN(channels, num_class, args.feature_dim, args.dataset)


class Client(BaseClientExecutor):
    """SAGE 风格单 DataLoader 的 ProxyFL 客户端。"""

    def __init__(self, args, device, num_class, **kwargs):
        super().__init__(args, device, num_class, **kwargs)
        self.model = build_proxyfl_model(args, num_class).to(device)
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
        self.model_g = build_proxyfl_model(args, num_class).to(device).eval()
        for parameter in self.model_g.parameters():
            parameter.requires_grad_(False)

        self.conf = args.conf
        self.lambda_u = args.lambda_u
        self.temperature = args.temperature

    def train(self):
        self.model.train()
        # 两个模型位于同一 GPU；直接同步避免 GPU→CPU→GPU 的完整状态往返。
        self.model_g.load_state_dict(self.model.state_dict())
        self.model_g.eval()
        self._class_counts = torch.zeros(self.num_class, dtype=torch.float32)
        total = 0.0
        batches = 0
        for epoch in range(self.epochs):
            total, batches = self.run_epoch(
                total,
                batches,
                epoch == self.epochs - 1,
            )
        return ClientResult(
            self.client_id,
            total / max(1, batches),
            clone_state(self.model.state_dict()),
            {"class_counts": self._class_counts.clone()},
        )

    def run_epoch(
        self,
        total: float,
        batches: int,
        collect_counts: bool,
    ):
        for (x_l, y_l), (x_u, *_) in self.get_ssl_loaders():
            x_l, y_l = x_l.to(self.device), y_l.to(self.device)
            x_u = x_u.to(self.device)
            x_l_w = weak_augment(x_l, self.dataset)
            x_u_w = weak_augment(x_u, self.dataset)
            x_u_s = strong_augment(x_u, self.dataset)

            feats_l, logits_l = self.model(x_l_w)
            feats_u_w, logits_u_w = self.model(x_u_w)
            feats_u_s, logits_u_s = self.model(x_u_s)
            loss = F.cross_entropy(logits_l, y_l)
            with torch.no_grad():
                _, logits_l_global = self.model_g(x_l_w)
                _, logits_u_global = self.model_g(x_u_w)
                probs_l_global = torch.softmax(
                    logits_l_global / self.temperature, dim=1
                )
                probs_u_global = torch.softmax(
                    logits_u_global / self.temperature, dim=1
                )
                probs_u_local = torch.softmax(
                    logits_u_w.detach() / self.temperature, dim=1
                )
                conf_l_global = probs_l_global.max(dim=1).values.ge(self.conf)
                conf_global, targets_u_global = probs_u_global.max(dim=1)
                valid_mask = probs_u_local.max(dim=1).values.ge(
                    self.conf
                ) | conf_global.ge(self.conf)

            loss_u = F.cross_entropy(logits_u_s, targets_u_global, reduction="none")
            loss = loss + self.lambda_u * (loss_u * valid_mask).mean()

            # 官方 ICPL 将 L weak、U weak 与 U strong 共同组成对比集合。
            projections = self.model.feat_proj(
                torch.cat((feats_l, feats_u_w, feats_u_s))
            )
            probabilities = torch.cat((probs_l_global, probs_u_global, probs_u_global))
            confidence = torch.cat((conf_l_global, valid_mask, valid_mask))
            loss = loss + self.lambda_u * self._icpl(
                projections,
                probabilities,
                confidence,
                conf_x_num=feats_l.size(0),
            )

            self.check_nan(loss)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if collect_counts:
                self._class_counts += torch.bincount(
                    y_l.detach().cpu(), minlength=self.num_class
                ).to(torch.float32)
                if valid_mask.any():
                    self._class_counts += torch.bincount(
                        targets_u_global[valid_mask].detach().cpu(),
                        minlength=self.num_class,
                    ).to(torch.float32)
            total += loss.item()
            batches += 1
        return total, batches

    def _icpl(
        self,
        features: torch.Tensor,
        probabilities: torch.Tensor,
        confidence: torch.Tensor,
        conf_x_num: int,
    ) -> torch.Tensor:
        """按 scratch/ProxyFL 的候选类别与代理相似度定义计算 ICPL。"""
        feature = F.normalize(features, p=2, dim=1)
        proxy = F.normalize(
            self.model.proxy_proj(self.model.classifier.weight), p=2, dim=1
        )
        payload = cast(dict[str, torch.Tensor], self.payload)
        prior = (
            payload["global_class_dist"]
            .to(self.device, dtype=probabilities.dtype)
            .unsqueeze(0)
        )

        sets_class = probabilities > prior
        top1 = F.one_hot(probabilities.argmax(dim=1), self.num_class).bool()
        predicted_set = torch.where(confidence.unsqueeze(1), top1, sets_class)

        candidate_weight = probabilities * sets_class
        candidate_proxy = candidate_weight @ proxy
        candidate_sim = (feature * candidate_proxy).sum(dim=1)
        proxy_sim = (feature * proxy[probabilities.argmax(dim=1)]).sum(dim=1)
        positive = torch.where(confidence, proxy_sim, candidate_sim)

        overlap = (predicted_set.unsqueeze(1) & predicted_set.unsqueeze(0)).any(dim=2)
        pairwise = feature @ feature.T
        negative = pairwise.masked_fill(overlap | pairwise.lt(1e-6), float("-inf"))
        logits = torch.cat((positive.unsqueeze(1), negative), dim=1)[conf_x_num:]
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=self.device)
        return F.cross_entropy(logits, labels)

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
                inputs = x.to(self.device)
                if self.is_ssl:
                    inputs = prepare_input_batch(inputs, self.dataset)
                _, logits = self.model(inputs)
                correct += logits.argmax(dim=1).eq(y.to(self.device)).sum().item()
                total += y.size(0)
        return EvalResult(task.eval_id, correct, total)


class Server(BaseServer):
    client_cls = Client
    supports_ssl = True

    def __init__(self, args, devices):
        if args.ssl in ("none", "client"):
            raise ValueError(
                "ProxyFL 仅支持 sample、double、sfd 等每客户端含标注样本的 SSL 模式。"
            )
        super().__init__(args, devices)
        # 当前 BaseServer 由调用方管理完整配置；ProxyFL 的 Server 端日志仍需它。
        self.args = args
        self.model = build_proxyfl_model(args, self.num_class)
        self.gpt = nn.Linear(args.feature_dim, self.num_class).to(self.device)
        self.gpt_optimizer = torch.optim.SGD(self.gpt.parameters(), lr=args.server_lr)
        self.server_epochs = args.server_epochs
        self.server_batch_size = args.server_batch_size
        self.gpt_threshold = args.gpt_threshold
        self.ema_beta = args.ema_beta
        self.global_class_dist = torch.full(
            (self.num_class,), 1.0 / self.num_class, dtype=torch.float32
        )

    def train_payloads(self):
        return {
            client_id: {
                "global_class_dist": self.global_class_dist.clone(),
                "unlabeled_ratio": self.args.unlabeled_ratio,
            }
            for client_id in self.selected
        }

    def _update_global_distribution(self, results):
        counts = torch.stack(
            [results[client_id].payload["class_counts"] for client_id in self.selected]
        ).sum(dim=0)
        if counts.sum() > 0:
            current = counts / counts.sum()
            self.global_class_dist.mul_(self.ema_beta).add_(
                current, alpha=1.0 - self.ema_beta
            )

    def _fuse_proxies(self, results):
        """FedAvg 后以客户端分类器行向量训练官方 GPT 代理层并回写分类头。"""
        state = clone_state(self.model.state_dict())
        self.gpt.load_state_dict(
            {
                "weight": state["classifier.weight"],
                "bias": state["classifier.bias"],
            }
        )
        proxies = torch.cat(
            [
                results[client_id].state["classifier.weight"]
                for client_id in self.selected
            ],
            dim=0,
        )
        labels = torch.arange(self.num_class).repeat(len(self.selected))
        loader = DataLoader(
            TensorDataset(proxies, labels),
            batch_size=self.server_batch_size,
            shuffle=True,
        )
        self.gpt.train()
        for _ in range(self.server_epochs):
            for proxy, target in loader:
                proxy = proxy.to(self.device)
                target = target.to(self.device)
                with torch.no_grad():
                    distances = torch.cdist(self.gpt.weight, self.gpt.weight)
                    distances.fill_diagonal_(float("inf"))
                    margin = (
                        distances.min(dim=1).values.max().clamp(max=self.gpt_threshold)
                    )
                distance = torch.cdist(proxy, self.gpt.weight)
                one_hot = F.one_hot(target, self.num_class).to(distance.dtype)
                loss = F.cross_entropy(-distance + one_hot * margin, target)
                self.gpt_optimizer.zero_grad()
                loss.backward()
                self.gpt_optimizer.step()
        self.gpt.eval()
        state["classifier.weight"] = self.gpt.weight.detach().cpu().clone()
        state["classifier.bias"] = self.gpt.bias.detach().cpu().clone()
        self.model.load_state_dict(state)

    def apply_result(self, results):
        self.aggregate_model(results)
        self._fuse_proxies(results)

    def run_round(self):
        results = self.train_clients()
        self._update_global_distribution(results)
        self.apply_result(results)
        loss = sum(results[client_id].loss for client_id in self.selected) / len(
            self.selected
        )
        self.record_round(loss, self.evaluate())
