import os
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .utils import BaseParams, BaseServer, clone_cpu_state, get_model
from .utils.augment import sage_strong_augment, sage_weak_augment
from .utils.ssl import build_fixmatch_loaders, iterate_ssl_batches


def get_path(args):
    args.file_name = args.common_name
    return os.path.join(args.log_path, f"{args.file_name}_{args.cur_time}.log")


@dataclass
class Params(BaseParams):
    unlabeled_ratio: int
    lam: float
    confidence: float


def fixmatch_loss(logits_l, logits_u_w, logits_u_s, y_l, confidence, lam):
    """计算单个拼接 batch 的经典 FixMatch 损失。"""
    loss_x = F.cross_entropy(logits_l, y_l)
    with torch.no_grad():
        probs_u = torch.softmax(logits_u_w.detach(), dim=1)
        max_probs, pseudo_targets = probs_u.max(dim=1)
        mask = max_probs.ge(confidence).float()
    loss_u = (
        F.cross_entropy(logits_u_s, pseudo_targets, reduction="none") * mask
    ).mean()
    return (
        loss_x + lam * loss_u,
        loss_x,
        loss_u,
        {
            "pseudo_selected": int(mask.sum().item()),
            "pseudo_total": int(mask.numel()),
            "pseudo_confidence_sum": (max_probs * mask).sum().item(),
        },
    )


def normalize_weights(weights):
    total = sum(weights)
    return [weight / total for weight in weights]


def aggregate_metrics(results, selected):
    loss_x_count = sum(results[i]["loss_x_count"] for i in selected)
    loss_u_count = sum(results[i]["loss_u_count"] for i in selected)
    pseudo_selected = sum(results[i]["pseudo_selected"] for i in selected)
    pseudo_total = sum(results[i]["pseudo_total"] for i in selected)
    pseudo_confidence_sum = sum(results[i]["pseudo_confidence_sum"] for i in selected)
    return {
        "loss_x": (
            sum(results[i]["loss_x_sum"] for i in selected) / loss_x_count
            if loss_x_count > 0
            else 0.0
        ),
        "loss_u": (
            sum(results[i]["loss_u_sum"] for i in selected) / loss_u_count
            if loss_u_count > 0
            else 0.0
        ),
        "pseudo_selected": pseudo_selected,
        "pseudo_total": pseudo_total,
        "pseudo_coverage": 100.0 * pseudo_selected / max(1, pseudo_total),
        "pseudo_confidence": pseudo_confidence_sum / max(1, pseudo_selected),
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

    total_loss = 0.0
    loss_x_sum = 0.0
    loss_u_sum = 0.0
    loss_x_count = 0
    loss_u_count = 0
    pseudo_selected = 0
    pseudo_total = 0
    pseudo_confidence_sum = 0.0
    num_batches = 0
    model.train()
    for _ in range(p.epochs):
        for labeled_batch, unlabeled_batch in iterate_ssl_batches(loaders):
            if labeled_batch is None:
                raise ValueError("fedavg_lpl 客户端缺少有标签 batch")
            x_l, y_l = labeled_batch
            x_u, _ = unlabeled_batch
            x_l, y_l = x_l.to(device), y_l.to(device)
            x_u = x_u.to(device)
            x_l = sage_weak_augment(x_l, p.dataset)
            x_u_w = sage_weak_augment(x_u, p.dataset)
            x_u_s = sage_strong_augment(x_u, p.dataset)

            labeled_batch_size = x_l.size(0)
            unlabeled_batch_size = x_u.size(0)
            logits = model(torch.cat((x_l, x_u_w, x_u_s)))
            logits_l = logits[:labeled_batch_size]
            logits_u_w = logits[
                labeled_batch_size : labeled_batch_size + unlabeled_batch_size
            ]
            logits_u_s = logits[labeled_batch_size + unlabeled_batch_size :]
            loss, loss_x, loss_u, diagnostics = fixmatch_loss(
                logits_l,
                logits_u_w,
                logits_u_s,
                y_l,
                p.confidence,
                p.lam,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loss_x_sum += loss_x.item() * labeled_batch_size
            loss_u_sum += loss_u.item() * unlabeled_batch_size
            loss_x_count += labeled_batch_size
            loss_u_count += unlabeled_batch_size
            pseudo_selected += diagnostics["pseudo_selected"]
            pseudo_total += diagnostics["pseudo_total"]
            pseudo_confidence_sum += diagnostics["pseudo_confidence_sum"]
            num_batches += 1

    state = clone_cpu_state(model.state_dict())
    return {
        "state": state,
        "loss": total_loss / max(1, num_batches),
        "loss_x_sum": loss_x_sum,
        "loss_x_count": loss_x_count,
        "loss_u_sum": loss_u_sum,
        "loss_u_count": loss_u_count,
        "pseudo_selected": pseudo_selected,
        "pseudo_total": pseudo_total,
        "pseudo_confidence_sum": pseudo_confidence_sum,
        "num_samples": len(p.train_set.y),
    }


class Server(BaseServer):
    def __init__(self, args):
        if args.ssl not in ("sample", "double", "sfd"):
            raise ValueError("fedavg_lpl 要求 ssl 为 sample、double 或 sfd。")
        super().__init__(args, is_ssl=True, pfl=False)
        self.unlabeled_ratio = args.unlabeled_ratio
        self.lambda_u = args.lam
        self.confidence = args.confidence
        self.loss_x = []
        self.loss_u = []
        self.pseudo_selected = []
        self.pseudo_total = []
        self.pseudo_coverage = []
        self.pseudo_confidence = []
        self.round_time = []

    def evaluate_loss(self):
        loader = DataLoader(self.test_set, batch_size=128, shuffle=False)
        self.model.to(self.device)
        self.model.eval()
        loss_sum = 0.0
        sample_count = 0
        with torch.no_grad():
            for data, target, *_ in loader:
                data = data.to(self.device)
                target = target.to(self.device)
                logits = self.model(data)
                loss_sum += F.cross_entropy(logits, target, reduction="sum").item()
                sample_count += target.size(0)
        self.model.cpu()
        return loss_sum / max(1, sample_count)

    def fit(self):
        num_join = max(1, int(self.num_clients * self.join_ratio))
        for round_id in range(self.rounds):
            started = time.time()
            selected = sorted(torch.randperm(self.num_clients)[:num_join].tolist())
            parameters = [
                Params(
                    **asdict(base),
                    unlabeled_ratio=self.unlabeled_ratio,
                    lam=self.lambda_u,
                    confidence=self.confidence,
                )
                for base in self.build_base_params(selected)
            ]
            results = self.run_clients(train, parameters)
            states = [results[client_id]["state"] for client_id in selected]
            weights = [self.weights[client_id] for client_id in selected]
            self.aggregate(states, weights=normalize_weights(weights))

            metrics = aggregate_metrics(results, selected)
            self.loss_x.append(metrics["loss_x"])
            self.loss_u.append(metrics["loss_u"])
            self.pseudo_selected.append(metrics["pseudo_selected"])
            self.pseudo_total.append(metrics["pseudo_total"])
            self.pseudo_coverage.append(metrics["pseudo_coverage"])
            self.pseudo_confidence.append(metrics["pseudo_confidence"])
            self.loss.append(self.evaluate_loss())
            self.evaluate()
            self.round_time.append(time.time() - started)
            print(
                f"\n--- FedAvg-LPL Round {round_id + 1}/{self.rounds} ---"
                f"\nGlobal Accuracy: {self.acc[-1]:.2f}%, "
                f"Avg Loss: {self.loss[-1]:.4f}, "
                f"Pseudo Coverage: {self.pseudo_coverage[-1]:.2f}%"
            )

    def save(self):
        metrics = {
            "acc": self.acc,
            "loss": self.loss,
            "loss_x": self.loss_x,
            "loss_u": self.loss_u,
            "pseudo_selected": self.pseudo_selected,
            "pseudo_total": self.pseudo_total,
            "pseudo_coverage": self.pseudo_coverage,
            "pseudo_confidence": self.pseudo_confidence,
            "round_time": self.round_time,
        }
        params = {"global": self.model.state_dict()}
        self.deal_save(metrics, params)
