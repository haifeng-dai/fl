import math
import os
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .utils import BaseParams, BaseServer, fmt_num, get_model
from .utils.augment import sage_strong_augment, sage_weak_augment
from .utils.ssl import build_fixmatch_loaders, iterate_ssl_batches

SAGE_KAPPA = math.log(2.0) / 0.05


@dataclass
class Params(BaseParams):
    unlabeled_ratio: int
    lambda_u: float
    confidence: float
    temperature: float


def get_path(args):
    args.file_name = f"{args.common_name}_{fmt_num(args.temperature)}"
    return os.path.join(args.log_path, f"{args.file_name}_{args.cur_time}.log")


def _sage_loss(student, teacher, labeled, unlabeled, p, device):
    unlabeled_x, _ = unlabeled
    unlabeled_weak = sage_weak_augment(unlabeled_x, p.dataset).to(device)
    unlabeled_strong = sage_strong_augment(unlabeled_x, p.dataset).to(device)

    if labeled is None:
        inputs = torch.cat((unlabeled_weak, unlabeled_strong))
    else:
        labeled_x, labeled_y = labeled
        labeled_x = sage_weak_augment(labeled_x, p.dataset).to(device)
        labeled_y = labeled_y.to(device)
        inputs = torch.cat((labeled_x, unlabeled_weak, unlabeled_strong))
    logits = student(inputs)
    if labeled is None:
        logits_u_w, logits_u_s = logits.chunk(2)
        supervised_loss = logits_u_s.sum() * 0.0
    else:
        batch_size = labeled[0].size(0)
        logits_x = logits[:batch_size]
        logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
        supervised_loss = F.cross_entropy(logits_x, labeled[1].to(device))
    with torch.no_grad():
        global_logits = teacher(unlabeled_weak)
        global_probs = torch.softmax(global_logits / p.temperature, dim=1)
        global_confidence, global_targets = global_probs.max(dim=1)

    local_probs = torch.softmax(logits_u_w.detach() / p.temperature, dim=1)
    local_confidence, local_targets = local_probs.max(dim=1)
    local_mask = local_confidence.ge(p.confidence).float()
    global_mask = global_confidence.ge(p.confidence).float()
    delta = (local_confidence - global_confidence).abs().clamp(1e-6, 1.0)
    correction = torch.exp(-SAGE_KAPPA * delta).clamp(1e-6, 1.0)
    local_one_hot = F.one_hot(local_targets, p.num_class).float()
    global_one_hot = F.one_hot(global_targets, p.num_class).float()
    targets = torch.where(
        local_mask.unsqueeze(1).bool(),
        correction.unsqueeze(1) * local_one_hot
        + (1.0 - correction).unsqueeze(1) * global_one_hot,
        global_one_hot,
    )
    valid_mask = torch.maximum(local_mask, global_mask)
    unsupervised_losses = (
        F.kl_div(F.log_softmax(logits_u_s, dim=1), targets, reduction="none").sum(dim=1)
        * valid_mask
    )
    unsupervised_loss = unsupervised_losses.mean()
    loss = supervised_loss + p.lambda_u * unsupervised_loss
    diagnostics = {
        "pseudo_selected": int(valid_mask.sum().item()),
        "pseudo_total": int(valid_mask.numel()),
        "pseudo_confidence_sum": (
            torch.maximum(local_confidence, global_confidence) * valid_mask
        )
        .sum()
        .item(),
    }
    return loss, supervised_loss, unsupervised_loss, diagnostics


def train(p: Params):
    device = torch.device(p.client_gpu)
    student = get_model(p.model_name, p.dataset, p.num_class, p.feature_dim).to(device)
    student.load_state_dict(p.model_state)
    teacher = get_model(p.model_name, p.dataset, p.num_class, p.feature_dim).to(device)
    teacher.load_state_dict(p.model_state)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False

    loaders = build_fixmatch_loaders(p.train_set, p.batch_size, p.unlabeled_ratio)
    optimizer = torch.optim.SGD(
        student.parameters(),
        lr=p.lr,
        momentum=p.momentum,
        weight_decay=p.weight_decay,
    )
    student.train()
    loss_sum = 0.0
    loss_x_sum = 0.0
    loss_u_sum = 0.0
    loss_x_count = 0
    loss_u_count = 0
    pseudo_selected = 0
    pseudo_total = 0
    pseudo_confidence_sum = 0.0
    steps = 0
    started = time.time()
    for _ in range(p.epochs):
        for labeled, unlabeled in iterate_ssl_batches(loaders):
            loss, supervised, unsupervised, diagnostics = _sage_loss(
                student, teacher, labeled, unlabeled, p, device
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            batch_x = 0 if labeled is None else labeled[0].size(0)
            batch_u = unlabeled[0].size(0)
            loss_x_sum += supervised.item() * batch_x
            loss_u_sum += unsupervised.item() * batch_u
            loss_x_count += batch_x
            loss_u_count += batch_u
            pseudo_selected += diagnostics["pseudo_selected"]
            pseudo_total += diagnostics["pseudo_total"]
            pseudo_confidence_sum += diagnostics["pseudo_confidence_sum"]
            steps += 1

    divisor = max(1, steps)
    return {
        "state": {
            key: value.cpu().detach().clone()
            for key, value in student.state_dict().items()
        },
        "loss": loss_sum / divisor,
        "loss_x_sum": loss_x_sum,
        "loss_x_count": loss_x_count,
        "loss_u_sum": loss_u_sum,
        "loss_u_count": loss_u_count,
        "pseudo_selected": pseudo_selected,
        "pseudo_total": pseudo_total,
        "pseudo_confidence_sum": pseudo_confidence_sum,
        "time_total": time.time() - started,
    }


class Server(BaseServer):
    def __init__(self, args):
        if args.ssl == "none":
            raise ValueError(
                "SAGE 仅支持半监督数据，请设置 ssl 为 sample、client、double 或 sfd。"
            )
        super().__init__(args, is_ssl=True)
        self.unlabeled_ratio = args.unlabeled_ratio
        self.lambda_u = args.lam
        self.confidence = args.confidence
        self.temperature = args.temperature
        self.momentum = args.momentum
        self.weight_decay = args.weight_decay
        self.loss_x = []
        self.loss_u = []
        self.pseudo_coverage = []
        self.pseudo_confidence = []
        self.pseudo_selected = []
        self.pseudo_total = []
        self.round_timing = []

        for dataset in self.train_sets.values():
            if dataset.is_labeled is None:
                raise ValueError("SAGE 需要半监督数据中的 is_labeled 字段")

    def fit(self):
        num_join = max(1, int(self.num_clients * self.join_ratio))
        for round_index in range(self.rounds):
            started = time.time()
            selected = sorted(torch.randperm(self.num_clients)[:num_join].tolist())
            print(
                f"\n--- SAGE Round {round_index + 1}/{self.rounds} ---"
                f" selected_clients={selected}"
            )
            parameters = [
                Params(
                    **asdict(base),
                    unlabeled_ratio=self.unlabeled_ratio,
                    lambda_u=self.lambda_u,
                    confidence=self.confidence,
                    temperature=self.temperature,
                )
                for base in self.build_base_params(selected)
            ]
            results = self.run_clients(train, parameters)
            states = [results[client_id]["state"] for client_id in selected]
            weights = [self.weights[client_id] for client_id in selected]
            self.aggregate(states, weights=[w / sum(weights) for w in weights])

            x_count = sum(results[i]["loss_x_count"] for i in selected)
            u_count = sum(results[i]["loss_u_count"] for i in selected)
            self.loss_x.append(
                sum(results[i]["loss_x_sum"] for i in selected) / x_count
                if x_count > 0
                else 0.0
            )
            self.loss_u.append(
                sum(results[i]["loss_u_sum"] for i in selected) / u_count
            )
            self.loss.append(self._evaluate_loss())
            pseudo_selected = sum(results[i]["pseudo_selected"] for i in selected)
            pseudo_total = sum(results[i]["pseudo_total"] for i in selected)
            self.pseudo_selected.append(pseudo_selected)
            self.pseudo_total.append(pseudo_total)
            self.pseudo_coverage.append(100.0 * pseudo_selected / max(1, pseudo_total))
            self.pseudo_confidence.append(
                sum(results[i]["pseudo_confidence_sum"] for i in selected)
                / max(1, pseudo_selected)
            )
            self.evaluate()
            self.round_timing.append(time.time() - started)
            print(
                f"Accuracy: {self.acc[-1]:.2f}% | Loss: {self.loss[-1]:.4f} | "
                f"Pseudo coverage: {self.pseudo_coverage[-1]:.2f}%"
            )

    def _evaluate_loss(self):
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

    def save(self):
        metrics = {
            "acc": self.acc,
            "loss": self.loss,
            "loss_x": self.loss_x,
            "loss_u": self.loss_u,
            "pseudo_coverage": self.pseudo_coverage,
            "pseudo_confidence": self.pseudo_confidence,
            "pseudo_selected": self.pseudo_selected,
            "pseudo_total": self.pseudo_total,
            "round_time": self.round_timing,
        }
        params = {"global": self.model.state_dict()}
        self.deal_save(metrics, params)
