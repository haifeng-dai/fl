import math
import os
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from .utils import (
    BaseParams,
    BaseServer,
    check_losses,
    clone_cpu_state,
    fmt_num,
    get_model,
)
from .utils.augment import strong_augment, weak_augment
from .utils.loss import masked_kl_loss
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


def train(p: Params):
    device = torch.device(p.client_gpu)
    model_l = get_model(p.model_name, p.dataset, p.num_class, p.feature_dim).to(device)
    model_l.load_state_dict(p.model_state)
    model_g = get_model(p.model_name, p.dataset, p.num_class, p.feature_dim).to(device)
    model_g.load_state_dict(p.model_state)
    model_g.eval()
    for parameter in model_g.parameters():
        parameter.requires_grad = False

    loaders = build_fixmatch_loaders(p.train_set, p.batch_size, p.unlabeled_ratio)
    optimizer = torch.optim.SGD(
        model_l.parameters(),
        lr=p.lr,
        momentum=p.momentum,
        weight_decay=p.weight_decay,
    )
    model_l.train()
    loss_sum = 0.0
    pseudo_count = 0
    pseudo_correct = 0
    steps = 0
    for _ in range(p.epochs):
        for labeled, (x_u, y_u) in iterate_ssl_batches(loaders):
            # SAGE 仅支持 sample/double/sfd 混合模式，每客户端必有有标签样本
            assert labeled is not None
            x_l, y_l = labeled

            # ── 数据增强：有标签弱增强，无标签弱/强增强 ──
            x_l = weak_augment(x_l.to(device), p.dataset)
            y_l = y_l.to(device)
            x_u = x_u.to(device)
            y_u = y_u.to(device)
            x_u_w = weak_augment(x_u, p.dataset)
            x_u_s = strong_augment(x_u, p.dataset)

            # ── 伪标签生成：完全在 no_grad 下执行（避免构建梯度图与中间视图悬空） ──
            with torch.no_grad():
                logits_g = model_g(x_u_w)
                p_g = torch.softmax(logits_g / p.temperature, dim=1)
                confidence_g, targets_g = p_g.max(dim=1)

                logits_u_w = model_l(x_u_w)
                p_l = torch.softmax(logits_u_w / p.temperature, dim=1)
                confidence_l, targets_l = p_l.max(dim=1)

                mask_l = confidence_l.ge(p.confidence).float()
                mask_g = confidence_g.ge(p.confidence).float()
                delta_C = (confidence_l - confidence_g).abs()
                correction = torch.exp(-SAGE_KAPPA * delta_C)
                delta_l = F.one_hot(targets_l, p.num_class).float()
                delta_g = F.one_hot(targets_g, p.num_class).float()
                targets = torch.where(
                    mask_l.unsqueeze(1).bool(),
                    correction.unsqueeze(1) * delta_l
                    + (1.0 - correction).unsqueeze(1) * delta_g,
                    delta_g,
                )
                valid_mask = torch.maximum(mask_l, mask_g)
                cur_valid_count = int(valid_mask.sum().item())
                pseudo_count += cur_valid_count

                b_valid = valid_mask.bool()
                final_targets = targets.argmax(dim=-1)
                pseudo_correct += int(((final_targets == y_u) & b_valid).sum().item())

            # ── 带梯度的前向计算：仅对有标签样本和无标签强增强样本求梯度 ──
            inputs = torch.cat((x_l, x_u_s))
            logits = model_l(inputs)
            batch_size = x_l.size(0)
            logits_l = logits[:batch_size]
            logits_u_s = logits[batch_size:]

            # ── 总损失：有监督 + λu × 无监督 ──
            supervised_loss = F.cross_entropy(logits_l, y_l)
            unsupervised_loss = None
            if cur_valid_count > 0:
                unsupervised_loss = masked_kl_loss(logits_u_s, targets, valid_mask)
                loss = supervised_loss + p.lambda_u * unsupervised_loss
            else:
                loss = supervised_loss

            optimizer.zero_grad()
            check_losses(loss, locals())
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            steps += 1

    final_state = clone_cpu_state(model_l.state_dict())
    del model_l, model_g, optimizer, loaders
    torch.cuda.empty_cache()

    return {
        "state": final_state,
        "loss": loss_sum / max(1, steps),
        "pseudo_count": pseudo_count,
        "pseudo_correct": pseudo_correct,
    }


class Server(BaseServer):
    def __init__(self, args):
        if args.ssl in ("none", "client"):
            raise ValueError(
                "SAGE 仅支持 sample、double、sfd 半监督模式"
                "（client 模式存在纯无标签客户端，SAGE 不支持）。"
            )
        super().__init__(args, is_ssl=True)
        self.unlabeled_ratio = args.unlabeled_ratio
        self.temperature = args.temperature
        self.pseudo_acc = []
        self.pseudo_count = []

        for dataset in self.train_sets.values():
            if dataset.is_labeled is None:
                raise ValueError("SAGE 需要半监督数据中的 is_labeled 字段")

    def fit(self):
        num_join = max(1, int(self.num_clients * self.join_ratio))
        for round_index in range(self.rounds):
            started = time.time()
            selected = sorted(torch.randperm(self.num_clients)[:num_join].tolist())
            print(f"\n--- SAGE Round {round_index + 1}/{self.rounds} ---")
            print(f"Selected clients: {selected}")
            parameters = [
                Params(
                    **asdict(base),
                    unlabeled_ratio=self.unlabeled_ratio,
                    lambda_u=self.lam,
                    confidence=self.confidence,
                    temperature=self.temperature,
                )
                for base in self.build_base_params(selected)
            ]
            results = self.run_clients(train, parameters)

            selected_states = []
            current_weights = []
            total_loss = 0.0
            total_pseudo_count = 0
            total_pseudo_correct = 0
            for cid, res in results.items():
                selected_states.append(res["state"])
                current_weights.append(self.weights[cid])
                total_loss += res["loss"]
                total_pseudo_count += res["pseudo_count"]
                total_pseudo_correct += res["pseudo_correct"]

            sum_weights = sum(current_weights)
            norm_weights = [w / sum_weights for w in current_weights]
            self.aggregate(selected_states, weights=norm_weights)

            round_pseudo_acc = (
                total_pseudo_correct / max(1, total_pseudo_count)
            ) * 100.0
            self.loss.append(total_loss / num_join)
            self.pseudo_acc.append(round_pseudo_acc)
            self.pseudo_count.append(total_pseudo_count)

            self.evaluate()
            round_duration = time.time() - started
            print(
                f"Global Acc: {self.acc[-1]:.2f}% | "
                f"Loss: {self.loss[-1]:.4f} | "
                f"Pseudo Acc: {self.pseudo_acc[-1]:.2f}% | "
                f"Pseudo Count: {self.pseudo_count[-1]}"
            )
            print(f"Round finished in {round_duration:.2f} seconds")

    def save(self):
        metrics = {
            "acc": self.acc,
            "loss": self.loss,
            "pseudo_acc": self.pseudo_acc,
            "pseudo_count": self.pseudo_count,
        }
        params = {"global": self.model.state_dict()}
        self.deal_save(metrics, params)
