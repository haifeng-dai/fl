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
from .utils.ssl import build_fixmatch_loaders, iterate_fixmatch_batches


@dataclass
class Params(BaseParams):
    unlabeled_ratio: int
    lambda_u: float
    conf: float
    kappa: torch.Tensor = torch.log(torch.tensor(2.0)) / 0.05


def get_path(args):
    args.file_name = f"{args.common_name}_{fmt_num(args.lambda_u)}_{fmt_num(args.conf)}"
    return os.path.join(args.log_path, f"{args.file_name}_{args.cur_time}.log")


def train(p: Params):
    device = torch.device(p.client_gpu)
    model_l = get_model(p).to(device)
    model_l.load_state_dict(p.model_state)
    model_g = get_model(p).to(device)
    model_g.load_state_dict(p.model_state)
    model_g.eval()
    for para in model_g.parameters():
        para.requires_grad = False

    loaders = build_fixmatch_loaders(p.train_set, p.batch_size, p.unlabeled_ratio)
    optimizer = torch.optim.SGD(
        model_l.parameters(),
        lr=p.lr,
        momentum=p.momentum,
        weight_decay=p.weight_decay,
    )
    model_l.train()
    loss_sum = 0.0
    supervised_count = 0
    pseudo_count = 0
    pseudo_correct = 0
    steps = 0
    for _ in range(p.epochs):
        for (x_l, y_l), (x_u, y_u) in iterate_fixmatch_batches(loaders):
            # ── 数据增强：有标签弱增强，无标签弱/强增强 ──
            supervised_count += y_l.numel()
            x_l = weak_augment(x_l.to(device), p.dataset)
            x_u, y_u, y_l = x_u.to(device), y_u.to(device), y_l.to(device)
            x_u_w = weak_augment(x_u, p.dataset)
            x_u_s = strong_augment(x_u, p.dataset)

            # ── 伪标签生成：完全在 no_grad 下执行 ──
            with torch.no_grad():
                logits_g = model_g(x_u_w)
                p_g = torch.softmax(logits_g, dim=1)
                confidence_g, targets_g = p_g.max(dim=1)

                logits_u_w = model_l(x_u_w)
                p_l = torch.softmax(logits_u_w, dim=1)
                confidence_l, targets_l = p_l.max(dim=1)

                mask_l = confidence_l.ge(p.conf).float()
                mask_g = confidence_g.ge(p.conf).float()
                delta_C = (confidence_l - confidence_g).abs()  # Eq. (2)
                correction = torch.exp(-p.kappa * delta_C)  # Eq. (3)
                delta_l = F.one_hot(targets_l, p.num_class).float()  # Eq. (4)
                delta_g = F.one_hot(targets_g, p.num_class).float()  # Eq. (5)
                w = correction.unsqueeze(1)
                targets = w * delta_l + (1.0 - w) * delta_g  # Eqs. (6) & (7-1)
                g_only = ~mask_l.bool() & mask_g.bool()
                targets[g_only] = delta_g[g_only]  # Eq. (7-2)

                valid_mask = torch.maximum(mask_l, mask_g)  # 选伪标签样本
                cur_valid_count = int(valid_mask.sum().item())
                pseudo_count += cur_valid_count
                b_valid = valid_mask.bool()
                final_targets = targets.argmax(dim=-1)
                pseudo_correct += int(((final_targets == y_u) & b_valid).sum().item())

            # ── 带梯度的前向计算：仅对有标签样本和无标签强增强样本求梯度 ──
            inputs = torch.cat((x_l, x_u_s))
            logits = model_l(inputs)
            batch_size = x_l.size(0)
            logits_l, logits_u_s = logits[:batch_size], logits[batch_size:]

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

    return {
        "state": clone_cpu_state(model_l.state_dict()),
        "loss": loss_sum / steps,
        "pseudo_count": pseudo_count,
        "pseudo_correct": pseudo_correct,
        "aggregation_count": supervised_count + pseudo_count,
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
        self.lambda_u = args.lambda_u
        self.conf = args.conf
        self.pseudo_acc = []
        self.pseudo_count = []

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
                    lambda_u=self.lambda_u,
                    conf=self.conf,
                )
                for base in self.build_base_params(selected)
            ]
            results = self.run_clients(train, parameters)

            selected_states = []
            aggregation_counts = []
            total_loss = 0.0
            total_pseudo_count = 0
            total_pseudo_correct = 0
            for res in results.values():
                selected_states.append(res["state"])
                aggregation_counts.append(res["aggregation_count"])
                total_loss += res["loss"]
                total_pseudo_count += res["pseudo_count"]
                total_pseudo_correct += res["pseudo_correct"]

            sum_counts = sum(aggregation_counts)
            norm_weights = [count / sum_counts for count in aggregation_counts]
            self.aggregate(selected_states, weights=norm_weights)

            round_pseudo_acc = (
                total_pseudo_correct / max(1, total_pseudo_count)
            ) * 100.0
            self.loss.append(total_loss / num_join)
            self.pseudo_acc.append(round_pseudo_acc)
            self.pseudo_count.append(total_pseudo_count)

            self.evaluate()
            print(
                f"Global Acc: {self.acc[-1]:.2f}% | "
                f"Loss: {self.loss[-1]:.4f} | "
                f"Pseudo Acc: {self.pseudo_acc[-1]:.2f}% | "
                f"Pseudo Count: {self.pseudo_count[-1]}"
            )
            print(f"Round finished in {time.time() - started:.2f} seconds")

    def save(self):
        metrics = {
            "acc": self.acc,
            "loss": self.loss,
            "pseudo_acc": self.pseudo_acc,
            "pseudo_count": self.pseudo_count,
        }
        params = {"global": self.model.state_dict()}
        self.deal_save(metrics, params)
