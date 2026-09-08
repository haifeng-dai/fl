import math
import os
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from .utils import (
    BaseParams,
    BaseServer,
    clone_cpu_state,
    fmt_num,
    get_model,
    masked_kl_loss,
)
from .utils.augment import strong_augment, weak_augment
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
    loss_x_sum = 0.0
    loss_u_sum = 0.0
    loss_x_count = 0
    loss_u_count = 0
    pseudo_selected = 0
    pseudo_total = 0
    pseudo_confidence_sum = 0.0
    g_selected = 0
    l_selected = 0
    global_correct = 0
    local_correct = 0
    g_selected_correct = 0
    l_selected_correct = 0
    final_selected_correct = 0
    conflict_g_wins = 0
    conflict_l_wins = 0
    steps = 0
    for _ in range(p.epochs):
        for labeled, (x_u, y_u) in iterate_ssl_batches(loaders):
            # SAGE 仅支持 sample/double/sfd 混合模式，每客户端必有有标签样本
            assert labeled is not None
            x_l, y_l = labeled

            # ── 数据增强：有标签弱增强，无标签弱/强增强 ──
            x_l = x_l.to(device)
            x_u = x_u.to(device)
            x_l = weak_augment(x_l, p.dataset)
            y_l = y_l.to(device)
            y_u = y_u.to(device)
            x_u_w = weak_augment(x_u, p.dataset)
            x_u_s = strong_augment(x_u, p.dataset)

            # ── 单次前向：拼接 [有标签 | 无标签弱 | 无标签强] ──
            inputs = torch.cat((x_l, x_u_w, x_u_s))
            logits = model_l(inputs)
            batch_size = x_l.size(0)
            logits_l = logits[:batch_size]
            logits_u_w, logits_u_s = logits[batch_size:].chunk(2)

            # ── 有监督损失：有标签数据的交叉熵 ──
            supervised_loss = F.cross_entropy(logits_l, y_l)

            # ── 伪标签生成：本地与全局（教师）置信度校正融合 ──
            with torch.no_grad():
                logits_g = model_g(x_u_w)
                p_g = torch.softmax(logits_g / p.temperature, dim=1)
                confidence_g, targets_g = p_g.max(dim=1)
            p_l = torch.softmax(logits_u_w.detach() / p.temperature, dim=1)
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

            # ── 无监督损失：强增强预测与伪标签目标的 KL 散度 ──
            unsupervised_loss = masked_kl_loss(logits_u_s, targets, valid_mask)
            cur_valid = int(valid_mask.sum().item())
            pseudo_selected += cur_valid
            pseudo_total += int(valid_mask.numel())
            pseudo_confidence_sum += (
                (torch.maximum(confidence_l, confidence_g) * valid_mask).sum().item()
            )

            # ── 伪标签质量与命中率监控 ──
            b_mask_g = mask_g.bool()
            b_mask_l = mask_l.bool()
            b_valid = valid_mask.bool()
            g_selected += int(b_mask_g.sum().item())
            l_selected += int(b_mask_l.sum().item())

            g_match = targets_g == y_u
            l_match = targets_l == y_u
            global_correct += int(g_match.sum().item())
            local_correct += int(l_match.sum().item())

            g_selected_correct += int((g_match & b_mask_g).sum().item())
            l_selected_correct += int((l_match & b_mask_l).sum().item())

            final_targets = targets.argmax(dim=-1)
            final_selected_correct += int(
                ((final_targets == y_u) & b_valid).sum().item()
            )

            # 冲突判定：两者均通过阈值但预测类别不同，谁判断更正确
            conflict_mask = b_mask_g & b_mask_l & (targets_g != targets_l)
            conflict_g_wins += int((g_match & conflict_mask).sum().item())
            conflict_l_wins += int((l_match & conflict_mask).sum().item())

            # ── 总损失：有监督 + λu × 无监督 ──
            loss = supervised_loss + p.lambda_u * unsupervised_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            batch_l = x_l.size(0)
            batch_u = x_u_w.size(0)
            loss_x_sum += supervised_loss.item() * batch_l
            loss_u_sum += unsupervised_loss.item() * batch_u
            loss_x_count += batch_l
            loss_u_count += batch_u
            steps += 1

    return {
        "state": clone_cpu_state(model_l.state_dict()),
        "loss": loss_sum / max(1, steps),
        "loss_x_sum": loss_x_sum,
        "loss_x_count": loss_x_count,
        "loss_u_sum": loss_u_sum,
        "loss_u_count": loss_u_count,
        "pseudo_selected": pseudo_selected,
        "pseudo_total": pseudo_total,
        "pseudo_confidence_sum": pseudo_confidence_sum,
        "g_selected": g_selected,
        "l_selected": l_selected,
        "global_correct": global_correct,
        "local_correct": local_correct,
        "g_selected_correct": g_selected_correct,
        "l_selected_correct": l_selected_correct,
        "final_selected_correct": final_selected_correct,
        "conflict_g_wins": conflict_g_wins,
        "conflict_l_wins": conflict_l_wins,
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
        self.loss_x = []
        self.loss_u = []
        self.pseudo_coverage = []
        self.pseudo_confidence = []
        self.pseudo_selected = []
        self.pseudo_total = []
        self.u_usage_ratio = []
        self.g_select_ratio = []
        self.l_select_ratio = []
        self.global_pseudo_acc = []
        self.local_pseudo_acc = []
        self.g_selected_acc = []
        self.l_selected_acc = []
        self.final_pseudo_acc = []
        self.round_timing = []

        for dataset in self.train_sets.values():
            if dataset.is_labeled is None:
                raise ValueError("SAGE 需要半监督数据中的 is_labeled 字段")

    def fit(self):
        num_join = max(1, int(self.num_clients * self.join_ratio))
        for round_index in range(self.rounds):
            started = time.time()
            selected = sorted(torch.randperm(self.num_clients)[:num_join].tolist())
            print(f"\n--- SAGE Round {round_index + 1}/{self.rounds} ---")
            print(f" selected_clients={selected}")
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

            # 汇集各客户端的回传结果，计算聚合权重与损失/伪标签统计
            selected_states = []
            current_weights = []
            total_loss = 0.0
            x_loss_sum = 0.0
            u_loss_sum = 0.0
            x_count = 0
            u_count = 0
            pseudo_selected = 0
            pseudo_total = 0
            pseudo_confidence_sum = 0.0
            g_selected = 0
            l_selected = 0
            global_correct = 0
            local_correct = 0
            g_selected_correct = 0
            l_selected_correct = 0
            final_selected_correct = 0
            conflict_g_wins = 0
            conflict_l_wins = 0
            for cid, res in results.items():
                selected_states.append(res["state"])
                current_weights.append(self.weights[cid])
                total_loss += res["loss"]
                x_loss_sum += res["loss_x_sum"]
                u_loss_sum += res["loss_u_sum"]
                x_count += res["loss_x_count"]
                u_count += res["loss_u_count"]
                pseudo_selected += res["pseudo_selected"]
                pseudo_total += res["pseudo_total"]
                pseudo_confidence_sum += res["pseudo_confidence_sum"]
                g_selected += res["g_selected"]
                l_selected += res["l_selected"]
                global_correct += res["global_correct"]
                local_correct += res["local_correct"]
                g_selected_correct += res["g_selected_correct"]
                l_selected_correct += res["l_selected_correct"]
                final_selected_correct += res["final_selected_correct"]
                conflict_g_wins += res["conflict_g_wins"]
                conflict_l_wins += res["conflict_l_wins"]
            sum_weights = sum(current_weights)
            norm_weights = [w / sum_weights for w in current_weights]
            self.aggregate(selected_states, weights=norm_weights)

            tot_u_safe = max(1, pseudo_total)
            u_usage_ratio = pseudo_selected / tot_u_safe
            g_select_ratio = g_selected / tot_u_safe
            l_select_ratio = l_selected / tot_u_safe
            global_pseudo_acc = global_correct / tot_u_safe
            local_pseudo_acc = local_correct / tot_u_safe
            g_selected_acc = g_selected_correct / max(1, g_selected)
            l_selected_acc = l_selected_correct / max(1, l_selected)
            final_pseudo_acc = final_selected_correct / max(1, pseudo_selected)

            self.loss_x.append(x_loss_sum / x_count)
            self.loss_u.append(u_loss_sum / u_count)
            self.loss.append(total_loss / num_join)
            self.pseudo_selected.append(pseudo_selected)
            self.pseudo_total.append(pseudo_total)
            self.pseudo_coverage.append(100.0 * u_usage_ratio)
            self.pseudo_confidence.append(
                pseudo_confidence_sum / max(1, pseudo_selected)
            )
            self.u_usage_ratio.append(u_usage_ratio)
            self.g_select_ratio.append(g_select_ratio)
            self.l_select_ratio.append(l_select_ratio)
            self.global_pseudo_acc.append(global_pseudo_acc)
            self.local_pseudo_acc.append(local_pseudo_acc)
            self.g_selected_acc.append(g_selected_acc)
            self.l_selected_acc.append(l_selected_acc)
            self.final_pseudo_acc.append(final_pseudo_acc)

            self.evaluate()
            self.round_timing.append(time.time() - started)
            print(
                f"Accuracy: {self.acc[-1]:.2f}% | Loss: {self.loss[-1]:.4f} | "
                f"U-Usage: {u_usage_ratio * 100:.2f}% (G_sel={g_select_ratio * 100:.2f}%, L_sel={l_select_ratio * 100:.2f}%)"
            )
            print(
                f"Selected Acc: Final={final_pseudo_acc * 100:.2f}%, G_sel_acc={g_selected_acc * 100:.2f}%, L_sel_acc={l_selected_acc * 100:.2f}% | "
                f"Conflict (G/L wins): {conflict_g_wins}/{conflict_l_wins}"
            )
            print(
                f"Direct Pseudo Acc: Global={global_pseudo_acc * 100:.2f}%, Local={local_pseudo_acc * 100:.2f}%"
            )
            print(f"Round finished in {time.time() - started:.2f} seconds")

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
            "u_usage_ratio": self.u_usage_ratio,
            "g_select_ratio": self.g_select_ratio,
            "l_select_ratio": self.l_select_ratio,
            "global_pseudo_acc": self.global_pseudo_acc,
            "local_pseudo_acc": self.local_pseudo_acc,
            "g_selected_acc": self.g_selected_acc,
            "l_selected_acc": self.l_selected_acc,
            "final_pseudo_acc": self.final_pseudo_acc,
            "round_time": self.round_timing,
        }
        params = {"global": self.model.state_dict()}
        self.deal_save(metrics, params)
