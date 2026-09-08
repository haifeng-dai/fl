import os
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .utils import (
    BaseParams,
    BaseServer,
    clone_cpu_state,
    dist_contrastive_loss,
    evaluate_model,
    fmt_num,
    get_model,
    masked_kl_loss,
    param_aggregate,
    prepare_input_batch,
)
from .utils.augment import strong_augment, weak_augment
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


def train(p: Params):
    device = torch.device(p.client_gpu)
    model = get_model(p.model_name, p.dataset, p.num_class, p.feature_dim).to(device)
    model.load_state_dict(p.model_state)

    model_g = get_model(p.model_name, p.dataset, p.num_class, p.feature_dim).to(device)
    model_g.load_state_dict(p.model_state)
    model_g.eval()
    for parameter in model_g.parameters():
        parameter.requires_grad_(False)

    loaders = build_fixmatch_loaders(p.train_set, p.batch_size, p.unlabeled_ratio)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=p.lr,
        momentum=p.momentum,
        weight_decay=p.weight_decay,
    )
    class_counts = torch.zeros(p.num_class, device=device)
    total_loss = 0.0
    num_batches = 0
    u_total = 0
    u_active_count = 0
    hc_count = 0
    hc_correct = 0
    lc_count = 0
    lc_hit = 0
    global_correct = 0
    local_correct = 0

    model.train()
    for local_epoch in range(p.epochs):
        for labeled_batch, (x_u, y_u) in iterate_ssl_batches(loaders):
            assert labeled_batch is not None
            x_l, y_l = labeled_batch
            x_l, y_l = x_l.to(device), y_l.to(device)
            x_u, y_u = x_u.to(device), y_u.to(device)
            x_l = weak_augment(x_l, p.dataset)
            x_u_w = weak_augment(x_u, p.dataset)
            x_u_s = strong_augment(x_u, p.dataset)

            inputs = torch.cat((x_l, x_u_w, x_u_s))
            z = model.extractor(inputs)
            logits = model.classifier(z)

            batch_size = x_l.size(0)
            unlabeled_size = x_u.size(0)
            logits_l = logits[:batch_size]
            logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
            loss_s = F.cross_entropy(logits_l, y_l)

            with torch.no_grad():
                y_g = model_g(x_u_w)  # 全局 logits（论文 Eq.3 的 y_i）
                p_g = torch.softmax(y_g / p.temperature, dim=-1)
                confidence_g, y_hat = p_g.max(dim=1)

            # 高置信度判定：max(y_i) > τ，仅用全局 logits（论文 §5.2.1）
            hc_mask = confidence_g.ge(p.confidence).float()
            # hc 类别集合 ξ={ŷ_i}，同时作为 L_u 的伪标签目标（Eq.9）
            xi_hc = F.one_hot(y_hat, p.num_class).bool()
            loss_u = masked_kl_loss(logits_u_s, xi_hc.float(), hc_mask)

            # ══════════════════════════════════════════════════════════════
            # ICPL 类别集合构建（论文 Eq.3-4）
            # 每个样本的类别集合 ξ 表示"它可能是哪些类"，用于：
            #   1) 定义正代理（损失处按 hc/lc 分别构造）
            #   2) 判定负样本（类别集合不相交 → 负样本）
            #   ξ：有标签={y_i}；hc={ŷ_i}；lc={c | y_i(c) > P_G'(Y(c))}
            # ══════════════════════════════════════════════════════════════
            # 有标签样本：类别集合 = 真实类（one-hot）
            xi_l = F.one_hot(y_l, p.num_class).bool()
            # 无标签样本：全局置信度达标 → hc
            is_hc = hc_mask.bool()

            # 服务器 EMA 的全局类别先验 P_G'(Y)（各类别占比），动态逐类阈值
            prior = p.global_class_dist.to(device=p_g.device, dtype=p_g.dtype)
            # 低置信度样本的犹豫集合 ξ_lc：全局概率高于类别先验的类
            # （模型认为可能是这些类，但不确信唯一——常含多个尾部类）
            xi_lc = p_g > prior.unsqueeze(0)

            # 无标签样本的类别集合：
            #   hc → 全局硬伪标签类 ξ={ŷ_i}；lc → 犹豫集合
            xi_u = torch.where(is_hc.unsqueeze(1), xi_hc, xi_lc)
            # 本地 logits 的分布 ỹ_i（论文 Eq.3/6）：lc 正代理的加权源
            y_tilde = torch.softmax(logits_u_w.detach(), dim=-1)

            # ══════════════════════════════════════════════════════════════
            # ICPL 特征池构建
            # 池 = [有标签 | 无标签弱]：仅弱增强特征（论文 Eq.3 z_i=f_m(T_w(u_i))）
            # 无标签样本 = 池的第二块，有标签只进池（当负样本/锚点用）
            # ══════════════════════════════════════════════════════════════
            # 1) 活跃无标签下标：类别集合非空（犹豫集合全空则无法锚定，剔除）
            active_idx = torch.where(xi_u.any(dim=1))[0]

            # 2) 池特征 = [有标签特征 | 活跃无标签弱特征]
            z_l = z[:batch_size]  # 有标签特征（全部进池）
            # 无标签弱特征，仅保留活跃样本（强增强不参与 ICPL）
            z_u_active = z[batch_size : batch_size + unlabeled_size][active_idx]
            z_pool = torch.cat((z_l, z_u_active), dim=0)

            # 3) 池类别集合（含标注样本，供负样本 overlap 判定）
            xi_pool = torch.cat((xi_l, xi_u[active_idx]), dim=0)

            # 4) 无标签块起点：池 = [有标签块 | 无标签块]，只有无标签块产生对比损失
            unlabeled_start = len(y_l)
            unlabeled_count = len(active_idx)

            # ══════════════════════════════════════════════════════════════
            # ICPL 损失（论文 Eq.6-8）
            # 每个无标签样本 z_i 与"正代理 ω_i"做对比：
            #   拉近 z_i·ω_i，推离所有类别不相交的池样本（Eq.7）
            #   hc 正代理 = 伪标签类代理 ω_k^{ŷ_i}（Eq.6 上）
            #   lc 正代理 = ξ 内按本地概率加权 Σ_{c'∈ξ} ỹ_i(c')·ω_k^{c'}（Eq.6 下）
            #   classifier.weight 每行就是全局类代理 ω_G^c
            # ══════════════════════════════════════════════════════════════
            if unlabeled_count > 0:
                z_norm = F.normalize(z_pool, p=2, dim=1)  # z_i（池）
                omega_G = F.normalize(model.classifier.weight, p=2, dim=1)  # 类代理

                # 活跃无标签样本的元数据（行号与池中无标签块一一对应）
                z_unlabeled = z_norm[unlabeled_start:]  # [U, D] 特征
                xi_unlabeled = xi_pool[unlabeled_start:]  # [U, C] 类别集合
                y_hat_active = y_hat[active_idx]  # [U] 全局伪标签（hc 用）
                y_tilde_active = y_tilde[active_idx]  # [U, C] 本地概率（lc 用）
                is_hc_active = is_hc[active_idx]  # [U] bool 高置信度标记

                # 负相似度：池内两两相似度，类别不相交者作为负样本（Eq.7）
                overlap = (xi_pool.unsqueeze(1) & xi_pool.unsqueeze(0)).any(dim=2)
                negative_mask = ~overlap  # 不共享类别 → 候选负样本
                negative_mask.fill_diagonal_(False)  # 自身不算负样本
                pairwise_sim = z_norm @ z_norm.T  # [P, P]
                negative_mask &= pairwise_sim >= 1e-6  # 去掉"平凡负样本"
                neg_sim = pairwise_sim.masked_fill(
                    ~negative_mask, float("-inf")
                )  # [P, P]
                neg_sim_unlabeled = neg_sim[unlabeled_start:]  # 只取无标签块的行 [U, P]

                # ── hc 组：正代理 = 伪标签类代理 ω_G[ŷ_i] ──
                loss_c_hc = 0.0
                z_hc = z_unlabeled[is_hc_active]
                if z_hc.size(0) > 0:
                    omega_hc = omega_G[y_hat_active[is_hc_active]]  # ω_k^{ŷ_i}
                    pos_hc = (z_hc * omega_hc).sum(dim=1)  # z_i·ω_i^{hc}
                    logits_hc = torch.cat(
                        (pos_hc.unsqueeze(1), neg_sim_unlabeled[is_hc_active]), dim=1
                    )
                    loss_c_hc = F.cross_entropy(
                        logits_hc,
                        torch.zeros(z_hc.size(0), dtype=torch.long, device=device),
                    )

                # ── lc 组：正代理 = ξ 内按本地概率加权（Eq.6 下）──
                loss_c_lc = 0.0
                z_lc = z_unlabeled[~is_hc_active]
                if z_lc.size(0) > 0:
                    xi_lc_active = xi_unlabeled[~is_hc_active]  # 犹豫集合 ξ
                    y_tilde_lc = y_tilde_active[~is_hc_active]  # 本地概率 ỹ_i
                    omega_lc = (
                        y_tilde_lc * xi_lc_active.float()
                    ) @ omega_G  # Σ_{c'∈ξ} ỹ_i(c')·ω_k^{c'}
                    pos_lc = (z_lc * omega_lc).sum(dim=1)  # z_i·ω_i^{lc}
                    logits_lc = torch.cat(
                        (pos_lc.unsqueeze(1), neg_sim_unlabeled[~is_hc_active]), dim=1
                    )
                    loss_c_lc = F.cross_entropy(
                        logits_lc,
                        torch.zeros(z_lc.size(0), dtype=torch.long, device=device),
                    )

                # hc/lc 两项各自平均后求和（Eq.8）
                loss_c = loss_c_hc + loss_c_lc
            else:
                # 无活跃无标签样本：返回 0（乘 0 保持计算图连通）
                loss_c = z_pool.sum() * 0.0

            loss = loss_s + p.lam * loss_u + p.lam * loss_c

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if local_epoch == p.epochs - 1:
                class_counts += F.one_hot(y_l, p.num_class).float().sum(dim=0)
                valid = hc_mask.bool()
                class_counts += F.one_hot(y_hat[valid], p.num_class).float().sum(dim=0)

                u_batch_size = len(y_u)
                u_total += u_batch_size
                u_active_count += len(active_idx)
                cur_hc = valid.sum().item()
                hc_count += cur_hc
                hc_correct += ((y_hat == y_u) & valid).sum().item()
                cur_lc = u_batch_size - cur_hc
                lc_count += cur_lc
                # 检查真实标签 y_u 是否在犹豫候选集 xi_lc 中
                y_u_indices = torch.arange(u_batch_size, device=device)
                in_lc = xi_lc[y_u_indices, y_u] & (~valid)
                lc_hit += in_lc.sum().item()
                global_correct += (y_hat == y_u).sum().item()
                y_local = y_tilde.argmax(dim=-1)
                local_correct += (y_local == y_u).sum().item()
            total_loss += loss.item()
            num_batches += 1

    state = clone_cpu_state(model.state_dict())
    return {
        "loss": total_loss / num_batches,
        "state": state,
        "num_samples": len(p.train_set.y),
        "class_counts": class_counts.cpu().detach().clone(),
        "u_total": u_total,
        "u_active_count": u_active_count,
        "hc_count": hc_count,
        "hc_correct": hc_correct,
        "lc_count": lc_count,
        "lc_hit": lc_hit,
        "global_correct": global_correct,
        "local_correct": local_correct,
    }


class Server(BaseServer):
    def __init__(self, args):
        if args.ssl not in ("sample", "double", "sfd"):
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

        self.gpt = torch.nn.Linear(self.feature_dim, self.num_class).to(self.device)
        self.gpt_optimizer = torch.optim.SGD(self.gpt.parameters(), lr=args.gpt_lr)
        self.global_class_dist = torch.full(
            (self.num_class,), 1.0 / self.num_class, dtype=torch.float32
        )
        self.pseudo_acc = []
        self.valid_ratio = []
        self.num_valid = []
        self.hc_acc = []
        self.hc_ratio = []
        self.lc_hit_ratio = []
        self.active_ratio = []
        self.global_pseudo_acc = []
        self.local_pseudo_acc = []
        self.fedavg_acc = []
        self.fedavg_pred_dist = []
        self.gpt_pred_dist = []
        self.gpt_loss = []
        self.gpt_min_class_distance = []

    def update_global_distribution(self, counts):
        total = torch.stack(counts).to(dtype=torch.float32).sum(dim=0)
        current = total / total.sum()
        self.global_class_dist = (
            self.ema_beta * self.global_class_dist + (1.0 - self.ema_beta) * current
        )

    def get_prediction_distribution(self):
        """返回当前全局模型的预测类别分布，不改变模型参数。"""
        loader = DataLoader(self.test_set, batch_size=128, shuffle=False)
        total = 0
        pred_counts = torch.zeros(self.num_class, dtype=torch.long)

        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            for x, y, *_ in loader:
                x_norm = prepare_input_batch(x.to(self.device), self.dataset)
                logits = self.model(x_norm)
                prediction = logits.argmax(dim=1)
                target = y.to(self.device)
                total += target.numel()
                pred_counts += torch.bincount(
                    prediction.cpu(), minlength=self.num_class
                )
        self.model.cpu()
        return pred_counts.float() / total

    def update_gpt(self, states, weights):
        classifier_states = [
            {
                "weight": state["classifier.weight"],
                "bias": state["classifier.bias"],
            }
            for state in states
        ]
        avg_classifier = param_aggregate(classifier_states, weights)
        self.gpt.load_state_dict(avg_classifier)

        proxies = torch.cat([state["classifier.weight"] for state in states], dim=0)
        labels = torch.arange(self.num_class, device=self.device).repeat(len(states))
        loader = DataLoader(
            TensorDataset(proxies.to(self.device), labels),
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
                loss = dist_contrastive_loss(
                    proxy,
                    self.gpt.weight,
                    label,
                    margin=min(max_dist.item(), self.gpt_threshold),
                )
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
        return total_loss / num_batches, min_class_distance

    def fit(self):
        num_join = max(1, int(self.num_clients * self.join_ratio))
        for r in range(self.rounds):
            start = time.time()
            print(f"\n--- ProxyFL-SSL Round {r + 1}/{self.rounds} ---")

            selected = sorted(torch.randperm(self.num_clients)[:num_join].tolist())
            print(f"Selected clients: {selected}")

            parameters = [
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
            results = self.run_clients(train, parameters)

            # 汇集各客户端的回传结果，计算聚合权重与统计量
            states = []
            sample_counts = []
            class_counts = []
            total_loss = 0.0
            total_u = 0
            total_active = 0
            total_hc = 0
            total_hc_correct = 0
            total_lc = 0
            total_lc_hit = 0
            total_global_correct = 0
            total_local_correct = 0
            for res in results.values():
                states.append(res["state"])
                sample_counts.append(res["num_samples"])
                class_counts.append(res["class_counts"])
                total_loss += res["loss"]
                total_u += res["u_total"]
                total_active += res["u_active_count"]
                total_hc += res["hc_count"]
                total_hc_correct += res["hc_correct"]
                total_lc += res["lc_count"]
                total_lc_hit += res["lc_hit"]
                total_global_correct += res["global_correct"]
                total_local_correct += res["local_correct"]
            total_samples = sum(sample_counts)
            weights = [count / total_samples for count in sample_counts]
            self.model.load_state_dict(param_aggregate(states, weights))
            fedavg_acc = evaluate_model(self.model, self.test_set, self.device)
            fedavg_pred_dist = self.get_prediction_distribution()
            self.update_global_distribution(class_counts)
            gpt_loss, min_class_distance = self.update_gpt(states, weights)
            gpt_pred_dist = self.get_prediction_distribution()

            self.fedavg_acc.append(fedavg_acc)
            self.fedavg_pred_dist.append(fedavg_pred_dist)
            self.gpt_pred_dist.append(gpt_pred_dist)
            self.gpt_loss.append(gpt_loss)
            self.gpt_min_class_distance.append(min_class_distance)

            self.loss.append(total_loss / num_join)
            self.evaluate()

            u_tot_safe = max(1, total_u)
            hc_ratio = total_hc / u_tot_safe
            hc_acc = total_hc_correct / max(1, total_hc)
            lc_hit_ratio = total_lc_hit / max(1, total_lc)
            active_ratio = total_active / u_tot_safe
            global_pseudo_acc = total_global_correct / u_tot_safe
            local_pseudo_acc = total_local_correct / u_tot_safe

            # 保持旧字段兼容
            self.pseudo_acc.append(global_pseudo_acc)
            self.valid_ratio.append(hc_ratio)
            self.num_valid.append(total_hc)

            self.hc_ratio.append(hc_ratio)
            self.hc_acc.append(hc_acc)
            self.lc_hit_ratio.append(lc_hit_ratio)
            self.active_ratio.append(active_ratio)
            self.global_pseudo_acc.append(global_pseudo_acc)
            self.local_pseudo_acc.append(local_pseudo_acc)

            print(
                f"Global Accuracy: {self.acc[-1]:.2f}%, Avg Loss: {self.loss[-1]:.4f}"
            )
            print(
                f"U-Usage: HC Ratio={hc_ratio * 100:.2f}%, Active Ratio={active_ratio * 100:.2f}% | "
                f"HC Acc: {hc_acc * 100:.2f}%, LC Hit Ratio: {lc_hit_ratio * 100:.2f}%"
            )
            print(
                f"Direct Pseudo Acc: Global={global_pseudo_acc * 100:.2f}%, Local={local_pseudo_acc * 100:.2f}%"
            )
            print(
                f"FedAvg Acc: {fedavg_acc:.2f}%, GPT Loss: {gpt_loss:.4f}, GPT Min Class Dist: {min_class_distance:.4f}"
            )
            fmt = lambda d: "[" + ", ".join(f"{v:.3f}" for v in d.tolist()) + "]"
            print(f"Pred Dist: {fmt(fedavg_pred_dist)} -> {fmt(gpt_pred_dist)}")
            print(f"Round finished in {time.time() - start:.2f} seconds")

    def save(self):
        self.deal_save(
            {
                "acc": self.acc,
                "loss": self.loss,
                "pseudo_acc": self.pseudo_acc,
                "valid_ratio": self.valid_ratio,
                "num_valid": self.num_valid,
                "hc_ratio": self.hc_ratio,
                "hc_acc": self.hc_acc,
                "lc_hit_ratio": self.lc_hit_ratio,
                "active_ratio": self.active_ratio,
                "global_pseudo_acc": self.global_pseudo_acc,
                "local_pseudo_acc": self.local_pseudo_acc,
                "fedavg_acc": self.fedavg_acc,
                "fedavg_pred_dist": self.fedavg_pred_dist,
                "gpt_pred_dist": self.gpt_pred_dist,
                "gpt_loss": self.gpt_loss,
                "gpt_min_class_distance": self.gpt_min_class_distance,
                "global_class_dist": self.global_class_dist.cpu().detach().clone(),
            },
            {"global_model": self.model.state_dict(), "gpt": self.gpt.state_dict()},
        )
