import os
import time
from dataclasses import asdict, dataclass

import numpy as np
import ray
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src import TrainingFailureError

from .utils import (
    BaseParams,
    BaseServer,
    check_losses,
    clone_cpu_state,
    evaluate,
    extract_prototypes,
    fmt_num,
    get_model,
)
from .utils.topology import compute_mh_weights, generate_adjacency_matrix


def get_path(args):
    adj_suffix = f"{args.adj_type}"
    if args.adj_type == "random":
        adj_suffix += f"_{fmt_num(args.edge_p)}"
    elif args.adj_type == "small_world":
        adj_suffix += f"_{fmt_num(args.k_small_world)}_{fmt_num(args.edge_p)}"
    elif args.adj_type == "scale_free":
        adj_suffix += f"_{fmt_num(args.m_scale_free)}"

    args.file_name = (
        f"{args.common_name}_{adj_suffix}_{fmt_num(args.lambda_sa)}"
        f"_{fmt_num(args.eta)}_{fmt_num(args.lambda_so)}"
    )
    return os.path.join(args.log_path, f"{args.file_name}_{args.cur_time}.log")


@dataclass
class Params(BaseParams):
    consensus_P: torch.Tensor
    lambda_sa: float
    lambda_so: float
    confidence_mode: str


def train(p: Params):
    """
    DFedSET Worker: 联合训练 + S/W 原型提取。
    """
    device = torch.device(p.client_gpu)

    # 1. 初始化模型
    model = get_model(p).to(device)
    model.load_state_dict(p.model_state)
    consensus_P = p.consensus_P.to(device)

    # 2. 设置优化器与数据加载器
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=p.lr,
        momentum=p.momentum,
        weight_decay=p.weight_decay,
    )
    loader = DataLoader(p.train_set, batch_size=p.batch_size, shuffle=True)

    # 3. 本地训练
    model.train()
    total_loss, num_batches = 0.0, 0
    for _ in range(p.epochs):
        for x, y, *_ in loader:
            x, y = x.to(device), y.to(device)
            features = model.extractor(x)
            logits = model.classifier(features)

            loss = F.cross_entropy(logits, y)
            if p.lambda_sa != 0 or p.lambda_so != 0:
                target_protos = consensus_P[y]
                if p.lambda_sa != 0:
                    loss = loss + p.lambda_sa * F.mse_loss(features, target_protos)
                if p.lambda_so != 0:
                    loss = (
                        loss
                        + p.lambda_so
                        * (
                            1 - F.cosine_similarity(features, target_protos, dim=-1)
                        ).mean()
                    )

            optimizer.zero_grad()
            check_losses(loss, locals())
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

    # 4. 提取本地最新原型 (S 和 W)
    local_protos, local_counts = extract_prototypes(
        model, loader, p.num_class, p.feature_dim, device, return_counts=True
    )
    if p.confidence_mode == "count":
        confidence = local_counts.unsqueeze(-1).float()
    elif p.confidence_mode == "none":
        confidence = torch.ones_like(local_counts.unsqueeze(-1))
    else:
        confidence = torch.log(1 + local_counts.unsqueeze(-1))
    S = confidence * local_protos
    W = confidence

    new_state = clone_cpu_state(model.state_dict())

    return {
        "loss": total_loss / max(1, num_batches),
        "state": new_state,
        "S": S.cpu().detach().clone(),
        "W": W.cpu().detach().clone(),
        "counts": local_counts.cpu().detach().clone(),
    }


class Server(BaseServer):
    def __init__(self, args):
        super().__init__(args, pfl=True)
        self.lambda_sa = args.lambda_sa
        self.lambda_so = args.lambda_so
        self.ablate = args.ablate
        self.gamma_global = self.ablate.get("gamma_global", 0.001)
        self.eta = args.eta

        # 生成静态物理拓扑邻接矩阵
        adj = generate_adjacency_matrix(args).to(self.device).float()
        self.adj = adj

        # 1. 计算静态 Metropolis-Hastings 双随机矩阵 (恒满足行和为 1、列和为 1 且对称)
        self.M_avg = compute_mh_weights(adj, device=self.device)

        # 状态缓存
        self.S_cache = [
            torch.zeros(self.num_class, args.feature_dim)
            for _ in range(self.num_clients)
        ]
        self.W_cache = [torch.zeros(self.num_class, 1) for _ in range(self.num_clients)]
        self.counts_cache = [
            torch.zeros(self.num_class) for _ in range(self.num_clients)
        ]
        self.consensus_P = [
            torch.zeros(self.num_class, args.feature_dim)
            for _ in range(self.num_clients)
        ]

        # 每个客户端追踪自身 GSD 的历史 EMA（P2P 触发基准）
        self.local_gsd_ema = torch.full((self.num_clients, 1), 0.5, device=self.device)
        self.gsd_log: list[list[float]] = []
        self.num_triggered_log: list[int] = []
        self.triggered_ids_log: list[list[int]] = []

        # 预计算 extractor 参数 flatten 映射（矩阵聚合用）
        self.extractor_keys = [
            k for k in self.clients_state[0] if k.startswith("extractor.")
        ]
        self.extractor_param_info = []
        total = 0
        for k in self.extractor_keys:
            shape = self.clients_state[0][k].shape
            n = self.clients_state[0][k].numel()
            self.extractor_param_info.append((k, shape, total, total + n))
            total += n
        self.extractor_total_size = total

    def fit(self):
        # num_join_clients = int(self.num_clients * self.join_ratio)
        # num_join_clients = max(1, num_join_clients)
        num_join_clients = self.num_clients
        selected_clients = np.arange(self.num_clients)

        for r in range(self.start_round, self.rounds):
            t0 = time.time()
            print(f"\n--- DFedSET Round {r + 1}/{self.rounds} ---")
            # selected_clients = np.random.choice(
            #     self.num_clients, num_join_clients, replace=False
            # )

            ablate = self.ablate
            confidence_mode = ablate.get("confidence", "log")
            trigger_mode = ablate.get("trigger", "adaptive")
            use_redirect = ablate.get("aggregator", True)

            base_params = self.build_base_params(selected_clients)
            for base in base_params:
                base.model_state = self.clients_state[base.client_id]

            p = [
                Params(
                    **asdict(base),
                    consensus_P=self.consensus_P[base.client_id],
                    lambda_sa=self.lambda_sa,
                    lambda_so=self.lambda_so,
                    confidence_mode=confidence_mode,
                )
                for base in base_params
            ]
            results = self.run_clients(train, p)

            total_loss = 0.0
            for cid, res in results.items():
                total_loss += res["loss"]
                if not np.isfinite(res["loss"]):
                    raise TrainingFailureError(
                        f"Client {cid} loss is NaN/Inf: {res['loss']}"
                    )
                for k, v in res["state"].items():
                    if not torch.isfinite(v).all():
                        raise TrainingFailureError(
                            f"Client {cid} state[{k}] has NaN/Inf"
                        )
                self.clients_state[cid] = res["state"]

                # 接力缓存法核心：如果某类别在本轮本地训练中没有样本计数，继承上一轮结束时的值而不是清零！
                counts = res["counts"]
                S_local = res["S"]
                W_local = res["W"]
                for c in range(self.num_class):
                    if counts[c] > 0:
                        self.S_cache[cid][c] = S_local[c]
                        self.W_cache[cid][c] = W_local[c]
                    elif not ablate.get("relay", True):
                        self.S_cache[cid][c].zero_()
                        self.W_cache[cid][c].zero_()
                self.counts_cache[cid] = counts

            self.loss.append(total_loss / num_join_clients)

            # 1. 计算 Gossip 前的本地原型 local_P
            S = torch.stack([self.S_cache[i] for i in range(self.num_clients)]).to(
                self.device
            )
            W = torch.stack([self.W_cache[i] for i in range(self.num_clients)]).to(
                self.device
            )
            local_P = S / (W + 1e-12)

            # ===== GPU 向量化计算本地 GSD =====
            counts = torch.stack(self.counts_cache).to(self.device)  # [N, C]
            total_n = counts.sum(dim=1, keepdim=True)  # [N, 1]
            probs = counts / torch.clamp(total_n, min=1e-12)  # [N, C]

            norm_local = torch.norm(local_P, dim=-1)  # [N, C]
            consensus_P_tensor = torch.stack(self.consensus_P).to(
                self.device
            )  # [N, C, D]
            norm_consensus = torch.norm(consensus_P_tensor, dim=-1)  # [N, C]

            valid_mask = (norm_local > 1e-8) & (norm_consensus > 1e-8)
            cos_sim = F.cosine_similarity(local_P, consensus_P_tensor, dim=-1)  # [N, C]
            cos_sim = torch.where(valid_mask, cos_sim, torch.zeros_like(cos_sim))

            gsd_raw = (probs * (1.0 - cos_sim)).sum(dim=1)  # [N]
            gsd_tensor = torch.where(
                total_n.squeeze(1) > 0, gsd_raw, torch.ones_like(gsd_raw)
            )

            self.gsd_log.append(gsd_tensor.tolist())
            D = gsd_tensor.view(self.num_clients, 1)

            # 3. 搭载 Gossip (S, W, D): 普通平均聚合 (using M_avg)
            S_flat, W_flat = S.view(self.num_clients, -1), W.view(self.num_clients, -1)
            S_flat = torch.mm(self.M_avg, S_flat)
            W_flat = torch.mm(self.M_avg, W_flat)
            D = torch.mm(self.M_avg, D)

            # 提取更新后的原型共识
            W_tensor = W_flat.view(self.num_clients, self.num_class, 1)
            S_tensor = S_flat.view(self.num_clients, self.num_class, self.feature_dim)
            consensus = S_tensor / (W_tensor + 1e-12)

            for i in range(self.num_clients):
                self.S_cache[i], self.W_cache[i], self.consensus_P[i] = (
                    S_tensor[i].cpu(),
                    W_tensor[i].cpu(),
                    consensus[i].cpu(),
                )
            # 4. 触发机制：自适应 / 全局阈值 / 全触发
            local_trigger = torch.zeros(
                self.num_clients, dtype=torch.bool, device=self.device
            )
            if trigger_mode == "all":
                local_trigger.fill_(True)
                print("  [Ablation] All clients force triggered.")
            elif trigger_mode == "global":
                gamma_global = self.gamma_global
                if r == 0:
                    local_trigger.fill_(True)
                    lines = [
                        f"Client {i} | Local GSD: {gsd_tensor[i].item():.6f} | [Warmup] Force Triggered"
                        for i in range(self.num_clients)
                    ]
                    print("\n".join(lines))
                elif r == 1:
                    local_trigger.fill_(True)
                    lines = []
                    for i in range(self.num_clients):
                        self.local_gsd_ema[i] = gsd_tensor[i]
                        lines.append(
                            f"Client {i} | Local GSD: {gsd_tensor[i].item():.6f} | [Transition] Force Triggered, EMA initialized to {self.local_gsd_ema[i].item():.6f}"
                        )
                    print("\n".join(lines))
                else:
                    lines = []
                    for i in range(self.num_clients):
                        self.local_gsd_ema[i] = (
                            self.eta * self.local_gsd_ema[i]
                            + (1.0 - self.eta) * gsd_tensor[i]
                        )
                        lines.append(
                            f"Client {i} | Local GSD: {gsd_tensor[i].item():.6f} | Own EMA: {self.local_gsd_ema[i].item():.6f} | Gamma Global: {gamma_global:.6f}"
                        )
                        if (
                            gsd_tensor[i] > gamma_global
                            or self.counts_cache[i].sum() == 0
                        ):
                            local_trigger[i] = True
                    print("\n".join(lines))
            else:
                # "adaptive"（默认）
                if r == 0:
                    local_trigger.fill_(True)
                    lines = [
                        f"Client {i} | Local GSD: {gsd_tensor[i].item():.6f} | [Warmup] Force Triggered"
                        for i in range(self.num_clients)
                    ]
                    print("\n".join(lines))
                elif r == 1:
                    local_trigger.fill_(True)
                    lines = []
                    for i in range(self.num_clients):
                        self.local_gsd_ema[i] = gsd_tensor[i]
                        lines.append(
                            f"Client {i} | Local GSD: {gsd_tensor[i].item():.6f} | [Transition] Force Triggered, EMA initialized to {self.local_gsd_ema[i].item():.6f}"
                        )
                    print("\n".join(lines))
                else:
                    lines = []
                    for i in range(self.num_clients):
                        self.local_gsd_ema[i] = (
                            self.eta * self.local_gsd_ema[i]
                            + (1.0 - self.eta) * gsd_tensor[i]
                        )
                        current_gamma = self.local_gsd_ema[i]
                        lines.append(
                            f"Client {i} | Local GSD: {gsd_tensor[i].item():.6f} | Own EMA: {self.local_gsd_ema[i].item():.6f} | Gamma: {current_gamma.item():.6f}"
                        )
                        if (
                            gsd_tensor[i] > current_gamma
                            or self.counts_cache[i].sum() == 0
                        ):
                            local_trigger[i] = True
                    print("\n".join(lines))

            # 邻域扩展激活（被激活链路双向开启，保持活跃子图无向对称）
            trigger_mask = local_trigger.clone()
            if use_redirect:
                for i in range(self.num_clients):
                    if local_trigger[i]:
                        neighbors = torch.where(self.adj[:, i] > 0)[0]
                        for nb in neighbors:
                            trigger_mask[int(nb.item())] = True

            triggered_ids = torch.where(trigger_mask)[0].tolist()
            num_triggered = len(triggered_ids)
            print(
                f"Event Triggered: {num_triggered}/{self.num_clients} clients will sync parameters (triggered clients: {triggered_ids})."
            )
            self.num_triggered_log.append(num_triggered)
            self.triggered_ids_log.append(triggered_ids)
            # 5. Extractor: 矩阵化 Flatten → torch.mm → Unflatten 聚合
            if self.extractor_keys and self.extractor_total_size > 0:
                flat = torch.zeros(
                    self.num_clients, self.extractor_total_size, device=self.device
                )
                for i in range(self.num_clients):
                    offset = 0
                    for k, _, start, end in self.extractor_param_info:
                        n = end - start
                        t = self.clients_state[i][k].to(self.device, non_blocking=True)
                        flat[i, offset : offset + n].copy_(t.reshape(-1))
                        offset += n

                # ===== GPU 向量化构建权重矩阵 W =====
                mask = trigger_mask.float().to(self.device)  # [N]
                eye_mask = torch.eye(self.num_clients, device=self.device)

                if use_redirect:
                    # ===== Redirect 重定向（沉默邻居权重重定向到自身） =====
                    phys_adj = self.adj * (1.0 - eye_mask)
                    active_mask = phys_adj * mask.unsqueeze(0)
                    silent_mask = phys_adj * (1.0 - mask.unsqueeze(0))
                    W = self.M_avg * active_mask
                    diag_vals = self.M_avg.diagonal() + (self.M_avg * silent_mask).sum(
                        dim=1
                    )
                    W = W + torch.diag(diag_vals)
                    row_identities = (1.0 - mask).unsqueeze(1) * eye_mask
                    W = W * mask.unsqueeze(1) + row_identities
                else:
                    # ===== Plain 聚合（丢弃沉默边，剩余边重归一化） =====
                    active_matrix = mask.unsqueeze(1) * mask.unsqueeze(0)
                    W = self.M_avg * active_matrix
                    row_identities = (1.0 - mask).unsqueeze(1) * eye_mask
                    W = W * mask.unsqueeze(1) + row_identities
                    row_sums = W.sum(dim=1, keepdim=True).clamp(min=1e-12)
                    W = W / row_sums
                new_flat = torch.mm(W, flat)
                new_flat_cpu = new_flat.cpu()
                for i in range(self.num_clients):
                    for k, shape, start, end in self.extractor_param_info:
                        n = end - start
                        self.clients_state[i][k] = (
                            new_flat_cpu[i, start:end].view(shape).clone()
                        )
            for i in range(self.num_clients):
                for k, v in self.clients_state[i].items():
                    if not torch.isfinite(v).all():
                        raise TrainingFailureError(
                            f"After aggregation, client {i} state[{k}] has NaN/Inf"
                        )

            self.evaluate(protos=self.consensus_P)
            print(
                f"Avg Loss: {self.loss[-1]:.4f}, Acc: {self.acc[-1]:.2f}%, Proto Acc: {self.acc_proto[-1]:.2f}%"
            )
            print(f"Round finished in {time.time() - t0:.2f} seconds")
            metrics = {
                "acc": self.acc,
                "acc_proto": self.acc_proto,
                "loss": self.loss,
                "gsd_log": self.gsd_log,
                "num_triggered_log": self.num_triggered_log,
                "triggered_ids_log": self.triggered_ids_log,
            }
            params = {
                "client": self.clients_state,
                "aux": {
                    "consensus_P": self.consensus_P,
                    "S_cache": self.S_cache,
                    "W_cache": self.W_cache,
                    "counts_cache": self.counts_cache,
                    "local_gsd_ema": self.local_gsd_ema,
                },
            }
            self.save_checkpoint(r + 1, metrics, params)

    def evaluate(self, model_states=None, protos=None):
        target_states = model_states if model_states is not None else self.clients_state
        futures = []
        for i in range(self.num_clients):
            client_proto = protos[i].cpu() if protos is not None else None
            futures.append(
                evaluate.options(
                    num_gpus=self.ray_gpu_fraction,
                    scheduling_strategy="SPREAD",
                ).remote(
                    self.model_name,
                    self.dataset,
                    self.feature_dim,
                    target_states[i],
                    self.test_set_refs[i],
                    self.client_gpu[i],
                    self.num_class,
                    client_proto,
                )
            )
        results = ray.get(futures)
        self.acc.append(sum(r["acc"] for r in results) / self.num_clients)
        if protos is not None:
            self.acc_proto.append(sum(r["p_acc"] for r in results) / self.num_clients)

    def load_checkpoint(self, path):
        params = super().load_checkpoint(path)
        aux = params["aux"]
        self.consensus_P = aux["consensus_P"]
        self.S_cache = aux["S_cache"]
        self.W_cache = aux["W_cache"]
        self.counts_cache = aux["counts_cache"]
        self.local_gsd_ema = aux["local_gsd_ema"].to(self.device)
        return params

    def save(self):
        metrics = {
            "acc": self.acc,
            "acc_proto": self.acc_proto,
            "loss": self.loss,
            "gsd": self.gsd_log,
            "num_triggered": self.num_triggered_log,
            "triggered_ids": self.triggered_ids_log,
        }
        params = {"client": self.clients_state, "proto": self.consensus_P}
        self.deal_save(metrics, params)
