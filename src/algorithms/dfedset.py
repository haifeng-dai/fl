import gc
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .utils import (
    BaseServer,
    ce_loss,
    compute_mh_weights,
    evaluate_prototype,
    extract_prototypes,
    generate_adjacency_matrix,
    get_model,
    mse_loss,
)


def get_path(args):
    adj_suffix = f"{args.adj_type}"
    if args.adj_type == "random":
        adj_suffix += f"_{args.edge_p}"
    elif args.adj_type == "small_world":
        adj_suffix += f"_{args.k_small_world}_{args.edge_p}"
    elif args.adj_type == "scale_free":
        adj_suffix += f"_{args.m_scale_free}"

    args.file_name = (
        f"{args.common_name}_{adj_suffix}_{args.lambda_sa}_{args.eta}_{args.lambda_so}"
    )
    return os.path.join(args.log_path, f"{args.file_name}_{args.cur_time}.log")


def client_worker(params):
    """
    DFedSET Worker: 联合训练 + S/W 原型提取。
    """
    (
        _,
        device,
        model_state,
        train_set,
        model_name,
        dataset_name,
        lr,
        batch_size,
        epochs,
        feature_dim,
        num_classes,
        consensus_P,
        lambda_sa,
        lambda_so,
        confidence_mode,
    ) = params

    # 1. 初始化模型
    model = get_model(model_name, dataset_name, feature_dim).to(device)
    model.load_state_dict(model_state)
    consensus_P = consensus_P.to(device)

    # 2. 设置优化器与数据加载器
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    # 3. 本地训练
    model.train()
    total_loss, num_batches = 0.0, 0
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            features = model.extractor(x)
            logits = model.classifier(features)

            l_ce = ce_loss(logits, y)
            target_protos = consensus_P[y]
            l_con = mse_loss(features, target_protos)
            l_cos = (1 - F.cosine_similarity(features, target_protos, dim=-1)).mean()
            loss = l_ce + lambda_sa * l_con + lambda_so * l_cos

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

    # 4. 提取本地最新原型 (S 和 W)
    local_protos, local_counts = extract_prototypes(
        model, loader, num_classes, feature_dim, device, return_counts=True
    )
    if confidence_mode == "count":
        confidence = local_counts.unsqueeze(-1).float()
    elif confidence_mode == "none":
        confidence = torch.ones_like(local_counts.unsqueeze(-1))
    else:
        confidence = torch.log(1 + local_counts.unsqueeze(-1))
    S = confidence * local_protos
    W = confidence

    new_state = {k: v.cpu().detach().clone() for k, v in model.state_dict().items()}

    return {
        "loss": total_loss / max(1, num_batches),
        "state": new_state,
        "S": S.cpu().detach().clone(),
        "W": W.cpu().detach().clone(),
        "counts": local_counts.cpu().detach().clone(),
    }


class Server(BaseServer):
    def __init__(self, args):
        super().__init__(pfl=True, args=args)
        # 生成静态物理拓扑邻接矩阵
        adj = generate_adjacency_matrix(args).to(self.device).float()
        self.adj = adj

        # 1. 计算静态 Metropolis-Hastings 双随机矩阵 (恒满足行和为 1、列和为 1 且对称)
        self.M_avg = compute_mh_weights(adj, device=self.device)

        # 列随机矩阵 M_ps（仅 aggregator="pushsum" 模式使用）
        row_sum = self.adj.sum(dim=1, keepdim=True)
        self.M_ps = (self.adj / row_sum).t().to(self.device)

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
        self.eta = self.args.eta
        self.gsd_log: list[list[float]] = []
        self.num_triggered_log: list[int] = []
        self.triggered_ids_log: list[list[int]] = []

        # 预计算 extractor 参数 flatten 映射（矩阵聚合用）
        self.extractor_keys = [
            k for k in self.clients_state[0].keys() if k.startswith("extractor.")
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
        num_join_clients = int(self.num_clients * self.args.join_ratio)
        num_join_clients = max(1, num_join_clients)

        for r in range(self.rounds):
            t0 = time.time()
            print(f"\n--- DFedSET Round {r + 1}/{self.rounds} ---")
            selected_clients = np.random.choice(
                self.num_clients, num_join_clients, replace=False
            )

            ablate = getattr(self.args, "ablate", {})
            confidence_mode = ablate.get("confidence", "log")
            trigger_mode = ablate.get("trigger", "adaptive")
            agg_mode = ablate.get("aggregator", "redirect")

            def get_client_param(i):
                return (
                    i,
                    self.client_gpu[i],
                    self.clients_state[i],
                    self.train_sets[i],
                    self.args.model,
                    self.args.dataset,
                    self.args.lr,
                    self.args.batch_size,
                    self.args.epochs,
                    self.args.feature_dim,
                    self.num_class,
                    self.consensus_P[i],
                    self.args.lambda_sa,
                    self.args.lambda_so,
                    confidence_mode,
                )


            params = [get_client_param(i) for i in selected_clients]
            results = self.run_clients(client_worker, params)

            total_loss = 0.0
            for cid, res in results.items():
                total_loss += res["loss"]
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

            # 2. 计算各客户端本轮的本地 GSD
            all_gsds = []
            for i in range(self.num_clients):
                weights = self.counts_cache[i].to(self.device)
                total_n = weights.sum()
                if total_n > 0:
                    probs = weights / total_n
                    norm_local = torch.norm(local_P[i], dim=-1)
                    norm_consensus = torch.norm(
                        self.consensus_P[i].to(self.device), dim=-1
                    )
                    valid_mask = (norm_local > 1e-8) & (norm_consensus > 1e-8)

                    cos_sim = torch.zeros(self.num_class, device=self.device)
                    if valid_mask.any():
                        cos_sim[valid_mask] = F.cosine_similarity(
                            local_P[i][valid_mask],
                            self.consensus_P[i].to(self.device)[valid_mask],
                            dim=-1,
                        )
                    gsd = (probs * (1.0 - cos_sim)).sum()
                else:
                    gsd = torch.tensor(1.0, device=self.device)
                all_gsds.append(gsd)

            # 将 GSD 装载为列向量进行 Gossip
            gsd_tensor = torch.stack(all_gsds)  # 一次性 stack，去掉冗余 .to(device)
            self.gsd_log.append(gsd_tensor.tolist())  # 批量 CPU 同步，替代逐个 .item()
            D = gsd_tensor.view(self.num_clients, 1)

            # 3. 搭载 Gossip (S, W, D): 普通平均聚合 (using M_avg)
            S_flat, W_flat = S.view(self.num_clients, -1), W.view(self.num_clients, -1)
            S_flat = torch.mm(self.M_avg, S_flat)
            W_flat = torch.mm(self.M_avg, W_flat)
            D = torch.mm(self.M_avg, D)

            # 提取更新后的原型共识
            W_tensor = W_flat.view(self.num_clients, self.num_class, 1)
            S_tensor = S_flat.view(
                self.num_clients, self.num_class, self.args.feature_dim
            )
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
                gamma_global = getattr(self.args, "gamma_global")
                if r == 0:
                    local_trigger.fill_(True)
                    lines = [
                        f"Client {i} | Local GSD: {all_gsds[i].item():.6f} | [Warmup] Force Triggered"
                        for i in range(self.num_clients)
                    ]
                    print("\n".join(lines))
                elif r == 1:
                    local_trigger.fill_(True)
                    lines = []
                    for i in range(self.num_clients):
                        self.local_gsd_ema[i] = all_gsds[i]
                        lines.append(
                            f"Client {i} | Local GSD: {all_gsds[i].item():.6f} | [Transition] Force Triggered, EMA initialized to {self.local_gsd_ema[i].item():.6f}"
                        )
                    print("\n".join(lines))
                else:
                    lines = []
                    for i in range(self.num_clients):
                        self.local_gsd_ema[i] = (
                            self.eta * self.local_gsd_ema[i]
                            + (1.0 - self.eta) * all_gsds[i]
                        )
                        lines.append(
                            f"Client {i} | Local GSD: {all_gsds[i].item():.6f} | Own EMA: {self.local_gsd_ema[i].item():.6f} | Gamma Global: {gamma_global:.6f}"
                        )
                        if all_gsds[i] > gamma_global or self.counts_cache[i].sum() == 0:
                            local_trigger[i] = True
                    print("\n".join(lines))
            else:
                # "adaptive"（默认）
                if r == 0:
                    local_trigger.fill_(True)
                    lines = [
                        f"Client {i} | Local GSD: {all_gsds[i].item():.6f} | [Warmup] Force Triggered"
                        for i in range(self.num_clients)
                    ]
                    print("\n".join(lines))
                elif r == 1:
                    local_trigger.fill_(True)
                    lines = []
                    for i in range(self.num_clients):
                        self.local_gsd_ema[i] = all_gsds[i]
                        lines.append(
                            f"Client {i} | Local GSD: {all_gsds[i].item():.6f} | [Transition] Force Triggered, EMA initialized to {self.local_gsd_ema[i].item():.6f}"
                        )
                    print("\n".join(lines))
                else:
                    lines = []
                    for i in range(self.num_clients):
                        self.local_gsd_ema[i] = (
                            self.eta * self.local_gsd_ema[i]
                            + (1.0 - self.eta) * all_gsds[i]
                        )
                        current_gamma = self.local_gsd_ema[i]
                        lines.append(
                            f"Client {i} | Local GSD: {all_gsds[i].item():.6f} | Own EMA: {self.local_gsd_ema[i].item():.6f} | Gamma: {current_gamma.item():.6f}"
                        )
                        if all_gsds[i] > current_gamma or self.counts_cache[i].sum() == 0:
                            local_trigger[i] = True
                    print("\n".join(lines))

            # 邻域扩展激活（被激活链路双向开启，保持活跃子图无向对称）
            trigger_mask = local_trigger.clone()
            if agg_mode != "pushsum":
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
                if agg_mode == "pushsum":
                    # ===== Push-Sum 列随机聚合 =====
                    M_dynamic = self.M_ps.clone()
                    for j in range(self.num_clients):
                        if not trigger_mask[j]:
                            M_dynamic[:, j] = 0.0
                            M_dynamic[j, j] = 1.0
                    flat = torch.zeros(
                        self.num_clients, self.extractor_total_size, device=self.device
                    )
                    for i in range(self.num_clients):
                        offset = 0
                        for k, _, start, end in self.extractor_param_info:
                            n = end - start
                            t = self.clients_state[i][k].to(self.device, non_blocking=True)
                            flat[i, offset:offset + n].copy_(t.reshape(-1))
                            offset += n

                    new_flat = torch.mm(M_dynamic.to(self.device), flat)
                else:
                    # ===== Redirect 重定向（默认） =====
                    W_redirect = torch.zeros(self.num_clients, self.num_clients)
                    for i in range(self.num_clients):
                        if not trigger_mask[i]:
                            W_redirect[i, i] = 1.0
                            continue
                        neighbors = torch.where(self.adj[i] > 0)[0].tolist()
                        physical = [int(nb) for nb in neighbors if nb != i]
                        active = [nb for nb in physical if trigger_mask[nb]]
                        silent = [nb for nb in physical if not trigger_mask[nb]]
                        W_redirect[i, i] = float(self.M_avg[i, i])
                        for nb in silent:
                            W_redirect[i, i] += float(self.M_avg[i, nb])
                        for nb in active:
                            W_redirect[i, nb] = float(self.M_avg[i, nb])

                    flat = torch.zeros(
                        self.num_clients, self.extractor_total_size, device=self.device
                    )
                    for i in range(self.num_clients):
                        offset = 0
                        for k, _, start, end in self.extractor_param_info:
                            n = end - start
                            t = self.clients_state[i][k].to(self.device, non_blocking=True)
                            flat[i, offset:offset + n].copy_(t.reshape(-1))
                            offset += n

                    new_flat = torch.mm(W_redirect.to(self.device), flat)

                for i in range(self.num_clients):
                    for k, shape, start, end in self.extractor_param_info:
                        n = end - start
                        self.clients_state[i][k] = (
                            new_flat[i, start:end].view(shape).cpu().clone()
                        )

            self.evaluate(protos=self.consensus_P)
            print(
                f"Avg Loss: {self.loss[-1]:.4f}, Acc: {self.acc[-1]:.2f}%, Proto Acc: {self.acc_proto[-1]:.2f}%"
            )
            print(f"Round finished in {time.time() - t0:.2f} seconds")

    def evaluate(self, model_states=None, protos=None):
        super().evaluate(model_states=model_states, protos=None)
        if protos is not None:
            p_accs = []
            target_states = (
                model_states if model_states is not None else self.clients_state
            )
            self.model.to(self.device)
            for i in range(self.num_clients):
                self.model.load_state_dict(target_states[i])
                client_proto = protos[i].to(self.device)
                p_accs.append(
                    evaluate_prototype(
                        self.model, client_proto, self.test_set[i], self.device
                    )
                )
            p_acc = sum(p_accs) / len(p_accs) if p_accs else 0.0
            self.acc_proto.append(p_acc)
            self.model.cpu()

    def save(self):
        f = {
            "acc": {"model": self.acc, "proto": self.acc_proto},
            "loss": self.loss,
            "gsd": self.gsd_log,
            "num_triggered": self.num_triggered_log,
            "triggered_ids": self.triggered_ids_log,
            "state_dict": self.clients_state,
            "consensus_P": self.consensus_P,
        }
        self.deal_save(f)
