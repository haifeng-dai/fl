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
    extract_prototypes,
    generate_adjacency_matrix,
    get_model,
    mse_loss,
    pushsum_param_aggregate,
)


def get_path(args):
    adj_suffix = f"{args.adj_type}"
    if args.adj_type == "random":
        adj_suffix += f"_{args.edge_p}"
    elif args.adj_type == "small_world":
        adj_suffix += f"_{args.k_small_world}_{args.edge_p}"
    elif args.adj_type == "scale_free":
        adj_suffix += f"_{args.m_scale_free}"

    args.file_name = f"{args.common_name}_{adj_suffix}_{args.mu}"
    return os.path.join(args.log_path, f"{args.file_name}_{args.cur_time}.log")


def client_worker(params):
    """
    DFedUP8 Worker: 联合训练 + S/W 原型提取。
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
        mu,
    ) = params

    model = get_model(model_name, dataset_name, feature_dim).to(device)
    model.load_state_dict(model_state)
    consensus_P = consensus_P.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

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
            loss = l_ce + mu * l_con

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

    local_protos, local_counts = extract_prototypes(
        model, loader, num_classes, feature_dim, device, return_counts=True
    )
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
        adj = generate_adjacency_matrix(args).to(self.device).float()

        # 1. 用于 Extractor Push-Sum 的列随机矩阵
        row_sum = adj.sum(dim=1, keepdim=True)
        self.M_ps = (adj / row_sum).t().to(self.device)

        # 2. 用于原型普通平均的双随机矩阵 (MH Weights)
        self.M_avg = compute_mh_weights(adj, device=self.device)

        self.S_cache = [torch.zeros(self.num_class, args.feature_dim) for _ in range(self.num_clients)]
        self.W_cache = [torch.zeros(self.num_class, 1) for _ in range(self.num_clients)]
        self.counts_cache = [torch.zeros(self.num_class) for _ in range(self.num_clients)]
        self.consensus_P = [torch.zeros(self.num_class, args.feature_dim) for _ in range(self.num_clients)]

        # Extractor Push-Sum 权重
        self.W_params = torch.ones(self.num_clients, 1)

        # 移动平均的 GSD 追踪
        self.historical_gsd_ema = torch.full((self.num_clients, 1), 0.5, device=self.device)
        self.eta = 0.9

    def fit(self):
        num_join_clients = int(self.num_clients * self.args.join_ratio)
        num_join_clients = max(1, num_join_clients)

        for r in range(self.rounds):
            t0 = time.time()
            print(f"\n--- DFedUP8 Round {r + 1}/{self.rounds} ---")
            selected_clients = np.random.choice(self.num_clients, num_join_clients, replace=False)

            def get_client_param(i):
                return (
                    i, self.client_gpu[i], self.clients_state[i],
                    self.train_sets[i], self.args.model, self.args.dataset,
                    self.args.lr, self.args.batch_size, self.args.epochs,
                    self.args.feature_dim, self.num_class, self.consensus_P[i], self.args.mu
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
                self.counts_cache[cid] = counts

            self.loss.append(total_loss / num_join_clients)

            gossip_rounds = getattr(self.args, "gossip_rounds", 1)

            # 1. 计算 Gossip 前的本地原型 local_P
            S = torch.stack([self.S_cache[i] for i in range(self.num_clients)]).to(self.device)
            W = torch.stack([self.W_cache[i] for i in range(self.num_clients)]).to(self.device)
            local_P = S / (W + 1e-12)

            # 2. 计算各客户端本轮的本地 GSD
            all_gsds = []
            for i in range(self.num_clients):
                weights = self.counts_cache[i].to(self.device)
                total_n = weights.sum()
                if total_n > 0:
                    probs = weights / total_n
                    norm_local = torch.norm(local_P[i], dim=-1)
                    norm_consensus = torch.norm(self.consensus_P[i].to(self.device), dim=-1)
                    valid_mask = (norm_local > 1e-8) & (norm_consensus > 1e-8)

                    cos_sim = torch.zeros(self.num_class, device=self.device)
                    if valid_mask.any():
                        cos_sim[valid_mask] = F.cosine_similarity(
                            local_P[i][valid_mask],
                            self.consensus_P[i].to(self.device)[valid_mask],
                            dim=-1
                        )
                    gsd = (probs * (1.0 - cos_sim)).sum()
                else:
                    gsd = torch.tensor(1.0, device=self.device)
                all_gsds.append(gsd)

            # 将 GSD 装载为列向量进行 Gossip
            D = torch.stack(all_gsds).to(self.device).view(self.num_clients, 1)

            # 3. 搭载 Gossip (S, W, D): 普通平均聚合 (using M_avg)
            S_flat, W_flat = S.view(self.num_clients, -1), W.view(self.num_clients, -1)
            for _ in range(gossip_rounds):
                S_flat = torch.mm(self.M_avg, S_flat)
                W_flat = torch.mm(self.M_avg, W_flat)
                D = torch.mm(self.M_avg, D)

            # 提取更新后的原型共识
            W_tensor = W_flat.view(self.num_clients, self.num_class, 1)
            S_tensor = S_flat.view(self.num_clients, self.num_class, self.args.feature_dim)
            consensus = S_tensor / (W_tensor + 1e-12)

            for i in range(self.num_clients):
                self.S_cache[i], self.W_cache[i], self.consensus_P[i] = S_tensor[i].cpu(), W_tensor[i].cpu(), consensus[i].cpu()

            # 4. 提取 Gossip 后每个客户端对全网平均 GSD 的估计值并做 EMA 和同步控制
            local_trigger = torch.zeros(self.num_clients, dtype=torch.bool, device=self.device)
            D_est = D
            for i in range(self.num_clients):
                # EMA 均值更新
                self.historical_gsd_ema[i] = self.eta * self.historical_gsd_ema[i] + (1.0 - self.eta) * D_est[i]
                current_gamma = torch.clamp(self.historical_gsd_ema[i] * 0.8, min=0.05)

                print(f"Client {i} | Local GSD: {all_gsds[i].item():.6f} | Est Avg GSD: {D_est[i].item():.6f} | EMA: {self.historical_gsd_ema[i].item():.6f} | Gamma: {current_gamma.item():.6f}")

                if all_gsds[i] > current_gamma or self.counts_cache[i].sum() == 0:
                    local_trigger[i] = True

            # 邻域扩展激活：一旦有节点触发，唤醒其所有邻居协同参数 Gossip (方案 A 核心)
            trigger_mask = local_trigger.clone()
            for i in range(self.num_clients):
                if local_trigger[i]:
                    # 寻找物理通信邻居
                    neighbors = torch.where(self.M_ps[:, i] > 0)[0]
                    for nb in neighbors:
                        trigger_mask[nb] = True

            triggered_ids = torch.where(trigger_mask)[0].tolist()
            num_triggered = len(triggered_ids)
            print(f"Event Triggered: {num_triggered}/{self.num_clients} clients will sync parameters (triggered clients: {triggered_ids}).")

            # 5. Extractor: 动态质量守恒 Push-Sum 聚合
            M_dynamic = self.M_ps.clone()
            for j in range(self.num_clients):
                if not trigger_mask[j]:
                    M_dynamic[:, j] = 0.0
                    M_dynamic[j, j] = 1.0

            new_states, self.W_params = pushsum_param_aggregate(
                self.clients_state, self.W_params, M_dynamic,
                gossip_rounds=gossip_rounds, prefix="extractor."
            )
            for i in range(self.num_clients):
                self.clients_state[i].update(new_states[i])

            self.evaluate()
            print(f"Avg Loss: {self.loss[-1]:.4f}, Acc: {self.acc[-1]:.2f}%")
            print(f"Round finished in {time.time() - t0:.2f} seconds")

    def save(self):
        f = {"acc": self.acc, "loss": self.loss, "state_dict": self.clients_state, "consensus_P": self.consensus_P}
        self.deal_save(f)
