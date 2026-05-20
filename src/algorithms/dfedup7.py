import os
import time

import numpy as np
import torch
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
    DFedUP7 Worker: 联合训练 + S/W 原型提取，支持样本计数返回。
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
        self.consensus_P = [torch.zeros(self.num_class, args.feature_dim) for _ in range(self.num_clients)]
        self.counts_cache = [torch.zeros(self.num_class) for _ in range(self.num_clients)]

        # Extractor Push-Sum 权重
        self.W_params = torch.ones(self.num_clients, 1)

    def fit(self):
        num_join_clients = int(self.num_clients * self.args.join_ratio)
        num_join_clients = max(1, num_join_clients)

        for r in range(self.rounds):
            t0 = time.time()
            print(f"\n--- DFedUP7 Round {r + 1}/{self.rounds} ---")
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
                
                # 虚拟历史权重锚定法核心：如果某类别在本轮本地训练中没有样本计数，赋予小权重 w_virtual，并以上一轮的共识原型为锚点
                counts = res["counts"]
                S_local = res["S"]
                W_local = res["W"]
                for c in range(self.num_class):
                    if counts[c] > 0:
                        self.S_cache[cid][c] = S_local[c]
                        self.W_cache[cid][c] = W_local[c]
                    else:
                        w_virtual = 0.1
                        self.W_cache[cid][c] = torch.tensor([w_virtual])
                        self.S_cache[cid][c] = w_virtual * self.consensus_P[cid][c]
                self.counts_cache[cid] = counts
            self.loss.append(total_loss / num_join_clients)

            gossip_rounds = getattr(self.args, "gossip_rounds", 1)

            # 1. 原型 (S, W): 普通平均聚合 (using M_avg)
            S = torch.stack([self.S_cache[i] for i in range(self.num_clients)]).to(self.device)
            W = torch.stack([self.W_cache[i] for i in range(self.num_clients)]).to(self.device)
            S_flat, W_flat = S.view(self.num_clients, -1), W.view(self.num_clients, -1)

            for _ in range(gossip_rounds):
                S_flat, W_flat = torch.mm(self.M_avg, S_flat), torch.mm(self.M_avg, W_flat)

            W_tensor = W_flat.view(self.num_clients, self.num_class, 1)
            S_tensor = S_flat.view(self.num_clients, self.num_class, self.args.feature_dim)
            consensus = S_tensor / (W_tensor + 1e-12)

            for i in range(self.num_clients):
                self.S_cache[i], self.W_cache[i], self.consensus_P[i] = S_tensor[i].cpu(), W_tensor[i].cpu(), consensus[i].cpu()

            # 2. Extractor: Push-Sum 聚合 (using M_ps)
            new_states, self.W_params = pushsum_param_aggregate(
                self.clients_state, self.W_params, self.M_ps,
                gossip_rounds=gossip_rounds, prefix="extractor."
            )
            for i in range(self.num_clients):
                self.clients_state[i].update(new_states[i])

            self.evaluate()
            print(f"Avg Loss: {self.loss[-1]:.4f}, Acc: {self.acc[-1]:.2f}%")

    def save(self):
        f = {"acc": self.acc, "loss": self.loss, "state_dict": self.clients_state, "consensus_P": self.consensus_P}
        self.deal_save(f)
