import os
import time

import numpy as np
import torch

from .utils import (
    BaseServer,
    ce_loss,
    compute_mh_weights,
    generate_adjacency_matrix,
    get_model,
    param_aggregate,
)


def get_path(args):
    adj_suffix = f"{args.adj_type}"
    if args.adj_type == "random":
        adj_suffix += f"_{args.edge_p}"
    elif args.adj_type == "small_world":
        adj_suffix += f"_{args.k_small_world}_{args.edge_p}"
    elif args.adj_type == "scale_free":
        adj_suffix += f"_{args.m_scale_free}"

    args.file_name = f"{args.name_pre}_{adj_suffix}"
    return os.path.join(args.log_path, f"{args.file_name}_{args.times}.log")


def client_worker(params):
    """
    标准的 FedAvg 本地训练流程。
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
    ) = params

    # 1. 初始化模型并加载最新的全局模型参数
    model = get_model(model_name, dataset_name, feature_dim).to(device)
    model.load_state_dict(model_state)

    # 2. 设置优化器与数据加载器
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    # 3. 本地模型多轮次 (Epochs) 训练
    total_loss = 0.0
    num_batches = 0
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = ce_loss(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
    avg_loss = total_loss / num_batches

    # 4. 整理返回结果（将模型状态移至 CPU 以节省 GPU 显存容量消耗）
    model_state = {k: v.cpu().detach().clone() for k, v in model.state_dict().items()}
    return [avg_loss, model_state]


class Server(BaseServer):
    # DFedAvgM Server: 使用 Metropolis-Hastings (MH) 权重矩阵进行去中心化模型聚合

    def __init__(self, args):
        super().__init__(pfl=True, args=args)

        # 使用通用的邻接矩阵生成函数
        self.adj_matrix = generate_adjacency_matrix(args)

        # 计算 Metropolis-Hastings (MH) 混合权重矩阵
        self.mh_weights = compute_mh_weights(self.adj_matrix, device=self.device)

    def aggregate_mh(self):
        """执行分布式聚合：根据 MH 权重矩阵聚合所有客户端的参数

        每个客户端 i 根据 self.mh_weights[i] 的权重，聚合所有客户端（包括自己）的参数。
        """
        new_client_states = {}

        for i in range(self.num_clients):
            # 获取所有客户端的状态
            all_states = [self.clients_state[j] for j in range(self.num_clients)]
            # 获取客户端 i 的权重向量
            weights = self.mh_weights[i].tolist()

            # 使用 param_aggregate 进行聚合
            new_client_states[i] = param_aggregate(all_states, weights)

        self.clients_state = new_client_states

    def fit(self):
        num_join_clients = int(self.num_clients * self.args.join_ratio)
        num_join_clients = max(1, num_join_clients)

        for r in range(self.rounds):
            t0 = time.time()
            print(f"\n--- DFedAvgM Round {r + 1}/{self.rounds} ---")

            selected_clients = np.random.choice(
                self.num_clients, num_join_clients, replace=False
            )
            print(f"Selected clients: {selected_clients}")

            # 1. 为每个选中的客户端准备参数（使用当前本地模型进行训练）
            def get_client_param(i):
                return [
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
                ]

            p = [get_client_param(i) for i in selected_clients]
            # 2. 启动客户端多进程并行训练
            results = self.run_clients(client_worker, p)

            # 3. 收集客户端训练后的状态和损失
            total_loss = 0.0
            for i in selected_clients:
                avg_loss, client_state = results[i]
                total_loss += avg_loss
                self.clients_state[i] = client_state
            self.loss.append(total_loss / num_join_clients)

            # 4. 执行分布式聚合：根据 MH 权重矩阵聚合邻居参数
            self.aggregate_mh()

            # 5. 执行评估
            self.evaluate()

            print(f"Avg Loss: {self.loss[-1]:.4f}, Acc: {self.acc[-1]:.2f}%")
            print(f"Round finished in {time.time() - t0:.2f} seconds")

    def save(self):
        f = {
            "acc": self.acc,
            "loss": self.loss,
            "state_dict": self.clients_state,
        }
        self.deal_save(f)
