import os
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from .utils import (
    BaseParams,
    BaseServer,
    clone_cpu_state,
    compute_mh_weights,
    flattened_matrix_aggregate,
    fmt_num,
    generate_adjacency_matrix,
    get_model,
)


def get_path(args):
    adj_suffix = f"{args.adj_type}"
    if args.adj_type == "random":
        adj_suffix += f"_{fmt_num(args.edge_p)}"
    elif args.adj_type == "small_world":
        adj_suffix += f"_{fmt_num(args.k_small_world)}_{fmt_num(args.edge_p)}"
    elif args.adj_type == "scale_free":
        adj_suffix += f"_{fmt_num(args.m_scale_free)}"

    args.file_name = f"{args.common_name}_{adj_suffix}"
    return os.path.join(args.log_path, f"{args.file_name}_{args.cur_time}.log")


@dataclass
class Params(BaseParams):
    optimizer_state: dict | None


def train(p: Params):
    """
    标准的 FedAvg 本地训练流程，支持动量。
    """
    device = torch.device(p.client_gpu)

    # 1. 初始化模型并加载最新的全局模型参数
    model = get_model(p.model_name, p.dataset, p.num_class, p.feature_dim).to(device)
    model.load_state_dict(p.model_state)

    # 2. 设置优化器与数据加载器
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=p.lr,
        momentum=p.momentum,
        weight_decay=p.weight_decay,
    )
    if p.optimizer_state is not None:
        optimizer.load_state_dict(p.optimizer_state)

    loader = torch.utils.data.DataLoader(
        p.train_set, batch_size=p.batch_size, shuffle=True
    )

    # 3. 本地模型多轮次 (Epochs) 训练
    total_loss = 0.0
    num_batches = 0
    model.train()
    for _ in range(p.epochs):
        for x, y, *_ in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
    avg_loss = total_loss / num_batches

    # 4. 整理返回结果
    model_state = clone_cpu_state(model.state_dict())
    return {
        "loss": avg_loss,
        "state": model_state,
        "opt_state": None,
    }


class Server(BaseServer):
    # DFedAvgM Server: 使用 Metropolis-Hastings (MH) 权重矩阵进行去中心化模型聚合

    def __init__(self, args):
        super().__init__(args, pfl=True)

        # 使用通用的邻接矩阵生成函数
        self.adj_matrix = generate_adjacency_matrix(args)

        # 计算 Metropolis-Hastings (MH) 混合权重矩阵
        self.mh_weights = compute_mh_weights(self.adj_matrix, device=self.device)

        # 初始化各客户端的优化器状态 (动量)
        self.opt_states = [None for _ in range(self.num_clients)]

    def aggregate_mh(self):
        """执行分布式聚合（GPU 矩阵化版本）：S' = MH_weights @ S"""
        state_list = list(self.clients_state)
        new_state_list = flattened_matrix_aggregate(
            state_list, self.mh_weights, self.device
        )
        self.clients_state = new_state_list

    def fit(self):
        num_join = max(1, int(self.num_clients * self.join_ratio))

        for r in range(self.rounds):
            t0 = time.time()
            print(f"\n--- DFedAvgM Round {r + 1}/{self.rounds} ---")

            selected = torch.randperm(self.num_clients)[:num_join].tolist()
            print(f"Selected clients: {selected}")

            # 1. 为每个选中的客户端准备参数
            base_params = self.build_base_params(selected)
            for base in base_params:
                base.model_state = self.clients_state[base.client_id]

            p = [
                Params(
                    **asdict(base),
                    optimizer_state=self.opt_states[base.client_id],
                )
                for base in base_params
            ]
            # 2. 启动客户端多进程并行训练
            results = self.run_clients(train, p)

            # 3. 收集客户端训练后的状态和损失
            total_loss = 0.0
            for cid, res in results.items():
                total_loss += res["loss"]
                self.clients_state[cid] = res["state"]
                self.opt_states[cid] = res["opt_state"]  # 保存最新动量
            self.loss.append(total_loss / num_join)

            # 4. 执行分布式聚合
            self.aggregate_mh()

            # 5. 执行评估
            self.evaluate()

            print(f"Avg Loss: {self.loss[-1]:.4f}, Acc: {self.acc[-1]:.2f}%")
            print(f"Round finished in {time.time() - t0:.2f} seconds")

    def save(self):
        metrics = {"acc": self.acc, "loss": self.loss}
        params = {"client": self.clients_state}
        self.deal_save(metrics, params)
