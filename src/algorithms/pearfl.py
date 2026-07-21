import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .utils import (
    BaseServer,
    fmt_num,
    ce_loss,
    extract_prototypes,
    flattened_matrix_aggregate,
    generate_adjacency_matrix,
    get_model,
    mse_loss,
    sinkhorn_knopp,
)


def get_path(args):
    """生成日志文件路径，包含拓扑参数"""
    adj_suffix = f"{args.adj_type}"
    if args.adj_type == "random":
        adj_suffix += f"_{fmt_num(args.edge_p)}"
    elif args.adj_type == "small_world":
        adj_suffix += f"_{fmt_num(args.k_small_world)}_{fmt_num(args.edge_p)}"
    elif args.adj_type == "scale_free":
        adj_suffix += f"_{fmt_num(args.m_scale_free)}"

    # 将算法的关键超参加入文件名，便于区分实验
    args.file_name = f"{args.common_name}_{adj_suffix}_{fmt_num(args.lamda)}"
    return os.path.join(args.log_path, f"{args.file_name}_{args.cur_time}.log")


def train(params):
    """
    PearFL 客户端工作函数：本地训练包含原型对齐损失，返回模型参数和本地原型
    """
    (
        _,
        device,
        model_state,
        train_set,
        model_name,
        dataset_name,
        num_classes,
        feature_dim,
        batch_size,
        lr,
        momentum,
        weight_decay,
        lamda,
        personalized_protos,
    ) = params

    # 1. 初始化模型并加载参数
    model = get_model(model_name, dataset_name, feature_dim).to(device)
    model.load_state_dict(model_state)
    model.train()

    # 2. 将个性化共识原型转移到设备
    if personalized_protos is not None:
        personalized_protos = personalized_protos.to(device)

    # 3. 准备数据加载器
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    # 4. 初始化优化器
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    # 5. 本地训练：仅执行 1 个 Epoch（由服务端控制多跳逻辑）
    total_loss = 0.0
    correct = 0
    total = 0
    num_batches = 0

    # 强制执行 1 个 epoch 以适配 Algorithm 3 的 Inter-Epoch 交换
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        # 模型前向传播
        features = model.extractor(x)
        logits = model.classifier(features)

        # 交叉熵损失
        l_ce = ce_loss(logits, y)

        # 原型正则化损失
        l_reg = torch.tensor(0.0, device=device)
        if personalized_protos is not None and personalized_protos.abs().sum() > 0:
            target_protos = personalized_protos[y]
            l_reg = mse_loss(features, target_protos)

        # 总损失
        loss = l_ce + lamda * l_reg
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(logits.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()
        num_batches += 1

    # 6. 提取本地经验原型及其样本统计
    local_protos, local_counts = extract_prototypes(
        model,
        loader,
        num_classes,
        feature_dim,
        device,
        return_counts=True,
    )

    return {
        "loss": total_loss / num_batches,  # avg_loss
        "state": {
            k: v.cpu().detach().clone() for k, v in model.state_dict().items()
        },  # model_state
        "protos": local_protos,  # local_protos [num_classes, feature_dim]
        "counts": local_counts,  # local_counts [num_classes]
    }


class Server(BaseServer):
    """
    PearFL Server：通过 Sinkhorn-Knopp 生成的双随机矩阵进行原型的去中心化聚合

    关键特性：
    - 使用双随机矩阵 W 进行分布式通信
    - 在每个客户端维护个性化共识原型 (personalized_protos)
    - 聚合包括原型加权平均和模型参数平均
    """

    def __init__(self, args):
        super().__init__(pfl=True, args=args)

        # 1. 通信矩阵初始化（基于 Sinkhorn-Knopp 的双随机矩阵）
        A = generate_adjacency_matrix(args)
        self.W = sinkhorn_knopp(A).to(self.device)

        # 2. 状态池初始化
        # local_protos_pool: 存储各节点的原始本地原型 [num_clients, num_classes, feature_dim]
        # local_counts_pool: 存储各节点的样本数量统计 [num_clients, num_classes]
        # personalized_protos: 存储各节点 Gossip 聚合后的最终共识原型
        self.local_protos_pool = torch.zeros(
            self.num_clients, self.num_class, args.feature_dim
        ).to(self.device)
        self.local_counts_pool = torch.zeros(self.num_clients, self.num_class).to(
            self.device
        )
        self.personalized_protos = torch.zeros(
            self.num_clients, self.num_class, args.feature_dim
        ).to(self.device)

        # 3. 初始化 clients_state 为当前全局模型
        self.clients_state = [self.model.state_dict() for _ in range(self.num_clients)]

    def fit(self):
        """主训练循环：实现 Algorithm 3 的 Inter-Epoch Prototype Exchange"""
        num_join_clients = int(self.num_clients * self.args.join_ratio)
        num_join_clients = max(1, num_join_clients)

        for r in range(self.rounds):
            t0 = time.time()
            print(f"\n--- PearFL Round {r + 1}/{self.rounds} ---")

            # 1. 随机选择参与的客户端
            selected_clients = np.random.choice(
                self.num_clients, num_join_clients, replace=False
            )
            print(f"Selected clients: {selected_clients}")

            # 2. 嵌套循环：执行 E 个本地 Epoch，并在每个 Epoch 结束后交换原型
            round_loss = 0.0
            for e in range(self.args.epochs):
                # 2.1 为每个选中的客户端准备参数
                def get_client_param(i):
                    return [
                        i,
                        self.client_gpu[i],
                        self.clients_state[i],
                        self.train_sets[i],
                        self.args.model,
                        self.args.dataset,
                        self.num_class,
                        self.args.feature_dim,
                        self.args.batch_size,
                        self.args.lr,
                        self.args.momentum,
                        self.args.weight_decay,
                        self.args.lamda,
                        self.personalized_protos[i].cpu(),
                    ]

                params = [get_client_param(i) for i in selected_clients]

                # 2.2 启动 Ray 并行训练 (1 Epoch)
                results = self.run_clients(train, params)

                # 2.3 回收结果：更新模型状态、原型和样本计数
                epoch_loss = 0.0
                for cid, res in results.items():
                    epoch_loss += res["loss"]
                    self.clients_state[cid] = res["state"]
                    self.local_protos_pool[cid] = res["protos"].to(self.device)
                    self.local_counts_pool[cid] = res["counts"].to(self.device)

                epoch_loss /= len(selected_clients)
                if (
                    e == self.args.epochs - 1
                ):  # 记录最后一个 epoch 的 loss 作为 round loss
                    round_loss = epoch_loss

                # 2.4 执行原型交换与聚合 (Algorithm 2)
                self.aggregate_prototypes()

            self.loss.append(round_loss)

            # 3. 执行模型聚合（可选，论文主要强调原型，但 FL 框架通常保留模型同步）
            self.aggregate_models()

            # 4. 评估模型
            self.evaluate()

            print(f"Avg Loss: {self.loss[-1]:.4f}, Acc: {self.acc[-1]:.2f}%")
            print(f"Round finished in {time.time() - t0:.2f} seconds")

    def aggregate_prototypes(self):
        """
        全向量化实现 Algorithm 2：基于样本量权重的分布式原型聚合
        根据用户反馈：原型聚合仅使用对应类的样本数，不使用 W 的权重。
        """
        # 1. 构造组合权重：mask(邻居关系) * count(样本量)
        # mask[i,j] 表示 j 是否为 i 的邻居
        mask = (self.W > 0).float()
        combine_weight = mask.unsqueeze(-1) * self.local_counts_pool.unsqueeze(0)

        # 2. 计算归一化分母（每个类别在每个节点的邻域内的总样本数）
        denom = combine_weight.sum(dim=1, keepdim=True)
        denom_safe = torch.where(denom > 0, denom, torch.ones_like(denom))

        # 3. 执行加权聚合 (Weighted Average)
        new_protos = torch.einsum(
            "ijc,jcd->icd",
            combine_weight / denom_safe,
            self.local_protos_pool,
        )

        self.personalized_protos = new_protos

    def aggregate_models(self):
        """
        模型参数聚合（GPU 矩阵化版本）：S' = W @ flatten(S)

        W 为双随机矩阵，W[i,j] = 0 表示非邻居，等价于邻居加权平均。
        """
        state_list = list(self.clients_state)
        new_state_list = flattened_matrix_aggregate(state_list, self.W, self.device)
        self.clients_state = new_state_list

        # 更新全局 model 供 evaluate() 全局统计使用
        avg_state = self.weighted_aggregate(self.clients_state, self.weights)
        self.model.load_state_dict(avg_state)

    def weighted_aggregate(self, states, weights):
        """
        加权聚合多个客户端的模型参数

        Args:
            states: 客户端模型参数列表
            weights: 对应权重列表

        Returns:
            聚合后的模型参数字典
        """
        aggregated_state = {}
        total_weight = sum(weights)

        for key in states[0].keys():
            aggregated_state[key] = torch.zeros_like(states[0][key])
            for state, weight in zip(states, weights):
                aggregated_state[key] += state[key] * (weight / total_weight)

        return aggregated_state

    def save(self):
        metrics = {"acc": self.acc, "loss": self.loss}
        params = {
            "client": self.clients_state,
            "aux": {
                "personalized_protos": self.personalized_protos.cpu(),
                "local_protos_pool": self.local_protos_pool.cpu(),
                "local_counts_pool": self.local_counts_pool.cpu(),
                "W": self.W.cpu(),
            },
        }
        self.deal_save(metrics, params)
