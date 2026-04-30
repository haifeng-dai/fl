import argparse
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .utils import (
    BaseServer,
    ce_loss,
    extract_prototypes,
    generate_adjacency_matrix,
    get_model,
    mse_loss,
    sinkhorn_knopp,
)


def get_path(args):
    """生成日志文件路径，包含拓扑参数"""
    adj_suffix = f"{args.adj_type}"
    if args.adj_type == "random":
        adj_suffix += f"_{args.edge_p}"
    elif args.adj_type == "small_world":
        adj_suffix += f"_{args.k}_{args.edge_p}"
    elif args.adj_type == "scale_free":
        adj_suffix += f"_{args.m}"

    # 将算法的关键超参加入文件名，便于区分实验
    args.file_name = f"{args.name_pre}_{adj_suffix}_{args.epochs}_{args.lamda}"
    return os.path.join(args.log_path, f"{args.file_name}_{args.times}.log")


def add_args(parser: argparse.ArgumentParser):
    """添加 PearFL 特定的命令行参数"""
    group = parser.add_argument_group("PearFL Specific Arguments")

    # 拓扑参数
    group.add_argument(
        "--adj_type",
        type=str,
        default="ring",
        choices=["ring", "complete", "random", "small_world", "scale_free", "star"],
        help="Topology of the decentralized network",
    )

    # 可选的拓扑参数
    group.add_argument(
        "--edge_p",
        type=float,
        default=0.3,
        help="Edge probability for random / small_world topologies (default: 0.3)",
    )
    group.add_argument(
        "--k",
        type=int,
        default=4,
        help="Neighborhood size k for small_world topology (default: 4)",
    )
    group.add_argument(
        "--m",
        type=int,
        default=2,
        help="Attachment parameter m for scale_free topology (default: 2)",
    )

    # PearFL 特定参数：原型正则化权重
    group.add_argument(
        "--lamda",
        type=float,
        default=1.0,
        help="Weight for prototype regularization loss (default: 1.0)",
    )

    # Sinkhorn-Knopp 参数
    group.add_argument(
        "--epsilon",
        type=float,
        default=1e-3,
        help="Tolerance for Sinkhorn-Knopp algorithm (default: 1e-3)",
    )

    # Global-on-Local 评估开关
    group.add_argument(
        "--do_global_on_local_eval",
        type=bool,
        default=False,
        help="Whether to perform global-on-local evaluation (default: False)",
    )

    return parser


def client_worker(params):
    """
    PearFL 客户端工作函数：本地训练包含原型对齐损失，返回模型参数和本地原型

    参数结构：
        params: [
            client_id,
            device,
            model_state (dict),                # 初始模型参数
            train_set,
            model_name,
            dataset_name,
            test_set,
            num_classes,
            feature_dim,
            batch_size,
            local_epochs,
            lr,
            momentum,
            weight_decay,
            lamda,                            # 原型正则化权重
            personalized_protos (torch.Tensor or None),  # 共识原型
        ]
    """
    (
        client_id,
        device,
        model_state,
        train_set,
        model_name,
        dataset_name,
        test_set,
        num_classes,
        feature_dim,
        batch_size,
        local_epochs,
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

    # 5. 本地训练循环
    total_loss = 0.0
    correct = 0
    total = 0
    num_batches = 0

    for _ in range(local_epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            # 模型前向传播：分别提取特征和 logits
            features = model.extractor(x)
            logits = model.classifier(features)

            # 计算交叉熵损失
            l_ce = ce_loss(logits, y)

            # 计算原型正则化损失：迫使特征靠近聚合后的个性化原型
            l_reg = torch.tensor(0.0, device=device)
            if personalized_protos is not None and personalized_protos.abs().sum() > 0:
                target_protos = personalized_protos[y]
                l_reg = mse_loss(features, target_protos)

            # 总损失 = 交叉熵 + lamda * 原型损失
            loss = l_ce + lamda * l_reg
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            num_batches += 1

    # 6. 提取本地经验原型及其样本统计
    # 使用 Subset 确保兼容性
    local_protos, local_counts = extract_prototypes(
        model,
        loader,
        num_classes,
        feature_dim,
        device,
        return_counts=True,
    )

    # 将本地原型转换为张量格式 [num_classes, feature_dim]
    protos_tensor = torch.zeros((num_classes, feature_dim), device=device)
    counts_tensor = torch.zeros(num_classes, device=device)

    for class_id, proto in local_protos.items():
        protos_tensor[class_id] = proto.to(device)

    for class_id, count in local_counts.items():
        counts_tensor[class_id] = float(count)

    return [
        total_loss / num_batches,  # avg_loss
        {k: v.cpu().detach().clone() for k, v in model.state_dict().items()},  # model_state
        protos_tensor.cpu(),  # local_protos [num_classes, feature_dim]
        counts_tensor.cpu(),  # local_counts [num_classes]
    ]


class Server(BaseServer):
    """
    PearFL Server：通过 Sinkhorn-Knopp 生成的双随机矩阵进行原型的去中心化聚合

    关键特性：
    - 使用双随机矩阵 W 进行分布式通信
    - 在每个客户端维护个性化共识原型 (personalized_protos)
    - 聚合包括原型加权平均和模型参数平均
    """

    def __init__(self, args: argparse.Namespace):
        super().__init__(pfl=True, args=args)

        # 1. 通信矩阵初始化（基于 Sinkhorn-Knopp 的双随机矩阵）
        A = generate_adjacency_matrix(args)
        self.W = sinkhorn_knopp(A, args.epsilon).to(self.device)

        # 2. 状态池初始化
        # local_protos_pool: 存储各节点的原始本地原型 [num_clients, num_classes, feature_dim]
        # local_counts_pool: 存储各节点的样本数量统计 [num_clients, num_classes]
        # personalized_protos: 存储各节点 Gossip 聚合后的最终共识原型
        self.local_protos_pool = torch.zeros(
            self.num_clients, self.num_class, args.feature_dim
        ).to(self.device)
        self.local_counts_pool = torch.zeros(
            self.num_clients, self.num_class
        ).to(self.device)
        self.personalized_protos = torch.zeros(
            self.num_clients, self.num_class, args.feature_dim
        ).to(self.device)

        # 3. 初始化 clients_state 为当前全局模型
        self.clients_state = [
            self.model.state_dict() for _ in range(self.num_clients)
        ]

    def fit(self):
        """主训练循环"""
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

            # 2. 为每个选中的客户端准备参数
            def get_client_param(i):
                return [
                    i,
                    self.client_gpu[i],
                    self.clients_state[i],
                    self.train_sets[i],
                    self.args.model,
                    self.args.dataset,
                    self.test_set[i],
                    self.num_class,
                    self.args.feature_dim,
                    self.args.batch_size,
                    self.args.epochs,
                    self.args.lr,
                    self.args.momentum,
                    self.args.weight_decay,
                    self.args.lamda,
                    self.personalized_protos[i].cpu(),
                ]

            params = [get_client_param(i) for i in selected_clients]

            # 3. 启动客户端并行训练
            results = self.run_clients(client_worker, params)

            # 4. 收集客户端的更新
            total_loss = 0.0
            for client_idx in selected_clients:
                avg_loss, model_state, protos, counts = results[client_idx]
                total_loss += avg_loss
                self.clients_state[client_idx] = model_state
                self.local_protos_pool[client_idx] = protos.to(self.device)
                self.local_counts_pool[client_idx] = counts.to(self.device)

            self.loss.append(total_loss / len(selected_clients))

            # 5. 执行去中心化聚合
            self.aggregate()

            # 6. 评估模型
            self.evaluate()

            print(f"Avg Loss: {self.loss[-1]:.4f}, Acc: {self.acc[-1]:.2f}%")
            print(f"Round finished in {time.time() - t0:.2f} seconds")

    def aggregate(self):
        """
        全向量化实现原型聚合（Algorithm 2）：基于双随机权重的分布式原型聚合

        聚合步骤：
        1. 采集本轮活跃节点的本地原型和计数
        2. 构造组合权重：combine_weight[i,k,c] = W[i,k] * count[k,c]
        3. 执行加权聚合：new_proto[i,c] = sum_k(combine_weight[i,k,c] * proto[k,c]) / sum_k(combine_weight[i,k,c])
        """
        # 1. 构造组合权重（按照 Algorithm 2）
        # W: [num_clients, num_clients] - 双随机矩阵
        # local_counts_pool: [num_clients, num_classes] - 各客户端各类别的样本数
        # combine_weight[i,j,c] = W[i,j] * count[j,c]
        combine_weight = (
            self.W.unsqueeze(-1) * self.local_counts_pool.unsqueeze(0)
        )  # [num_clients, num_clients, num_classes]

        # 2. 计算归一化分母
        denom = combine_weight.sum(dim=1, keepdim=True)  # [num_clients, 1, num_classes]
        denom_safe = torch.where(
            denom > 0, denom, torch.ones_like(denom)
        )

        # 3. 矩阵运算实现加权聚合
        # new_proto[i,c,d] = sum_j(combine_weight[i,j,c] * proto[j,c,d]) / sum_j(combine_weight[i,j,c])
        # 使用 einsum：(combine_weight / denom) @ protos
        new_protos = torch.einsum(
            "ijc,jcd->icd",
            combine_weight / denom_safe,
            self.local_protos_pool,
        )  # [num_clients, num_classes, feature_dim]

        self.personalized_protos = new_protos

        # 4. 更新虚拟全局模型供日志记录
        states = [self.clients_state[cid] for cid in range(self.num_clients)]
        avg_state = self.weighted_aggregate(states, self.weights)
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
        """保存模型和相关状态"""
        f = {
            "acc": self.acc,
            "loss": self.loss,
            "state_dict": {
                "client": self.clients_state,
                "personalized_protos": self.personalized_protos.cpu(),
                "local_protos_pool": self.local_protos_pool.cpu(),
                "local_counts_pool": self.local_counts_pool.cpu(),
                "W": self.W.cpu(),
            },
        }
        self.deal_save(f)
