import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .utils import (
    BaseServer,
    ce_loss,
    generate_adjacency_matrix,
    get_model,
    sinkhorn_knopp,
)


def _generate_etf(num_classes, feature_dim):
    """
    构造单形等角紧框架 (Simplex ETF)。
    使用固定种子保证所有客户端生成的矩阵完全一致。
    """
    gen = torch.Generator()
    gen.manual_seed(42)

    eye = torch.eye(num_classes)
    one = torch.ones(num_classes, num_classes)
    # 构造 K 维空间中的 Simplex ETF: sqrt(K/(K-1)) * (I - 1/K * 11^T)
    etf = torch.sqrt(torch.tensor(num_classes / (num_classes - 1.0))) * (
        eye - 1.0 / num_classes * one
    )

    if feature_dim >= num_classes:
        # 使用随机正交矩阵将 K 维 ETF 映射到 feature_dim 维空间
        P = torch.empty(feature_dim, feature_dim)
        torch.nn.init.orthogonal_(P, generator=gen)
        # W = etf @ P[:K, :] -> [K, K] @ [K, d] = [K, d]
        W = torch.matmul(etf, P[:num_classes, :])
    else:
        # 如果维度不足，则截断
        W = etf[:, :feature_dim]
    return W


def _apply_etf_classifier(model, num_classes, feature_dim):
    """
    将模型的分类器替换为固定的 ETF，并冻结参数。
    """
    weight = _generate_etf(num_classes, feature_dim)
    with torch.no_grad():
        model.classifier.weight.copy_(weight)
        if model.classifier.bias is not None:
            model.classifier.bias.zero_()

    # 强制冻结参数
    model.classifier.weight.requires_grad = False
    if model.classifier.bias is not None:
        model.classifier.bias.requires_grad = False


def get_path(args):
    """生成日志文件路径，包含拓扑参数"""
    adj_suffix = f"{args.adj_type}"
    if args.adj_type == "random":
        adj_suffix += f"_{args.edge_p}"
    elif args.adj_type == "small_world":
        adj_suffix += f"_{args.k_small_world}_{args.edge_p}"
    elif args.adj_type == "scale_free":
        adj_suffix += f"_{args.m_scale_free}"

    # 将算法的关键超参加入文件名，便于区分实验
    args.file_name = f"{args.common_name}_{adj_suffix}_{args.mu}"
    return os.path.join(args.log_path, f"{args.file_name}_{args.cur_time}.log")


def client_worker(params):
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
        local_epochs,
        lr,
        momentum,
        weight_decay,
    ) = params

    # 1. 初始化模型并加载参数
    model = get_model(model_name, dataset_name, feature_dim).to(device)
    _apply_etf_classifier(model, num_classes, feature_dim)
    model.load_state_dict(model_state)

    # 再次确保分类器被冻结（load_state_dict 可能会覆盖 requires_grad 标志）
    model.classifier.weight.requires_grad = False
    if model.classifier.bias is not None:
        model.classifier.bias.requires_grad = False

    model.train()

    # 2. 准备数据加载器
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    # 4. 初始化优化器：仅优化需要梯度的参数（排除 ETF 分类器）
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
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
            loss = ce_loss(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            num_batches += 1


    return {
        "loss": total_loss / num_batches,  # avg_loss
        "state": {
            k: v.cpu().detach().clone() for k, v in model.state_dict().items()
        },  # model_state
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

        # 1.1 应用 ETF 分类器到全局模型骨架
        _apply_etf_classifier(self.model, self.num_class, args.feature_dim)

        # 2. 初始化 clients_state 为当前全局模型
        self.clients_state = [self.model.state_dict() for _ in range(self.num_clients)]

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
                    self.num_class,
                    self.args.feature_dim,
                    self.args.batch_size,
                    self.args.epochs,
                    self.args.lr,
                    self.args.momentum,
                    self.args.weight_decay,
                ]

            params = [get_client_param(i) for i in selected_clients]

            # 3. 启动客户端并行训练
            results = self.run_clients(client_worker, params)

            # 4. 收集客户端的更新
            total_loss = 0.0
            for cid, res in results.items():
                total_loss += res["loss"]
                # 缓存每个客户端更新后的个性化模型
                self.clients_state[cid] = res["state"]

            self.loss.append(total_loss / len(selected_clients))

            # 5. 执行去中心化聚合
            self.aggregate()

            # 6. 评估模型
            self.evaluate()

            print(f"Avg Loss: {self.loss[-1]:.4f}, Acc: {self.acc[-1]:.2f}%")
            print(f"Round finished in {time.time() - t0:.2f} seconds")

    def aggregate(self):
        """
        模型聚合（当前为全局平均）
        """
        # 更新虚拟全局模型供日志记录
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
                "W": self.W.cpu(),
            },
        }
        self.deal_save(f)
