import logging
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from .utils import (
    BaseServer,
    ce_loss,
    generate_adjacency_matrix,
    get_model,
)

logger = logging.getLogger(__name__)


def get_path(args):
    """生成日志文件路径，包含拓扑参数和元学习超参数"""
    adj_suffix = f"{args.adj_type}"
    if args.adj_type == "random":
        adj_suffix += f"_{args.edge_p}"
    elif args.adj_type == "small_world":
        adj_suffix += f"_{args.k_small_world}_{args.edge_p}"
    elif args.adj_type == "scale_free":
        adj_suffix += f"_{args.m_scale_free}"

    # 将算法的关键超参加入文件名
    args.file_name = (
        f"{args.name_pre}_{adj_suffix}_{args.epochs}"
        f"_{args.val_ratio}_{args.lr_alpha}_{args.threshold}"
    )
    return os.path.join(args.log_path, f"{args.file_name}_{args.cur_time}.log")


def client_worker_phase1(params):
    """
    L2C 客户端第一阶段：本地训练并计算参数增量 Delta Theta

    参数结构：
        params: [
            client_id,
            device,
            model_state (dict),
            train_set,
            model_name,
            dataset_name,
            test_set,
            num_classes,
            feature_dim,
            batch_size,
            local_epochs,
            lr,
            val_ratio,
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
        val_ratio,
    ) = params

    # 1. 初始化模型并加载参数
    model = get_model(model_name, dataset_name, feature_dim).to(device)
    model.load_state_dict(model_state)

    # 保存初始状态用于计算 Delta
    theta_t = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # 2. 准备数据划分（分训练集和验证集）
    n_samples = len(train_set)
    n_val = int(n_samples * val_ratio)
    indices = list(range(n_samples))
    np.random.shuffle(indices)

    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    # 3. 执行本地训练
    train_subset = Subset(train_set, train_indices)
    loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    num_batches = 0

    for _ in range(local_epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = ce_loss(output, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            num_batches += 1

    # 4. 计算 Delta = theta_t - theta_updated
    theta_mid = model.state_dict()
    delta_theta = {
        k: (theta_t[k].to(device) - theta_mid[k]).cpu() for k in theta_t.keys()
    }

    return [
        total_loss / num_batches,  # avg_loss
        theta_t,  # 返回初始参数供第二阶段使用
        {k: v.detach().clone() for k, v in delta_theta.items()},  # delta_theta
        train_indices,  # 用于第二阶段的验证集
        val_indices,
    ]


def client_worker_phase2(params):
    """
    L2C 客户端第二阶段：元学习更新 alpha 并执行最终加权聚合

    参数结构：
        params: [
            client_id,
            device,
            model_state (dict),
            theta_t (dict),
            train_set,
            model_name,
            dataset_name,
            test_set,
            num_classes,
            feature_dim,
            batch_size,
            local_epochs,
            lr,
            neighbors (list),
            neighbor_deltas (list),
            alpha (tensor),
            val_indices (list),
            lr_alpha,
            threshold,
        ]
    """
    (
        client_id,
        device,
        model_state,
        theta_t,
        train_set,
        model_name,
        dataset_name,
        test_set,
        num_classes,
        feature_dim,
        batch_size,
        local_epochs,
        lr,
        neighbors,
        neighbor_deltas,
        alpha,
        val_indices,
        lr_alpha,
        threshold,
    ) = params

    # 1. 初始化模型
    model = get_model(model_name, dataset_name, feature_dim).to(device)
    model.load_state_dict(model_state)

    # 2. 准备验证集
    alpha = alpha.to(device).detach().requires_grad_(True)

    # 3. 计算混合权重 w = softmax(alpha)
    w = F.softmax(alpha, dim=0)

    # 4. 虚拟聚合：theta_agg = theta^t - sum(w_j * delta_j)
    theta_agg = {k: theta_t[k].to(device).clone() for k in theta_t.keys()}
    for k in theta_agg.keys():
        # 堆叠所有邻居的增量
        layer_deltas = torch.stack([d[k].to(device) for d in neighbor_deltas])
        # w: [num_neighbors] -> reshape for broadcasting
        dims = [1] * (layer_deltas.dim() - 1)
        theta_agg[k] -= torch.sum(layer_deltas * w.view(-1, *dims), dim=0)

    # 5. 在验证集上执行元更新
    if len(val_indices) > 0:
        val_subset = Subset(train_set, val_indices)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        # 使用第一个 batch 进行估算
        if len(val_loader) > 0:
            x_val, y_val = next(iter(val_loader))
            x_val, y_val = x_val.to(device), y_val.to(device)

            # 在聚合模型上计算验证损失 (使用 functional_call 以支持元梯度回传)
            outputs = torch.func.functional_call(model, theta_agg, (x_val,))
            loss = ce_loss(outputs, y_val)

            # 计算关于 alpha 的梯度
            alpha_grads = torch.autograd.grad(loss, alpha, retain_graph=False)[0]

            # 元学习步长更新 alpha
            with torch.no_grad():
                alpha -= lr_alpha * alpha_grads

    # 6. 执行最终聚合（带阈值去边）
    with torch.no_grad():
        w_final = F.softmax(alpha, dim=0)

        # 阈值去边：去掉权重小于阈值的连接
        w_final = w_final * (w_final >= threshold).float()
        if w_final.sum() > 0:
            w_final = w_final / w_final.sum()

        # 最终状态聚合
        final_state = {k: theta_t[k].to(device).clone() for k in theta_t.keys()}
        w_final_gpu = w_final.to(device)

        for k in final_state.keys():
            layer_deltas = torch.stack([d[k].to(device) for d in neighbor_deltas])
            dims = [1] * (layer_deltas.dim() - 1)
            final_state[k] -= torch.sum(
                layer_deltas * w_final_gpu.view(-1, *dims), dim=0
            )

    return [
        {k: v.cpu().detach().clone() for k, v in final_state.items()},  # 最终模型状态
        alpha.cpu().detach().clone(),  # 更新后的 alpha
        w_final.cpu().detach().clone().tolist(),  # 最终的混合权重
    ]


class Server(BaseServer):
    """
    L2C Server：协调两阶段协作更新

    两阶段流程：
    Phase 1: 并行计算各客户端的本地参数增量 Delta
    Phase 2: 分发邻居 Delta 并执行元更新与最终聚合
    """

    def __init__(self, args):
        super().__init__(pfl=True, args=args)

        # 1. 自动生成邻接矩阵（用于确定协作节点）
        self.A = generate_adjacency_matrix(args)

        # 2. 初始化各节点的混合权重参数 alpha（元学习对象）
        # 每个节点 i 对其邻居都有一个独立的权重分量
        self.alphas = {}
        for i in range(self.num_clients):
            num_collaborators = int(torch.sum(self.A[i] > 0).item())
            self.alphas[i] = torch.zeros(num_collaborators)

        # 3. 为 phase1 结果准备存储
        self.phase1_results = {}

    def fit(self):
        """主训练流程：执行 L2C 的两阶段协作更新"""
        num_join_clients = int(self.num_clients * self.args.join_ratio)
        num_join_clients = max(1, num_join_clients)

        for round_idx in range(self.rounds):
            t0 = time.time()
            logger.info(f"--- L2C Round {round_idx + 1}/{self.rounds} ---")

            # 1. 随机选择参与的客户端
            selected_clients = np.random.choice(
                self.num_clients, num_join_clients, replace=False
            )
            logger.info(f"Selected clients: {selected_clients}")

            # --- Phase 1：并行计算所有客户端的本地 Delta ---
            payloads_p1 = []
            for cid in selected_clients:
                payloads_p1.append(
                    [
                        cid,
                        self.client_gpu[cid],
                        self.clients_state[cid],
                        self.train_sets[cid],
                        self.args.model,
                        self.args.dataset,
                        self.test_set[cid],
                        self.num_class,
                        self.args.feature_dim,
                        self.args.batch_size,
                        self.args.epochs,
                        self.args.lr,
                        self.args.val_ratio,
                    ]
                )

            p1_results = self.run_clients(client_worker_phase1, payloads_p1)

            # 整理中间变量
            cid_to_delta = {}
            cid_to_theta_t = {}
            cid_to_indices = {}

            total_loss = 0.0
            for cid in selected_clients:
                avg_loss, theta_t, delta_theta, train_indices, val_indices = p1_results[
                    cid
                ]
                total_loss += avg_loss
                cid_to_delta[cid] = delta_theta
                cid_to_theta_t[cid] = theta_t
                cid_to_indices[cid] = {"train": train_indices, "val": val_indices}

            self.loss.append(total_loss / len(selected_clients))

            # --- Phase 2：分发邻居 Delta 并执行元更新与最终聚合 ---
            payloads_p2 = []

            for i in selected_clients:
                # 获取节点 i 的协作邻居
                neighbors = torch.where(self.A[i] > 0)[0].tolist()
                neighbors.sort()

                # 收集邻居的 Delta（如果邻居也被选中）
                neighbor_deltas = []
                for nb in neighbors:
                    if nb in cid_to_delta:
                        neighbor_deltas.append(cid_to_delta[nb])
                    else:
                        # 使用之前保存的 Delta
                        neighbor_deltas.append(
                            self.phase1_results.get(nb, cid_to_delta[neighbors[0]])
                        )

                payloads_p2.append(
                    [
                        i,
                        self.client_gpu[i],
                        self.clients_state[i],
                        cid_to_theta_t[i],
                        self.train_sets[i],
                        self.args.model,
                        self.args.dataset,
                        self.test_set[i],
                        self.num_class,
                        self.args.feature_dim,
                        self.args.batch_size,
                        self.args.epochs,
                        self.args.lr,
                        neighbors,
                        neighbor_deltas,
                        self.alphas[i],
                        cid_to_indices[i]["val"],
                        self.args.lr_alpha,
                        self.args.threshold,
                    ]
                )

            p2_results = self.run_clients(client_worker_phase2, payloads_p2)

            # --- 更新 Server 端状态 ---
            for i in selected_clients:
                model_state, alpha, w_final = p2_results[i]
                self.clients_state[i] = model_state
                self.alphas[i] = alpha
                # 保存该轮的 Delta 用于下一轮
                self.phase1_results[i] = cid_to_delta[i]

            # --- 聚合虚拟全局模型（Evaluation Oracle）---
            self.aggregate()

            # --- 评估与日志记录 ---
            self.evaluate()

            print(
                f"[Round {round_idx + 1}] Avg Loss: {self.loss[-1]:.4f}, Acc: {self.acc[-1]:.2f}%"
            )
            print(f"[Round {round_idx + 1}] Time spent: {time.time() - t0:.2f}s")

    def aggregate(self):
        """
        L2C 虽然是个性化/去中心化算法，但我们依然可以计算一个虚拟全局模型作为参考。
        """
        states = [self.clients_state[cid] for cid in range(self.num_clients)]
        weights = [1.0 / self.num_clients] * self.num_clients
        aggregated_state = {}

        for key in states[0].keys():
            aggregated_state[key] = torch.zeros_like(states[0][key]).to(self.device)
            for state, weight in zip(states, weights):
                aggregated_state[key] += state[key].to(self.device) * weight

        # 注意：这里仅更新 self.model 用于可能的全局评估参考，不影响 clients_state
        self.model.load_state_dict(aggregated_state)

    def save(self):
        """保存模型和相关状态"""
        f = {
            "acc": self.acc,
            "loss": self.loss,
            "state_dict": {
                "client": self.clients_state,
                "topology": self.A.cpu()
                if isinstance(self.A, torch.Tensor)
                else self.A,
                "alphas": self.alphas,
            },
        }
        self.deal_save(f)
