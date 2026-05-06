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
)


def get_path(args):
    """生成日志文件路径，包含拓扑参数和稀疏化参数"""
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
        f"_{args.dense_ratio}_{args.anneal_factor}"
    )
    return os.path.join(args.log_path, f"{args.file_name}_{args.cur_time}.log")


def client_worker(params):
    """
    DisPFL 客户端工作函数：稀疏训练，带动态掩码搜索

    参数结构：
        params: [
            client_id,
            device,
            model_state (dict),
            masks (dict),              # 稀疏掩码
            train_set,
            model_name,
            dataset_name,
            test_set,
            num_classes,
            feature_dim,
            batch_size,
            local_epochs,
            lr,
            round_idx,
            num_rounds,
            anneal_factor,
        ]
    """
    (
        _,
        device,
        model_state,
        masks,
        train_set,
        model_name,
        dataset_name,
        feature_dim,
        batch_size,
        local_epochs,
        lr,
        round_idx,
        num_rounds,
        anneal_factor,
    ) = params

    # 1. 初始化模型并加载参数
    model = get_model(model_name, dataset_name, feature_dim).to(device)
    model.load_state_dict(model_state)
    model.train()

    # 2. 准备数据加载器
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # 3. 本地训练循环（带梯度掩码）
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

            # DisPFL 核心：梯度掩码（只更新被掩码为 1 的位置）
            for name, param in model.named_parameters():
                if name in masks:
                    param.grad.mul_(masks[name].to(device))

            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            num_batches += 1

    # 4. 动态掩码搜索（Algorithm 2：Local mask searching）
    # 4.1 计算当前的剪枝率 alpha_t（余弦退火）
    # alpha_t 随训练动态衰减：alpha_t = alpha_0 * 0.5 * (1 + cos(t * pi / T))
    alpha_t = (
        anneal_factor
        * 0.5
        * (1 + torch.cos(torch.tensor(round_idx * torch.pi / num_rounds)))
    ).to(device)

    # 4.2 获取用于重生长的梯度信息
    model.zero_grad()
    # 从 loader 中取一个 batch 的数据来计算全梯度
    x, y = next(iter(loader))
    x, y = x.to(device), y.to(device)
    output = model(x)
    loss = ce_loss(output, y)
    loss.backward()

    # 4.3 逐层进行剪枝（Pruning）和生长（Regrowing）
    new_masks = {}
    for name, param in model.named_parameters():
        if name in masks:
            mask = masks[name].to(device)
            weights = param.data
            grads = param.grad.data

            # A. Magnitude Pruning：按权重绝对值裁剪
            num_active = int(torch.sum(mask).item())
            n_update = int(num_active * alpha_t)

            if n_update > 0:
                active_weights = weights[mask > 0].abs()
                # 计算阈值：保留绝对值最大的 (num_active - n_update) 个权重
                k_val = max(1, num_active - n_update)
                threshold = torch.topk(active_weights, k_val, largest=True).values.min()
                mask = (weights.abs() >= threshold).float()

            # B. Gradient Regrowing：按梯度强度恢复权重
            # 保持总密度不变：重生长的数量等于剪掉的数量 (n_update)
            if n_update > 0:
                # 仅在当前掩码为 0 的位置寻找梯度最大的权重恢复
                inactive_mask = (mask == 0).float()
                inactive_grads = (grads * inactive_mask).abs()

                # 确定恢复阈值
                num_available = int(torch.sum(inactive_mask).item())
                actual_regrow = min(n_update, num_available)
                if actual_regrow > 0:
                    regrow_threshold = torch.topk(
                        inactive_grads.view(-1), actual_regrow, largest=True
                    ).values.min()
                    regrow_mask = (inactive_grads >= regrow_threshold).float()
                    mask = mask + regrow_mask

            new_masks[name] = mask.cpu()

    return [
        total_loss / num_batches,  # avg_loss
        {
            k: v.cpu().detach().clone() for k, v in model.state_dict().items()
        },  # model_state
        new_masks,  # updated_masks
        correct / total if total > 0 else 0.0,  # accuracy
    ]


class Server(BaseServer):
    """
    DisPFL Server：动态去中心化稀疏学习框架

    关键特性：
    - 使用动态稀疏掩码进行参数更新
    - 基于去中心化拓扑的聚合
    - 余弦退火剪枝率
    - 梯度驱动的权重恢复
    """

    def __init__(self, args):
        super().__init__(pfl=True, args=args)

        # 1. 生成拓扑结构（generate_adjacency_matrix 已内置自环）
        self.A = generate_adjacency_matrix(args).to(self.device).float()

        # 2. 初始化稀疏掩码
        self.dense_ratio = args.dense_ratio
        initial_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

        self.client_masks = {
            cid: self._init_masks(initial_state) for cid in range(args.num_clients)
        }

        # 3. 严格执行初始掩码：直接将未被掩码的参数置为 0
        for cid in range(args.num_clients):
            for k, mask in self.client_masks[cid].items():
                self.clients_state[cid][k] = self.clients_state[cid][k].mul_(mask)

    def _init_masks(self, states):
        """随机初始化密度为 dense_ratio 的掩码"""
        masks = {}
        for k, v in states.items():
            if "weight" in k or "bias" in k:
                mask = torch.rand(v.shape) < self.dense_ratio
                masks[k] = mask.float()
        return masks

    def fit(self):
        """主训练流程"""
        num_join_clients = int(self.num_clients * self.args.join_ratio)
        num_join_clients = max(1, num_join_clients)

        for r in range(self.rounds):
            t0 = time.time()
            print(f"\n--- DisPFL Round {r + 1}/{self.rounds} ---")

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
                    self.client_masks[i],
                    self.train_sets[i],
                    self.args.model,
                    self.args.dataset,
                    self.args.feature_dim,
                    self.args.batch_size,
                    self.args.epochs,
                    self.args.lr,
                    r,  # round_idx
                    self.rounds,  # num_rounds
                    self.args.anneal_factor,
                ]

            params = [get_client_param(i) for i in selected_clients]

            # 3. 启动客户端并行训练
            results = self.run_clients(client_worker, params)

            # 4. 收集客户端的更新
            total_loss = 0.0
            for client_idx in selected_clients:
                avg_loss, model_state, masks, _ = results[client_idx]
                total_loss += avg_loss
                self.clients_state[client_idx] = model_state
                self.client_masks[client_idx] = masks

            self.loss.append(total_loss / len(selected_clients))

            # 5. 执行去中心化聚合
            self.aggregate()

            # 6. 评估模型
            self.evaluate()

            print(f"Avg Loss: {self.loss[-1]:.4f}, Acc: {self.acc[-1]:.2f}%")
            print(f"Round finished in {time.time() - t0:.2f} seconds")

    def aggregate(self):
        """
        去中心化聚合：依据掩码交集进行稀疏聚合

        每个客户端从邻居（通过邻接矩阵 A 定义）处聚合参数，
        仅在两个参数都被掩码的位置更新值。
        """
        new_client_states = {i: {} for i in range(self.num_clients)}

        for k in self.clients_state[0].keys():
            # 堆叠所有客户端的该层参数 [N, ...]
            layer_stacked = torch.stack(
                [
                    self.clients_state[i][k].to(self.device)
                    for i in range(self.num_clients)
                ]
            )
            orig_shape = layer_stacked.shape[1:]
            layer_flat = layer_stacked.view(self.num_clients, -1)

            # 判断该层是否具有掩码（DisPFL 逻辑：只有 weight/bias 有 mask）
            if k in self.client_masks[0]:
                mask_stacked = torch.stack(
                    [
                        self.client_masks[i][k].to(self.device)
                        for i in range(self.num_clients)
                    ]
                )
                mask_flat = mask_stacked.view(self.num_clients, -1)

                # 邻域覆盖计数: CountM = A @ M
                count_mask_flat = torch.mm(self.A, mask_flat)

                # 邻域加权和: SumW = A @ (S * M)
                sum_w_flat = torch.mm(self.A, layer_flat * mask_flat)

                # 计算平均并应用当前客户端的掩码限制
                # 避免零除：count_mask_flat 为 0 的位置结果为 0
                denom = torch.where(
                    count_mask_flat > 0,
                    count_mask_flat,
                    torch.ones_like(count_mask_flat),
                )
                new_layer_flat = (sum_w_flat / denom) * mask_flat
            else:
                # 非掩码参数（如 BN 层）：直接根据邻居数量求均值
                neighbor_counts = self.A.sum(dim=1, keepdim=True)
                new_layer_flat = torch.mm(self.A, layer_flat) / neighbor_counts

            new_layer = new_layer_flat.view(self.num_clients, *orig_shape)
            for i in range(self.num_clients):
                new_client_states[i][k] = new_layer[i].cpu()

        # 更新客户端状态
        for i in range(self.num_clients):
            self.clients_state[i] = new_client_states[i]

        # 计算虚拟全局模型（Evaluation Oracle）
        states = [self.clients_state[cid] for cid in range(self.num_clients)]
        weights = self.weights
        aggregated_state = {}

        for key in states[0].keys():
            aggregated_state[key] = torch.zeros_like(states[0][key])
            for state, weight in zip(states, weights):
                aggregated_state[key] += state[key] * weight

        self.model.load_state_dict(aggregated_state)

    def save(self):
        """保存模型和相关状态"""
        f = {
            "acc": self.acc,
            "loss": self.loss,
            "state_dict": {
                "client": self.clients_state,
                "client_masks": self.client_masks,
                "topology": self.A.cpu(),
            },
        }
        self.deal_save(f)
