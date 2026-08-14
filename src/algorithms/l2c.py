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
    fmt_num,
    generate_adjacency_matrix,
    get_model,
)

logger = logging.getLogger(__name__)


def get_path(args):
    """生成日志文件路径，包含拓扑参数和元学习超参数"""
    adj_suffix = f"{args.adj_type}"
    if args.adj_type == "random":
        adj_suffix += f"_{fmt_num(args.edge_p)}"
    elif args.adj_type == "small_world":
        adj_suffix += f"_{fmt_num(args.k_small_world)}_{fmt_num(args.edge_p)}"
    elif args.adj_type == "scale_free":
        adj_suffix += f"_{fmt_num(args.m_scale_free)}"

    # 将算法的关键超参加入文件名
    args.file_name = (
        f"{args.common_name}_{adj_suffix}"
        f"_{fmt_num(args.val_ratio)}_{fmt_num(args.lr_alpha)}"
        f"_{fmt_num(args.prune_round)}_{fmt_num(args.prune_num)}"
    )
    return os.path.join(args.log_path, f"{args.file_name}_{args.cur_time}.log")


def train_phase1(params):
    """
    L2C 客户端第一阶段：本地训练并计算参数增量 Delta Theta

    参数结构：
        params: [
            device,
            model_state (dict),
            train_set,
            model_name,
            dataset_name,
            feature_dim,
            batch_size,
            local_epochs,
            lr,
            val_ratio,
        ]
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
        local_epochs,
        feature_dim,
        num_class,
        val_ratio,
    ) = params

    # 1. 初始化模型并加载参数
    model = get_model(model_name, dataset_name, num_class, feature_dim).to(device)
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
    num_batches = 0

    for _ in range(local_epochs):
        for x, y, *_ in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = ce_loss(output, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

    # 4. 计算 Delta = theta_t - theta_updated
    theta_mid = model.state_dict()
    delta_theta = {
        k: (theta_t[k].to(device) - theta_mid[k]).cpu() for k in theta_t.keys()
    }

    return {
        "loss": total_loss / num_batches,  # avg_loss
        "state_t": theta_t,  # 返回初始参数供第二阶段使用
        "delta": {k: v.detach().clone() for k, v in delta_theta.items()},  # delta_theta
        "train_idx": train_indices,  # 用于第二阶段的验证集
        "val_idx": val_indices,
    }


def train_phase2(params):
    """
    L2C 客户端第二阶段：元学习更新 alpha 并执行最终加权聚合

    参数结构：
        params: [
            device,
            model_state (dict),
            train_set,
            theta_t (dict),
            model_name,
            dataset_name,
            feature_dim,
            batch_size,
            neighbor_deltas (list),
            alpha (tensor),
            val_indices (list),
            lr_alpha,
        ]
    """
    (
        _,
        device,
        model_state,
        train_set,
        model_name,
        dataset_name,
        _,
        batch_size,
        _,
        feature_dim,
        num_class,
        theta_t,
        neighbor_deltas,
        alpha,
        val_indices,
        lr_alpha,
    ) = params

    # 1. 初始化模型
    model = get_model(model_name, dataset_name, num_class, feature_dim).to(device)
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

    # 6. 整理返回结果（严格遵守伪代码：返回 alpha 更新前的聚合模型）
    return {
        "state": {
            k: v.cpu().detach().clone() for k, v in theta_agg.items()
        },  # 对应伪代码 Line 16 的 theta_i^{t+1}
        "alpha": alpha.cpu().detach().clone(),  # 更新后的 alpha 用于下一轮
        "weights": w.cpu()
        .detach()
        .clone()
        .tolist(),  # 返回更新前的权重用于 Server 端剪枝判断
    }


class Server(BaseServer):
    """
    L2C Server：协调两阶段协作更新

    两阶段流程：
    Phase 1: 并行计算各客户端的本地参数增量 Delta
    Phase 2: 分发邻居 Delta 并执行元更新与最终聚合
    """

    def __init__(self, args):
        super().__init__(pfl=True, args=args)
        self.val_ratio = args.val_ratio
        self.lr_alpha = args.lr_alpha
        self.prune_round = args.prune_round
        self.prune_num = args.prune_num

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
        num_join = max(1, int(self.num_clients * self.join_ratio))

        for round_idx in range(self.rounds):
            t0 = time.time()
            logger.info(f"--- L2C Round {round_idx + 1}/{self.rounds} ---")

            # 1. 随机选择参与的客户端
            selected = torch.randperm(self.num_clients)[:num_join].tolist()
            logger.info(f"Selected clients: {selected}")

            # --- Phase 1：并行计算所有客户端的本地 Delta ---
            payloads_p1 = self.build_base_params(selected)
            for params, cid in zip(payloads_p1, selected):
                params[2] = self.clients_state[cid]
                params.append(self.val_ratio)

            p1_results = self.run_clients(train_phase1, payloads_p1)

            # 整理中间变量
            cid_to_delta = {}
            cid_to_theta_t = {}
            cid_to_indices = {}

            total_loss = 0.0
            for cid, res in p1_results.items():
                total_loss += res["loss"]
                cid_to_delta[cid] = res["delta"]
                cid_to_theta_t[cid] = res["state_t"]
                cid_to_indices[cid] = {"train": res["train_idx"], "val": res["val_idx"]}

            self.loss.append(total_loss / len(selected))

            # --- Phase 2：分发邻居 Delta 并执行元更新与最终聚合 ---
            p2_base = self.build_base_params(selected)
            payloads_p2 = []
            for params, i in zip(p2_base, selected):
                # 获取节点 i 的协作邻居
                neighbors = torch.where(self.A[i] > 0)[0].tolist()
                neighbors.sort()

                # 收集邻居的 Delta
                neighbor_deltas = []
                for nb in neighbors:
                    if nb in cid_to_delta:
                        neighbor_deltas.append(cid_to_delta[nb])
                    elif nb in self.phase1_results:
                        neighbor_deltas.append(self.phase1_results[nb])
                    else:
                        # 既无本轮增量也无缓存，则使用零增量（不贡献变化），严禁随机 fallback
                        zero_delta = {
                            k: torch.zeros_like(v) for k, v in cid_to_delta[i].items()
                        }
                        neighbor_deltas.append(zero_delta)

                params[2] = self.clients_state[i]
                params.append(cid_to_theta_t[i])
                params.append(neighbor_deltas)
                params.append(self.alphas[i])
                params.append(cid_to_indices[i]["val"])
                params.append(self.lr_alpha)
                payloads_p2.append(params)

            p2_results = self.run_clients(train_phase2, payloads_p2)

            # --- 更新 Server 端状态 ---
            all_weights = {}
            for cid, res in p2_results.items():
                self.clients_state[cid] = res["state"]
                self.alphas[cid] = res["alpha"]
                all_weights[cid] = res["weights"]
                # 保存该轮的 Delta 用于下一轮
                self.phase1_results[cid] = cid_to_delta[cid]

            # --- 拓扑演化：Top-K 剪枝 (对应伪代码 Line 20-22) ---
            # 仅在特定的 prune_round (T0) 执行
            if round_idx + 1 == self.prune_round and self.prune_num > 0:
                logger.info(
                    f"Applying Top-K pruning (K={self.prune_num}) at round {round_idx + 1}"
                )
                for i in range(self.num_clients):
                    if i not in all_weights:
                        continue

                    # 获取当前邻居列表（对应权重向量的顺序）
                    neighbors = torch.where(self.A[i] > 0)[0].tolist()
                    neighbors.sort()

                    weights = np.array(all_weights[i])
                    # 排除自环（不剪掉自己）
                    neighbor_indices = [
                        idx for idx, nb in enumerate(neighbors) if nb != i
                    ]
                    if len(neighbor_indices) <= self.prune_num:
                        continue

                    # 找到权重最小的 K0 个邻居的索引
                    neighbor_weights = weights[neighbor_indices]
                    to_prune_indices = np.argsort(neighbor_weights)[: self.prune_num]

                    for idx in to_prune_indices:
                        neighbor_to_remove = neighbors[neighbor_indices[idx]]
                        self.A[i, neighbor_to_remove] = 0.0
                        logger.debug(
                            f"Client {i}: Removed neighbor {neighbor_to_remove}"
                        )

            # --- 聚合虚拟全局模型（Evaluation Oracle）---
            self.aggregate()

            # --- 评估与日志记录 ---
            self.evaluate()

            print(
                f"[Round {round_idx + 1}] Avg Loss: {self.loss[-1]:.4f}, Acc: {self.acc[-1]:.2f}%, Time spent: {time.time() - t0:.2f}s"
            )

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
        metrics = {"acc": self.acc, "loss": self.loss}
        params = {
            "client": self.clients_state,
            "aux": {
                "topology": self.A.cpu()
                if isinstance(self.A, torch.Tensor)
                else self.A,
                "alphas": self.alphas,
            },
        }
        self.deal_save(metrics, params)
