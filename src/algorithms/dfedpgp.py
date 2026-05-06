import os
import time

import numpy as np
import torch

from .utils import (
    BaseServer,
    ce_loss,
    generate_adjacency_matrix,
    get_model,
    param_aggregate,
)


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
    args.file_name = f"{args.name_pre}_{adj_suffix}_{args.epochs}_{args.local_v_epochs}_{args.lr_v}_{args.momentum_v}_{args.weight_decay_v}"
    return os.path.join(args.log_path, f"{args.file_name}_{args.cur_time}.log")


def client_worker(params):
    """
    DFedPGP 客户端工作函数：实现解耦更新和梯度推送

    参数结构：
        params: [
            client_id,
            device,
            body_biased (dict),           # 带偏的特征提取器参数 (U)
            mu (float),                    # 偏置标量
            head_state (dict),             # 私有分类头参数 (V)
            train_set (torch.utils.data.Dataset),
            model_name,
            dataset_name,
            lr_u,                          # 特征提取器学习率
            lr_v,                          # 分类头学习率
            batch_size,
            local_u_epochs,
            local_v_epochs,
            feature_dim,
            momentum,
            weight_decay,
        ]
    """
    (
        client_id,
        device,
        body_biased,
        mu,
        head_state,
        train_set,
        model_name,
        dataset_name,
        lr_u,
        lr_v,
        batch_size,
        local_u_epochs,
        local_v_epochs,
        feature_dim,
        momentum,
        weight_decay,
    ) = params

    # 1. 初始化模型并加载参数
    model = get_model(model_name, dataset_name, feature_dim).to(device)

    # 合并 body 和 head 参数以加载完整模型
    full_state = {}
    full_state.update(body_biased)
    full_state.update(head_state)
    model.load_state_dict(full_state)

    # 2. 准备解偏后的特征提取器参考值 (z_0 = u/mu)
    with torch.no_grad():
        z_0 = {k: v.to(device) / mu for k, v in body_biased.items()}

    # 3. 初始化两个独立的优化器
    optimizer_v = torch.optim.SGD(
        model.classifier.parameters(),
        lr=lr_v,
    )
    optimizer_u = torch.optim.SGD(
        model.extractor.parameters(),
        lr=lr_u,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    # 4. 准备数据加载器
    loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    # ========== Phase 1: 训练分类头 V (固定 Body 为初始解偏值 z_0) ==========
    for param in model.extractor.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

    # 将解偏后的 z_0 注入 Extractor
    model.extractor.load_state_dict(
        {k.replace("extractor.", ""): v for k, v in z_0.items()}
    )

    model.train()
    for _ in range(local_v_epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer_v.zero_grad()
            out = model(x)
            loss = ce_loss(out, y)
            loss.backward()
            optimizer_v.step()

    # ========== Phase 2: 训练特征提取器 U (固定已训练好的分类头) ==========
    for param in model.extractor.parameters():
        param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = False

    # 恢复为原始带偏的 body_biased (U)
    model.extractor.load_state_dict(
        {k.replace("extractor.", ""): v.to(device) for k, v in body_biased.items()}
    )

    total_loss = 0.0
    num_batches = 0

    for _ in range(local_u_epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            # a. 执行 U -> Z 转换 (除以 mu)，使 Forward 作用在解偏状态上
            with torch.no_grad():
                for param in model.extractor.parameters():
                    param.data.div_(mu)

            # b. 标准前向与反向传播
            optimizer_u.zero_grad()
            out = model(x)
            loss = ce_loss(out, y)
            loss.backward()

            # c. 梯度修正与状态回滚：将梯度适配到 u，并将参数乘回 mu 复位
            with torch.no_grad():
                for param in model.extractor.parameters():
                    if param.grad is not None:
                        param.grad.data.div_(mu)
                    param.data.mul_(mu)

            # d. 执行局部更新
            optimizer_u.step()

            total_loss += loss.item()
            num_batches += 1

    # 5. 提取并返回更新后的 body 和 head
    new_full_state = model.state_dict()

    shared_keys = [k for k in new_full_state.keys() if k.startswith("extractor.")]
    head_keys = [k for k in new_full_state.keys() if k.startswith("classifier.")]

    return [
        total_loss / num_batches,  # avg_loss
        {
            k: new_full_state[k].cpu().detach().clone() for k in shared_keys
        },  # body_shared
        {k: new_full_state[k].cpu().detach().clone() for k in head_keys},  # head_state
    ]


class Server(BaseServer):
    """
    DFedPGP Server：处理基于 Push-Sum 的去中心化个性化聚合

    关键特性：
    - 使用 Gossip 算法进行去中心化聚合
    - 维护每个客户端的共享体参数 (body) 和私有头参数 (head)
    - 使用标量 mu 进行偏置校正
    - 支持任意网络拓扑
    """

    def __init__(self, args):
        super().__init__(pfl=True, args=args)

        # 验证模型架构：必须有 extractor 和 classifier
        init_state = self.model.state_dict()
        self.shared_keys = [k for k in init_state.keys() if k.startswith("extractor.")]
        self.head_keys = [k for k in init_state.keys() if k.startswith("classifier.")]

        # 1. 拓扑初始化与混合矩阵预计算
        self.adj_matrix = generate_adjacency_matrix(args)
        out_degrees = self.adj_matrix.sum(dim=1)
        # 预计算 M = (A/d)^T，用于向量化混合：U_next = M @ U_curr
        self.M = (self.adj_matrix / out_degrees.view(-1, 1)).t().to(self.device)

        # 2. 客户端状态池初始化
        body_proto = {k: init_state[k].clone().cpu() for k in self.shared_keys}
        head_proto = {k: init_state[k].clone().cpu() for k in self.head_keys}

        self.client_body = {
            cid: {k: v.clone() for k, v in body_proto.items()}
            for cid in range(self.num_clients)
        }
        self.client_head = {
            cid: {k: v.clone() for k, v in head_proto.items()}
            for cid in range(self.num_clients)
        }
        self.client_mu = {cid: 1.0 for cid in range(self.num_clients)}

        # 3. 初始化 clients_state 为完整模型（用于评估）
        self.update_clients_state()

        # 4. 设置学习率
        self.lr_u = args.lr
        self.local_u_epochs = args.epochs
        self.lr_v = args.lr_v
        self.local_v_epochs = args.local_v_epochs
        self.momentum_v = args.momentum_v
        self.weight_decay_v = args.weight_decay_v

    def fit(self):
        """主训练循环"""
        num_join_clients = int(self.num_clients * self.args.join_ratio)
        num_join_clients = max(1, num_join_clients)

        for r in range(self.rounds):
            t0 = time.time()
            print(f"\n--- DFedPGP Round {r + 1}/{self.rounds} ---")

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
                    self.client_body[i],
                    self.client_mu[i],
                    self.client_head[i],
                    self.train_sets[i],
                    self.args.model,
                    self.args.dataset,
                    self.lr_u,
                    self.lr_v,
                    self.args.batch_size,
                    self.local_u_epochs,
                    self.local_v_epochs,
                    self.args.feature_dim,
                    self.momentum_v,
                    self.weight_decay_v,
                ]

            params = [get_client_param(i) for i in selected_clients]

            # 3. 启动客户端并行训练
            results = self.run_clients(client_worker, params)

            # 4. 收集客户端的更新
            total_loss = 0.0
            for cid in selected_clients:
                avg_loss, body_shared, head_state = results[cid]
                total_loss += avg_loss
                self.client_body[cid] = body_shared
                self.client_head[cid] = head_state

            self.loss.append(total_loss / len(selected_clients))

            # 5. 执行去中心化聚合（Gossip/Push-Sum）
            self.aggregate()

            # 6. 更新客户端状态并评估
            self.update_clients_state()
            self.evaluate()

            print(f"Avg Loss: {self.loss[-1]:.4f}, Acc: {self.acc[-1]:.2f}%")
            print(f"Round finished in {time.time() - t0:.2f} seconds")

    def aggregate(self):
        """
        执行基于 Push-Sum 的去中心化聚合。

        使用 param_aggregate 逐个客户端聚合邻居状态：
        1. 混合共享模型参数 body
        2. 混合标量权重 mu
        3. 注意：私有头 head 保持个性化，不参与混合
        """
        new_client_body = {}
        new_client_mu = {}

        for i in range(self.num_clients):
            # 获取节点 i 的权重向量 (M 的第 i 行)
            # M = (A/d)^T，所以 M[i, j] 表示节点 j 发送给节点 i 的权重
            weights = self.M[i].tolist()

            # 1. 聚合共享体参数 (Body)
            all_bodies = [self.client_body[j] for j in range(self.num_clients)]
            new_client_body[i] = param_aggregate(all_bodies, weights)

            # 2. 聚合标量权重 (Mu)
            mu_val = 0.0
            for j in range(self.num_clients):
                mu_val += weights[j] * self.client_mu[j]
            new_client_mu[i] = mu_val

        # 更新服务器端维护的状态池
        self.client_body = new_client_body
        self.client_mu = new_client_mu

    def update_clients_state(self):
        """
        将 (body, mu, head) 转换为完整的去偏模型供评估使用

        对于每个客户端：
        - 共享部分：z = u / mu（执行去偏）
        - 私有部分：v（保持不变）
        """
        for cid in range(self.num_clients):
            full_state = {}
            # 合并私有分类头
            full_state.update({k: v.clone() for k, v in self.client_head[cid].items()})
            # 共享部分执行推拉偏置校正 (z = u / mu)
            for k, v in self.client_body[cid].items():
                full_state[k] = v.clone() / self.client_mu[cid]
            self.clients_state[cid] = full_state

    def save(self):
        """保存模型和相关状态"""
        f = {
            "acc": self.acc,
            "loss": self.loss,
            "state_dict": {
                "client": self.clients_state,
                "body": self.client_body,
                "head": self.client_head,
                "mu": self.client_mu,
                "topology": self.adj_matrix.cpu()
                if isinstance(self.adj_matrix, torch.Tensor)
                else self.adj_matrix,
            },
        }
        self.deal_save(f)
