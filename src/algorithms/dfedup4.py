import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .utils import (
    BaseServer,
    ce_loss,
    extract_prototypes,
    generate_adjacency_matrix,
    get_model,
    mse_loss,
    param_aggregate,
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
    DFedUP Worker: 执行本地模型训练、对比损失计算及原型提取。
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

    # 1. 初始化模型并加载状态
    model = get_model(model_name, dataset_name, feature_dim).to(device)
    model.load_state_dict(model_state)
    consensus_P = consensus_P.to(device)

    # 2. 设置数据加载器与原型标签
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    proto_labels = torch.arange(num_classes, device=device)

    # === Phase 1: Classifier Calibration (固定 1 Epoch) ===
    # 目的：利用共识原型校准分类头的决策边界
    for param in model.extractor.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

    optimizer_head = torch.optim.SGD(model.classifier.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            # 本地交叉熵 + 全局原型校准
            loss_ce_local = ce_loss(model(x), y)
            p_out = model.classifier(consensus_P)
            loss_ce_proto = ce_loss(p_out, proto_labels)

            loss = loss_ce_local + mu * loss_ce_proto

            optimizer_head.zero_grad()
            loss.backward()
            optimizer_head.step()

    # === Phase 2: Extractor Alignment (执行 epochs 次) ===
    # 目的：微调特征表示，使其向共识原型靠拢
    for param in model.classifier.parameters():
        param.requires_grad = False
    for param in model.extractor.parameters():
        param.requires_grad = True

    optimizer_body = torch.optim.SGD(model.extractor.parameters(), lr=lr)
    model.train()
    total_loss, num_batches = 0.0, 0
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            features = model.extractor(x)
            logits = model.classifier(features)

            # 损失组合：交叉熵损失 + 原型 MSE 对齐损失
            l_ce = ce_loss(logits, y)
            target_protos = consensus_P[y]
            l_con = mse_loss(features, target_protos)
            loss = l_ce + mu * l_con

            optimizer_body.zero_grad()
            loss.backward()
            optimizer_body.step()

            total_loss += loss.item()
            num_batches += 1

    # 4. 提取本地最新原型并转换为 Push-Sum 状态量 (S 和 W)
    local_protos, local_counts = extract_prototypes(
        model, loader, num_classes, feature_dim, device, return_counts=True
    )

    # 向量化计算置信度驱动的 Push-Sum 初始值
    confidence = torch.log(1 + local_counts.unsqueeze(-1))
    S = confidence * local_protos
    W = confidence

    # 5. 整理返回结果（严格遵守 GEMINI.md 规约）
    new_state = {k: v.cpu().detach().clone() for k, v in model.state_dict().items()}

    return {
        "loss": total_loss / num_batches,
        "state": new_state,
        "S": S.cpu().detach().clone(),
        "W": W.cpu().detach().clone(),
    }


class Server(BaseServer):
    """
    DFedUP Server: 管理去中心化原型共识的演化。
    """

    def __init__(self, args):
        super().__init__(pfl=True, args=args)

        # 1. 通信拓扑初始化
        adj = generate_adjacency_matrix(args).to(self.device).float()

        # 预计算混合矩阵 M (列随机，保证 Push-Sum 质量守恒)
        row_sum = adj.sum(dim=1, keepdim=True)
        self.M = (adj / row_sum).t().to(self.device)

        # 2. 初始化各客户端的 Push-Sum 状态缓存与共识原型
        self.S_cache = [
            torch.zeros(self.num_class, args.feature_dim)
            for _ in range(self.num_clients)
        ]
        self.W_cache = [torch.zeros(self.num_class, 1) for _ in range(self.num_clients)]
        self.W_E_cache = torch.ones(self.num_clients, 1)
        self.consensus_P = [
            torch.zeros(self.num_class, args.feature_dim)
            for _ in range(self.num_clients)
        ]

    def fit(self):
        num_join_clients = int(self.num_clients * self.args.join_ratio)
        num_join_clients = max(1, num_join_clients)

        for r in range(self.rounds):
            t0 = time.time()
            print(f"\n--- DFedUP Round {r + 1}/{self.rounds} ---")

            selected_clients = np.random.choice(
                self.num_clients, num_join_clients, replace=False
            )
            print(f"Selected clients: {selected_clients}")

            # 1. 准备客户端参数
            def get_client_param(i):
                return (
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
                    self.num_class,
                    self.consensus_P[i],
                    self.args.mu,
                )

            params = [get_client_param(i) for i in selected_clients]

            # 2. 启动 Ray 远程训练
            results = self.run_clients(client_worker, params)

            # 3. 回收状态并更新缓存
            total_loss = 0.0
            for cid, res in results.items():
                total_loss += res["loss"]
                self.clients_state[cid] = res["state"]
                self.S_cache[cid] = res["S"]
                self.W_cache[cid] = res["W"]

            self.loss.append(total_loss / num_join_clients)

            # 4. 执行全图 Gossip
            S = torch.stack([self.S_cache[i] for i in range(self.num_clients)]).to(
                self.device
            )
            W = torch.stack([self.W_cache[i] for i in range(self.num_clients)]).to(
                self.device
            )

            S_flat = S.view(self.num_clients, -1)
            W_flat = W.view(self.num_clients, -1)

            # 根据配置执行多轮 Gossip (默认为 1，如果 args 中没有则取 1)
            gossip_rounds = getattr(self.args, "gossip_rounds", 1)
            for _ in range(gossip_rounds):
                S_flat = torch.mm(self.M, S_flat)
                W_flat = torch.mm(self.M, W_flat)

            # 5. 计算并写回共识原型
            # S_flat: [N, C*D], W_flat: [N, C*1] -> 这里 W_flat 展开其实是 [N, C]
            # 修正 W_flat 的形状以匹配 S_flat 的块结构
            W_tensor = W_flat.view(self.num_clients, self.num_class, 1)
            S_tensor = S_flat.view(
                self.num_clients, self.num_class, self.args.feature_dim
            )

            consensus = S_tensor / (W_tensor + 1e-12)

            for i in range(self.num_clients):
                self.S_cache[i] = S_tensor[i].cpu()
                self.W_cache[i] = W_tensor[i].cpu()
                self.consensus_P[i] = consensus[i].cpu()

            # 5. 执行模型 Classifier 的去中心化聚合 (Push-Sum)
            # 消融实验：仅聚合分类器，特征提取器保持私有
            new_classifiers, self.W_E_cache = pushsum_param_aggregate(
                self.clients_state,
                self.W_E_cache,
                self.M,
                gossip_rounds=gossip_rounds,
                prefix="classifier.",
            )

            for i in range(self.num_clients):
                self.clients_state[i].update(new_classifiers[i])

            # 6. 执行评估
            self.evaluate()

            print(f"Avg Loss: {self.loss[-1]:.4f}, Acc: {self.acc[-1]:.2f}%")
            print(f"Round finished in {time.time() - t0:.2f} seconds")

    def save(self):
        f = {
            "acc": self.acc,
            "loss": self.loss,
            "state_dict": self.clients_state,
            "consensus_P": self.consensus_P,
        }
        self.deal_save(f)
