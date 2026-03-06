import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import (
    BaseServer,
    ce_loss,
    get_model,
    run_parallel_clients,
)


def add_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("FedLSA Specific Arguments")
    group.add_argument(
        "--lambda_com",
        type=float,
        default=0.1,
        help="Weight for Compactness Loss (L_COM) on client side",
    )
    group.add_argument(
        "--alpha_sep",
        type=float,
        default=0.1,
        help="Weight for Separation Loss (L_SEP) on server side",
    )
    group.add_argument(
        "--server_epochs",
        type=int,
        default=1,
        help="Number of server-side optimization epochs (Es)",
    )
    group.add_argument(
        "--server_lr",
        type=float,
        default=0.01,
        help="Learning rate for server-side optimization",
    )
    group.add_argument(
        "--tau",
        type=float,
        default=0.1,
        help="Temperature parameter for separation loss",
    )
    return parser


def get_path(args):
    args.file_name = f"{args.name_pre}_{args.lambda_com}_{args.alpha_sep}_{args.server_epochs}_{args.server_lr}_{args.tau}"
    return os.path.join(args.log_path, f"{args.file_name}_{args.times}.log")


def separation_loss(anchors, tau=0.1):
    """
    计算分离损失 (L_SEP)，公式如下：
    L_SEP = log( sum_{j!=i} exp(a_i * a_j^T / tau) / (C-1) )
    """
    C = anchors.shape[0]
    # 归一化锚点到单位球面上
    anchors_norm = F.normalize(anchors, p=2, dim=1)
    # 计算成对余弦相似度: sim_matrix[i, j] = dot(a_i, a_j)
    sim_matrix = torch.matmul(anchors_norm, anchors_norm.T)
    # 指数化
    exp_sim = torch.exp(sim_matrix / tau)

    # 掩盖对角线（自身相似度），只对 j != i 求和
    mask = torch.eye(C, device=anchors.device).bool()
    exp_sim = exp_sim.masked_fill(mask, 0.0)

    # 对 j != i 求和
    sum_exp = exp_sim.sum(dim=1) / (C - 1)

    return torch.log(sum_exp + 1e-20).mean()


class AnchorMapping(nn.Module):
    """
    两层 MLP 映射函数 Theta(.) 用于将随机向量 R 映射到语义锚点 A。
    """

    def __init__(self, feature_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
        )

    def forward(self, x):
        return self.net(x)


def client_worker(params):
    """
    FedLSA 基于位置感知语义锚点 (Location-aware Semantic Anchors) 的本地训练流程 (计算紧凑度损失 Compactness Loss)。
    """
    (
        _,
        device,
        model_state,
        global_anchors,
        train_set,
        model_name,
        dataset_name,
        lr,
        batch_size,
        epochs,
        lambda_com,
        tau,
        feature_dim,
    ) = params

    model = get_model(model_name, dataset_name, feature_dim).to(device)
    model.load_state_dict(model_state)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    model.train()
    total_loss = 0.0
    num_batches = 0

    # 确保锚点在正确的设备上并分离计算图 (在客户端训练期间保持固定)
    global_anchors = global_anchors.to(device).detach()
    anchors_norm = F.normalize(global_anchors, p=2, dim=1)

    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            _, features = model(x)

            # 伪代码第 4 行: h_i = nor(phi(psi(x_i)))
            # 必须先归一化特征，再送入分类器，保证与服务端 L_ACE 的输入分布一致
            features_norm = F.normalize(features, p=2, dim=1)

            # L_CE: 使用归一化特征通过分类器计算交叉熵损失
            logits = model.classifier(features_norm)
            loss_ce = ce_loss(logits, y)

            # L_COM: 带有温度系数 (Temperature) 的紧凑度损失
            logits_com = torch.matmul(features_norm, anchors_norm.T) / tau
            loss_com = ce_loss(logits_com, y)

            # 整体损失: L_HC = L_CE + lambda * L_COM
            loss = loss_ce + lambda_com * loss_com

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

    model_state = {k: v.cpu() for k, v in model.state_dict().items()}
    return [avg_loss, model_state]


class Server(BaseServer):
    def __init__(self, args: argparse.Namespace):
        super().__init__(False, args)

        # 1. 初始化随机向量 R (可学习)
        # R 的形状与语义锚点保持一致: [C, d]
        self.R = torch.randn(self.num_class, self.args.feature_dim, device=self.device)

        # 2. 初始化映射函数 Theta (即 MLP)
        self.anchor_mapping = AnchorMapping(self.args.feature_dim).to(self.device)

        self.labels = torch.arange(self.num_class, device=self.device)

    def get_anchors(self):
        """生成语义锚点 A = Theta(R)"""
        return self.anchor_mapping(self.R)

    def fit(self):
        num_join_clients = int(self.num_clients * self.args.join_ratio)
        num_join_clients = max(1, num_join_clients)

        print(
            f"FedLSA Training with lambda_com={self.args.lambda_com}, alpha_sep={self.args.alpha_sep}"
        )

        for r in range(self.rounds):
            t0 = time.time()
            print(f"\n--- FedLSA Round {r + 1}/{self.rounds} ---")

            selected_clients = np.random.choice(
                self.num_clients, num_join_clients, replace=False
            )
            print(f"Selected clients: {selected_clients}")

            # 生成下发前的最新当前语义锚点
            # 注意: 此处必须分离计算图，因为客户端不负责优化 R 或 Theta
            current_anchors = self.get_anchors().detach().cpu()

            p = [
                [
                    i,
                    self.client_gpu[i],
                    self.clients_state[i],
                    current_anchors,
                    self.train_sets[i],
                    self.args.model,
                    self.args.dataset,
                    self.args.lr,
                    self.args.batch_size,
                    self.args.epochs,
                    self.args.lambda_com,
                    self.args.tau,
                    self.args.feature_dim,
                ]
                for i in selected_clients
            ]

            results = run_parallel_clients(
                client_worker=client_worker,
                parameters=p,
                gpu_pools=self.gpu_pools,
                mp=self.mp,
            )

            total_loss = 0.0
            selected_states = []
            current_weights = []
            for i in selected_clients:
                client_loss, client_state = results[i]
                total_loss += client_loss
                self.clients_state[i] = client_state
                selected_states.append(client_state)
                current_weights.append(self.weights[i])
            self.loss.append(total_loss / num_join_clients)
            sum_weights = sum(current_weights)
            norm_weights = [w / sum_weights for w in current_weights]

            self.aggregate(selected_states, norm_weights)
            self.server_optimization()
            self.evaluate()
            print(
                f"Global Accuracy: {self.acc[-1]:.2f}%, Avg Loss: {self.loss[-1]:.4f}"
            )
            print(f"Round finished in {time.time() - t0:.2f} seconds")

    def server_optimization(self):
        """
        使用 L_LSA = L_ACE + alpha * L_SEP 损失优化潜在向量集 R 和锚点映射函数 Theta。

        注意：全局模型 Phi_glo 在此阶段是冻结的，仅仅用于正向评估。
        """
        self.model.eval()  # 设置模型为评估模式 (即冻结处理)
        self.model.to(self.device)
        self.anchor_mapping.train()

        # 为 R 和 Theta 构建联合优化器
        # 此处严禁包含 self.model.parameters()
        if not self.R.requires_grad:
            self.R.requires_grad_(True)

        optimizer = torch.optim.SGD(
            [self.R] + list(self.anchor_mapping.parameters()),
            lr=self.args.server_lr,
        )

        # 临时禁用模型主参数的梯度计算，以确保它们不被更新，
        # 同时能最大化地节省内存和算力
        for param in self.model.parameters():
            param.requires_grad = False

        print(f"-> Server Optimization for {self.args.server_epochs} epochs...")

        for e in range(self.args.server_epochs):
            # 1. 生成语义锚点 A = Theta(R)
            anchors = self.get_anchors()

            # 为计算损失将锚点特征归一化
            anchors_norm = F.normalize(anchors, p=2, dim=1)

            # 2. 计算自适应类别能量损失 L_ACE (Adaptive Class Energy Loss)
            # 使用冻结的全局分类器对锚点进行计算分类
            logits = self.model.classifier(anchors_norm)
            loss_ace = ce_loss(logits, self.labels)

            # 3. 计算分离损失 L_SEP (Separation Loss)
            loss_sep = separation_loss(anchors, tau=self.args.tau)

            # 整体服务端优化损失
            loss_lsa = loss_ace + self.args.alpha_sep * loss_sep

            optimizer.zero_grad()
            loss_lsa.backward()
            optimizer.step()

            if e == 0 or (e + 1) == self.args.server_epochs:
                print(
                    f"   Epoch {e + 1}: L_ACE={loss_ace.item():.4f}, L_SEP={loss_sep.item():.4f}"
                )

        # 恢复模型主参数的梯度计算能力 (为后续多轮聚合和客户端下发做准备)
        for param in self.model.parameters():
            param.requires_grad = True

    def evaluate(self, **kwargs):
        """
        FedLSA 专用评估：必须对特征归一化后再通过分类器，
        与训练时的流程保持一致（训练时分类器接收的是归一化后的特征）。
        """
        loader = torch.utils.data.DataLoader(
            self.test_set, batch_size=128, shuffle=False
        )
        self.model.to(self.device)
        self.model.eval()
        correct = 0.0
        count = 0.0
        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)
                _, features = self.model(data)
                features_norm = F.normalize(features, p=2, dim=1)
                logits = self.model.classifier(features_norm)
                pred = logits.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                count += target.size(0)
        self.model.cpu()
        self.acc.append(100.0 * correct / count)

    def save(self):
        f = {
            "acc": self.acc,
            "loss": self.loss,
            "state_dict": {
                "global": self.model.state_dict(),
                "client": self.clients_state,
                "proto": self.get_anchors().detach().cpu(),
                "aux": {
                    "R": self.R.cpu(),
                    "anchor_mapping": self.anchor_mapping.state_dict(),
                },
            },
        }
        super().deal_save(f)
