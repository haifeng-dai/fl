import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import (
    BaseServer,
    ce_loss,
    cos_contrastive_loss,
    get_model,
    _fmt_num,
)


def get_path(args):
    args.file_name = f"{args.common_name}_{_fmt_num(args.lambda_com)}_{_fmt_num(args.alpha_sep)}_{_fmt_num(args.server_epochs)}_{_fmt_num(args.server_lr)}_{_fmt_num(args.tau)}"
    return os.path.join(args.log_path, f"{args.file_name}_{args.cur_time}.log")


def separation_loss(anchors, tau=0.1):
    """
    计算分离损失 (L_SEP)，公式 (4)：
    L_SEP = log( sum_{j!=i} exp(a_i * a_j^T / tau) / (C-1) )
    """
    C = anchors.shape[0]
    # 确保输入锚点是归一化的
    anchors = F.normalize(anchors, p=2, dim=1)
    # 计算相似度矩阵并除以温度参数
    sim_matrix = torch.matmul(anchors, anchors.T) / tau

    # 掩盖对角线（自身相似度），将其设为负无穷，使 exp(-inf) = 0
    mask = torch.eye(C, device=anchors.device).bool()
    sim_matrix = sim_matrix.masked_fill(mask, float("-inf"))

    # 使用 logsumexp 计算 log(sum(exp(sim)))
    # 然后减去 log(C-1) 以实现对 (C-1) 取平均
    loss_sep = torch.logsumexp(sim_matrix, dim=1) - torch.log(
        torch.tensor(C - 1, device=anchors.device, dtype=sim_matrix.dtype)
    )

    return loss_sep.mean()


class FedLSAModel(nn.Module):
    """
    FedLSA 模型结构。
    显式组合 Backbone 和 Classifier，并在 extractor 方法中注入 L2 归一化。
    """

    def __init__(self, base_model: nn.Module):
        super().__init__()
        # 内部保存原始模块，名称稍作修改以避免与方法名冲突
        self.backbone = base_model.extractor
        self.head = base_model.classifier

    def extractor(self, x):
        """实现公式 (4): h = nor(psi(x))"""
        z = self.backbone(x)
        return F.normalize(z, p=2, dim=1)

    def classifier(self, h):
        return self.head(h)

    def forward(self, x):
        # 此时 self.extractor() 返回的是归一化后的特征 h
        h = self.extractor(x)
        logits = self.classifier(h)
        return logits


class AnchorMapping(nn.Module):
    """
    两层 MLP 映射函数 Theta(.) 用于将随机向量 R (embedding 空间) 映射到语义锚点 A (feature 空间)。
    论文定义: Theta: R^{C x I} -> R^{C x L}
    """

    def __init__(self, embedding_dim: int, feature_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
        )

    def forward(self, x):
        out = self.net(x)
        return F.normalize(out, p=2, dim=-1)


def client_worker(params):
    """
    FedLSA 客户端训练流程。

    严格对齐伪代码 Algorithm 1 (Client Side, Lines 1-13):
      L4:  h_i = nor(φ_m(ψ_m(x_i)))
      L6:  L_CE ← (softmax(ϕ_m(h_i)), y_i)     [公式 (9), 不带 τ]
      L8:  L_COM ← ({a_j}, h_i)                  [公式 (8), 带 τ]
      L9:  L_HC = L_CE + λ * L_COM               [公式 (10)]
    """
    (
        _,
        device,
        model_state,
        train_set,
        global_anchors,
        model_name,
        dataset_name,
        lr,
        batch_size,
        epochs,
        lambda_com,
        tau,
        feature_dim,
    ) = params

    # 1. 初始化模型
    raw_model = get_model(model_name, dataset_name, feature_dim)
    model = FedLSAModel(raw_model)
    model.load_state_dict(model_state)
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    model.train()
    total_loss = 0.0
    num_batches = 0
    global_anchors = global_anchors.to(device)

    # 2. 训练循环 (伪代码 L3-L11)
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            # L4: h = nor(φ(ψ(x)))，extractor 的 hook 已自动完成 L2 归一化
            h = model.extractor(x)
            logits = model.classifier(h)

            # 公式 (9): L_CE = -1_{y_i} log(softmax(q_i))，不带 τ
            loss_ce = ce_loss(logits, y)

            # 公式 (8): L_COM = -log(exp(a_{y_i}^T h_i / τ) / Σ_j exp(a_j^T h_i / τ))
            loss_com = cos_contrastive_loss(h, global_anchors, y, temperature=tau)

            # 公式 (10): L_HC = L_CE + λ * L_COM
            loss = loss_ce + lambda_com * loss_com

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    model_state = {k: v.cpu().detach().clone() for k, v in model.state_dict().items()}
    return {"loss": avg_loss, "state": model_state}


class Server(BaseServer):
    def __init__(self, args):
        super().__init__(False, args)
        self.model = FedLSAModel(self.model).to(self.device)

        # 核心：必须同步更新所有客户端的状态字典，以匹配新的 FedLSAModel 结构 (backbone/head)
        init_state = {
            k: v.cpu().detach().clone() for k, v in self.model.state_dict().items()
        }
        self.clients_state = [init_state for _ in range(self.num_clients)]

        # 1. 初始化随机向量 R (可学习)
        # 论文定义: R ∈ R^{C × I}，在 embedding 空间中
        self.R = torch.randn(self.num_class, args.feature_dim, device=self.device)

        # 2. 初始化映射函数 Theta: R^I -> R^L (从 embedding 空间映射到 feature 空间)
        self.anchor_mapping = AnchorMapping(args.feature_dim, args.feature_dim)
        self.anchor_mapping.to(self.device)
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
            current_anchors = self.get_anchors().detach().cpu()

            p = [
                [
                    i,
                    self.client_gpu[i],
                    self.clients_state[i],
                    self.train_sets[i],
                    current_anchors,
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
            results = self.run_clients(client_worker, p)

            total_loss = 0.0
            selected_states = []
            current_weights = []
            for cid, res in results.items():
                total_loss += res["loss"]
                self.clients_state[cid] = res["state"]
                selected_states.append(res["state"])
                current_weights.append(self.weights[cid])
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
        伪代码 Algorithm 1 (Server Side, Lines 14-27):
          L17: A = Θ(R)
          L19: L_ACE ← (softmax(ϕ_glo(a^i)), y_i)  [公式 (3), 不带 τ]
          L21: L_SEP ← ({a_j})                       [公式 (4), 带 τ]
          L22: L_LSA = L_ACE + α * L_SEP              [公式 (5)]
          L23-24: 更新 R 和 Θ
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
            # L17: 生成语义锚点 A = Theta(R)
            anchors = self.get_anchors()

            # 公式 (3): L_ACE = -1_{y_i} log(softmax(ρ_i))
            # 其中 ρ_i = ϕ_glo(a_i)，直接将锚点喂入冻结分类器，不带 τ
            logits = self.model.classifier(anchors)
            loss_ace = ce_loss(logits, self.labels)

            # 公式 (4): L_SEP，带 τ
            loss_sep = separation_loss(anchors, tau=self.args.tau)

            # 公式 (5): L_LSA = L_ACE + α * L_SEP
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

    def save(self):
        f = {
            "acc": self.acc,
            "loss": self.loss,
            "state_dict": {
                "global": self.model.state_dict(),
                "proto": self.get_anchors().detach().cpu(),
            },
        }
        self.deal_save(f)
