import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F

from .utils import (
    BaseServer,
    ce_loss,
    get_model,
    param_aggregate,
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


def client_worker(params):
    """
    FedLSA local training with Location-aware Semantic Anchors (Compactness Loss).
    """
    (
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
    ) = params

    model = get_model(model_name, dataset_name).to(device)
    model.load_state_dict(model_state)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    model.train()
    total_loss = 0.0
    num_batches = 0

    # Ensure anchors are on the correct device and detached (fixed during client training)
    global_anchors = global_anchors.to(device).detach()

    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, features = model(x)
            # L_CE: Standard Cross Entropy Loss
            loss_ce = ce_loss(logits, y)

            # L_COM: Compactness Loss using Softmax with Temperature
            # 1. Normalize features and anchors
            features_norm = F.normalize(features, p=2, dim=1)
            anchors_norm = F.normalize(global_anchors, p=2, dim=1)

            # 2. Compute Cosine Similarity Matrix [Batch, NumClasses]
            logits_com = torch.matmul(features_norm, anchors_norm.T)

            # 3. Apply Temperature scaling
            logits_com = logits_com / tau

            # 4. CE Loss on similarity logits (encourages feature to be close to its class anchor)
            loss_com = ce_loss(logits_com, y)

            # Total Loss: L_HC = L_CE + lambda * L_COM
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

        self.anchors = torch.randn(self.num_class, self.model.feature_dim).to(
            self.device
        )
        self.anchors = (
            F.normalize(self.anchors, p=2, dim=1).detach().requires_grad_(True)
        )
        self.clients_state = [self.model.state_dict() for _ in range(self.num_clients)]

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

            p = [
                [
                    self.client_gpu[i],
                    self.clients_state[i],
                    self.anchors.detach().cpu(),
                    self.train_sets[i],
                    self.args.model,
                    self.args.dataset,
                    self.args.lr,
                    self.args.batch_size,
                    self.args.epochs,
                    self.args.lambda_com,
                    self.args.tau,
                ]
                for i in selected_clients
            ]

            results = run_parallel_clients(
                client_worker=client_worker,
                num_clients=num_join_clients,
                parameters=p,
                gpu_pools=self.gpu_pools,
                mp=self.mp,
            )

            losses = [res[0] for res in results]
            new_client_states = [res[1] for res in results]

            avg_loss = sum(losses) / len(losses)
            self.loss.append(avg_loss)

            # Update local states for selected clients
            for i, state in enumerate(new_client_states):
                client_idx = selected_clients[i]
                self.clients_state[client_idx] = state

            # Calculate weights for selected clients
            current_weights = [self.weights[i] for i in selected_clients]
            sum_weights = sum(current_weights)
            norm_weights = [w / sum_weights for w in current_weights]

            # 聚合得到初步的全局模型
            self.model.load_state_dict(param_aggregate(new_client_states, norm_weights))

            # 步骤 4: 服务器端优化
            # 优化全局模型 (Theta) 和锚点 (R)
            self.server_optimization()

            self.evaluate()
            print(f"Global Accuracy: {self.acc[-1]:.2f}%, Avg Loss: {avg_loss:.4f}")
            print(f"Round finished in {time.time() - t0:.2f} seconds")

    def server_optimization(self):
        """
        使用 L_LSA = L_ACE + alpha * L_SEP 优化全局模型和锚点
        """
        self.model.train()
        self.model.to(self.device)

        # 确保锚点计算梯度
        if not self.anchors.requires_grad:
            self.anchors.requires_grad_(True)

        # 联合优化器：优化模型参数和锚点
        optimizer = torch.optim.SGD(
            list(self.model.parameters()) + [self.anchors], lr=self.args.server_lr
        )

        print(f"-> Server Optimization for {self.args.server_epochs} epochs...")

        for e in range(self.args.server_epochs):
            anchors_norm = F.normalize(self.anchors, p=2, dim=1)
            # 分类器输出 Logits
            logits = self.model.classifier(anchors_norm)

            # 锚点的标签就是 0, 1, ..., C-1
            labels = torch.arange(self.num_class, device=self.device)

            # 1. L_ACE: 锚点分类误差
            loss_ace = ce_loss(logits, labels)

            # 2. L_SEP: 分离损失
            loss_sep = separation_loss(self.anchors, tau=self.args.tau)

            # 总的服务器端损失
            loss_lsa = loss_ace + self.args.alpha_sep * loss_sep

            optimizer.zero_grad()
            loss_lsa.backward()
            optimizer.step()

            if e == 0 or (e + 1) == self.args.server_epochs:
                print(
                    f"   Epoch {e + 1}: L_ACE={loss_ace.item():.4f}, L_SEP={loss_sep.item():.4f}"
                )

        # 优化结束后 detach，防止计算图无限增长
        self.anchors = self.anchors.detach()

    def save(self):
        file_name = f"{self.args.lambda_com}_{self.args.alpha_sep}_{self.args.server_epochs}_{self.args.server_lr}_{self.args.tau}"
        f = {
            "acc": self.acc,
            "loss": self.loss,
            "state_dict": {
                "anchors": self.anchors.data,
                "clients": self.model.state_dict(),
            },
        }
        super().deal_save(f, file_name)
