import argparse
import time
import os

import numpy as np
import torch
import torch.nn as nn
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


def get_path(args):
    args.file_name = f"{args.name_pre}_{args.lambda_com}_{args.alpha_sep}_{args.server_epochs}_{args.server_lr}_{args.tau}"
    return os.path.join(args.log_path, f"{args.file_name}.log")


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
    Two-layer MLP mapping function Theta(.) to map random vectors R to anchors A.
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
    FedLSA local training with Location-aware Semantic Anchors (Compactness Loss).
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

            # 4. CE Loss on similarity logits
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

        # 1. Initialize Random Vectors R (learnable)
        # R has the same shape as Anchors: [C, d]
        self.R = torch.randn(self.num_class, self.args.feature_dim, device=self.device)

        # 2. Initialize Mapping Function Theta (MLP)
        self.anchor_mapping = AnchorMapping(self.args.feature_dim).to(self.device)

        self.clients_state = [self.model.state_dict() for _ in range(self.num_clients)]
        self.labels = torch.arange(self.num_class, device=self.device)

    def get_anchors(self):
        """Generate anchors A = Theta(R)"""
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

            # Generate current anchors for distribution
            # Note: We must detach here because clients don't update R or Theta
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
                total_loss += results[i][0]
                self.clients_state[i] = results[i][1]
                selected_states.append(results[i][1])
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
        Optimize Latent Vectors R and Anchor Mapping Function Theta using
        L_LSA = L_ACE + alpha * L_SEP

        NOTE: Global model Phi_glo is FROZEN during this phase and only used for evaluation.
        """
        self.model.eval()  # Set model to eval mode (frozen)
        self.model.to(self.device)
        self.anchor_mapping.train()

        # Joint optimizer for R and Theta
        # We DO NOT include self.model.parameters() here
        if not self.R.requires_grad:
            self.R.requires_grad_(True)

        optimizer = torch.optim.SGD(
            [self.R] + list(self.anchor_mapping.parameters()),
            lr=self.args.server_lr,
        )

        # Temporarily disable gradients for model parameters to ensure they are not updated
        # and to save memory/computation
        for param in self.model.parameters():
            param.requires_grad = False

        print(f"-> Server Optimization for {self.args.server_epochs} epochs...")

        for e in range(self.args.server_epochs):
            # 1. Generate Anchors A = Theta(R)
            anchors = self.get_anchors()

            # Normalize anchors for losses
            anchors_norm = F.normalize(anchors, p=2, dim=1)

            # 2. Compute L_ACE (Adaptive Class Energy Loss)
            # Use frozen global classifier to classify anchors
            logits = self.model.classifier(anchors_norm)
            loss_ace = ce_loss(logits, self.labels)

            # 3. Compute L_SEP (Separation Loss)
            loss_sep = separation_loss(anchors, tau=self.args.tau)

            # Total Server Loss
            loss_lsa = loss_ace + self.args.alpha_sep * loss_sep

            optimizer.zero_grad()
            loss_lsa.backward()
            optimizer.step()

            if e == 0 or (e + 1) == self.args.server_epochs:
                print(
                    f"   Epoch {e + 1}: L_ACE={loss_ace.item():.4f}, L_SEP={loss_sep.item():.4f}"
                )

        # Re-enable gradients for model parameters (for aggregation/client updates if needed)
        for param in self.model.parameters():
            param.requires_grad = True

    def save(self):
        f = {
            "acc": self.acc,
            "loss": self.loss,
            "state_dict": {
                "R": self.R.cpu(),
                "anchor_mapping": self.anchor_mapping.state_dict(),
                "model": self.model.state_dict(),
                "clients": self.clients_state,
            },
        }
        super().deal_save(f)
