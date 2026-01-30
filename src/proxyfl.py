import argparse
import copy
import time, os

import numpy as np
import torch

from .utils import (
    BaseServer,
    ce_loss,
    evaluate_model,
    get_model,
    kl_loss,
    param_aggregate,
    run_parallel_clients,
)


def add_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("ProxyFL Specific Arguments")
    group.add_argument(
        "--mu",
        type=float,
        default=1.0,
        help="Weight for Mutual Learning Distillation",
    )
    group.add_argument(
        "--adj_type",
        type=str,
        default="ring",
        choices=["ring", "centralized"],
        help="Topology of the decentralized network",
    )
    return parser


def get_path(args):
    args.file_name = f"{args.name_pre}_{args.mu}_{args.adj_type}"
    return os.path.join(args.log_path, f"{args.file_name}.log")


def client_worker(params):
    """
    ProxyFL local training with mutual distillation between private local model and shared proxy model.
    """
    (
        _,
        device,
        proxy_state,
        local_state,
        train_set,
        model_name,
        dataset_name,
        lr,
        batch_size,
        epochs,
        mu,
        feature_dim,
    ) = params

    # 1. Initialize Proxy Model (Shared/Public)
    proxy_model = get_model(model_name, dataset_name, feature_dim).to(device)
    proxy_model.load_state_dict(proxy_state)

    # 2. Initialize Local Model (Private/Personalized)
    local_model = get_model(model_name, dataset_name, feature_dim).to(device)
    local_model.load_state_dict(local_state)

    # Optimizers
    # Usually ProxyFL allows different LRs, but we use the same for simplicity unless specified
    opt_p = torch.optim.SGD(proxy_model.parameters(), lr=lr)
    opt_l = torch.optim.SGD(local_model.parameters(), lr=lr)

    loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    proxy_model.train()
    local_model.train()

    total_loss_p = 0.0
    total_loss_l = 0.0
    num_batches = 0

    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            # Forward pass
            out_p, _ = proxy_model(x)
            out_l, _ = local_model(x)

            # Cross Entropy Loss
            ce_p = ce_loss(out_p, y)
            ce_l = ce_loss(out_l, y)

            # Mutual Distillation (KL Divergence)
            # KL(Local || Proxy) -> Proxy learns from Local (to aggregate info)
            loss_kl_p = kl_loss(out_p, out_l.detach())

            # KL(Proxy || Local) -> Local learns from Proxy (to gain global info)
            loss_kl_l = kl_loss(out_l, out_p.detach())

            loss_p = ce_p + mu * loss_kl_p
            loss_l = ce_l + mu * loss_kl_l

            # Update Proxy
            opt_p.zero_grad()
            loss_p.backward()
            opt_p.step()

            # Update Local
            opt_l.zero_grad()
            loss_l.backward()
            opt_l.step()

            total_loss_p += loss_p.item()
            total_loss_l += loss_l.item()
            num_batches += 1

    avg_loss_p = total_loss_p / num_batches
    avg_loss_l = total_loss_l / num_batches

    proxy_state = {k: v.cpu() for k, v in proxy_model.state_dict().items()}
    local_state = {k: v.cpu() for k, v in local_model.state_dict().items()}
    return [avg_loss_p, avg_loss_l, proxy_state, local_state]


class Server(BaseServer):
    def __init__(self, args: argparse.Namespace):
        super().__init__(True, args)

        # Initialize local models for each client (Private)
        self.client_states = [self.model.state_dict() for _ in range(self.num_clients)]
        # Initialize proxy models for each client (Public/Shared)
        self.client_states_p = [
            self.model.state_dict() for _ in range(self.num_clients)
        ]

        self.loss_p = []
        self.adj_matrix = self.generate_adj_matrix()

    def generate_adj_matrix(self):
        num_clients = self.num_clients
        adj = torch.zeros(num_clients, num_clients)
        if self.args.adj_type == "ring":
            for i in range(num_clients):
                adj[i, i] = 1.0
                adj[i, (i - 1) % num_clients] = 1.0
                adj[i, (i + 1) % num_clients] = 1.0
        elif self.args.adj_type == "centralized":
            adj.fill_(1.0)

        # Normalize weights for each client
        row_sums = adj.sum(dim=1, keepdim=True)
        adj = adj / row_sums
        return adj

    def fit(self):
        num_join_clients = int(self.num_clients * self.args.join_ratio)
        num_join_clients = max(1, num_join_clients)

        for r in range(self.rounds):
            t0 = time.time()
            print(f"\n--- ProxyFL Round {r + 1}/{self.rounds} ---")

            selected_clients = np.random.choice(
                self.num_clients, num_join_clients, replace=False
            )
            print(f"Selected clients: {selected_clients}")

            # 1. Neighbor Aggregation for each client
            # Each client aggregates models from its neighbors based on the adjacency matrix
            def get_client_param(i):
                # Identify neighbors and their weights
                neighbor_indices = torch.where(self.adj_matrix[i] > 0)[0].tolist()
                neighbor_weights = self.adj_matrix[i, neighbor_indices].tolist()

                # Perform local aggregation
                # Note: We access self.proxy_model_states which contains the latest available states (possibly from previous rounds for non-active clients)
                neighbor_states = [self.client_states_p[j] for j in neighbor_indices]
                aggregated_proxy_state = param_aggregate(
                    neighbor_states, neighbor_weights
                )

                return [
                    i,
                    self.client_gpu[i],
                    aggregated_proxy_state,
                    self.client_states[i],
                    self.train_sets[i],
                    self.args.model,
                    self.args.dataset,
                    self.args.lr,
                    self.args.batch_size,
                    self.args.epochs,
                    self.args.mu,
                    self.args.feature_dim,
                ]

            p = [get_client_param(i) for i in selected_clients]

            # 2. Parallel Client Training
            results = run_parallel_clients(
                client_worker=client_worker,
                parameters=p,
                gpu_pools=self.gpu_pools,
                mp=self.mp,
            )

            # 3. Update states and calculate average losses
            total_loss = 0.0
            total_loss_p = 0.0
            selected_states = []
            selected_states_p = []
            current_weights = []
            for i in selected_clients:
                total_loss += results[i][0]
                total_loss_p += results[i][1]
                selected_states.append(results[i][2])
                self.client_states[i] = results[i][2]
                selected_states_p.append(results[i][3])
                self.client_states_p[i] = results[i][3]
                current_weights.append(self.weights[i])
            self.loss.append(total_loss / num_join_clients)
            self.loss_p.append(total_loss_p / num_join_clients)

            # 4. Evaluation (using local personalized models)
            self.evaluate()
            print(
                f"Avg Local Accuracy: {self.acc[-1]:.2f}%, Avg Local Loss: {self.loss[-1]:.4f}"
            )
            print(f"Round finished in {time.time() - t0:.2f} seconds")

    def evaluate(self):
        # Evaluate Personalized Local Models on Local Test Sets
        accs = []
        for i in range(self.num_clients):
            self.model.load_state_dict(self.client_states[i])
            acc = evaluate_model(self.model, self.test_set[i], self.device)
            accs.append(acc)

        avg_acc = sum(accs) / len(accs)
        self.acc.append(avg_acc)

    def save(self):
        f = {
            "acc": self.acc,
            "loss": {"model": self.loss, "proxy": self.loss_p},
            "state_dict": self.client_states,
        }
        super().deal_save(f)
