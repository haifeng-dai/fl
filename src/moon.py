import argparse
import copy
import time, os

import numpy as np
import torch

from .utils import (
    BaseServer,
    ce_loss,
    get_model,
    run_parallel_clients,
)


def add_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("MOON Specific Arguments")
    group.add_argument(
        "--mu", type=float, default=1.0, help="Weight for contrastive loss"
    )
    group.add_argument(
        "--tau",
        type=float,
        default=0.5,
        help="Temperature parameter for contrastive loss",
    )
    return parser


def get_path(args):
    args.file_name = f"{args.name_pre}_{args.mu}_{args.tau}"
    return os.path.join(args.log_path, f"{args.file_name}_{args.times}.log")


def client_worker(params):
    """
    MOON local training with Model-Contrastive Loss.
    """
    (
        _,
        device,
        global_state,
        prev_state,
        train_set,
        model_name,
        dataset_name,
        lr,
        batch_size,
        epochs,
        mu,
        tau,
        feature_dim,
    ) = params

    # 1. Initialize current local model with global state
    model = get_model(model_name, dataset_name, feature_dim).to(device)
    model.load_state_dict(global_state)

    # 2. Initialize global model (frozen) for contrastive loss
    global_model = get_model(model_name, dataset_name, feature_dim).to(device)
    global_model.load_state_dict(global_state)
    global_model.eval()

    # 3. Initialize previous local model (frozen) for contrastive loss
    prev_model = get_model(model_name, dataset_name, feature_dim).to(device)
    prev_model.load_state_dict(prev_state)
    prev_model.eval()

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    ce_moon = torch.nn.CosineSimilarity(dim=-1)

    total_loss = 0.0
    num_batches = 0
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            # Forward pass: get output and representation (z)
            output, z = model(x)
            with torch.no_grad():
                _, z_glob = global_model(x)
                _, z_prev = prev_model(x)

            # Standard Cross Entropy Loss
            loss_ce = ce_loss(output, y)

            # MOON Contrastive Loss
            # Similarity with global model (positive) and previous local model (negative)
            pos_sim = ce_moon(z, z_glob)
            neg_sim = ce_moon(z, z_prev)
            logits = torch.cat([pos_sim.reshape(-1, 1), neg_sim.reshape(-1, 1)], dim=1)
            logits /= tau
            labels = torch.zeros(z.size(0)).to(device).long()
            loss_con = ce_loss(logits, labels)

            # Total Loss
            loss = loss_ce + mu * loss_con
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    model_state = {k: v.cpu() for k, v in model.state_dict().items()}
    return [avg_loss, model_state]


class Server(BaseServer):
    def __init__(self, args: argparse.Namespace):
        super().__init__(False, args)
        # Initialize previous model states for all clients with the initial global model
        self.clients_state = [self.model.state_dict() for _ in range(self.num_clients)]

    def fit(self):
        num_join_clients = int(self.num_clients * self.args.join_ratio)
        num_join_clients = max(1, num_join_clients)

        for r in range(self.rounds):
            t0 = time.time()
            print(f"\n--- MOON Round {r + 1}/{self.rounds} ---")

            selected_clients = np.random.choice(
                self.num_clients, num_join_clients, replace=False
            )
            print(f"Selected clients: {selected_clients}")

            p = [
                [
                    i,
                    self.client_gpu[i],
                    self.model.state_dict(),
                    self.clients_state[i],
                    self.train_sets[i],
                    self.args.model,
                    self.args.dataset,
                    self.args.lr,
                    self.args.batch_size,
                    self.args.epochs,
                    self.args.mu,
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

            # Calculate average loss using incremental summation
            total_loss = 0.0
            selected_states = []
            current_weights = []
            for i in selected_clients:
                client_loss, client_state = results[i]
                total_loss += client_loss
                selected_states.append(client_state)
                self.clients_state[i] = client_state
                current_weights.append(self.weights[i])
            self.loss.append(total_loss / num_join_clients)
            sum_weights = sum(current_weights)
            norm_weights = [w / sum_weights for w in current_weights]

            self.aggregate(selected_states, weights=norm_weights)
            self.evaluate()
            print(
                f"Global Accuracy: {self.acc[-1]:.2f}%, Avg Loss: {self.loss[-1]:.4f}"
            )
            print(f"Round finished in {time.time() - t0:.2f} seconds")

    def save(self):
        f = {"acc": self.acc, "loss": self.loss, "state_dict": self.model.state_dict()}
        super().deal_save(f)
