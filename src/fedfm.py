import argparse
import copy
import time
import os
import torch
import torch.nn.functional as F
import numpy as np

from .utils import BaseServer, ce_loss, get_model, run_parallel_clients, mse_loss


def add_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("FedFM Specific Arguments")
    group.add_argument(
        "--mu", type=float, default=1.0, help="Weight for contrastive guiding loss"
    )
    return parser


def get_path(args):
    args.file_name = f"{args.name_pre}_{args.mu}"
    return os.path.join(args.log_path, f"{args.file_name}_{args.times}.log")


def client_worker(params):
    (
        client_id,
        device,
        model_state,
        global_anchors,
        train_set,
        model_name,
        dataset_name,
        lr,
        batch_size,
        epochs,
        mu,
        num_classes,
        feature_dim,
    ) = params

    # 1. Initialize Model
    model = get_model(model_name, dataset_name, feature_dim).to(device)
    model.load_state_dict(model_state)
    global_anchors = global_anchors.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    # 2. Train Model
    total_loss = 0.0
    num_batches = 0

    model.train()
    for _ in range(epochs):
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output, features = model(data)

            # Cross Entropy Loss
            loss_ce = ce_loss(output, target)

            # MSE Loss between features and corresponding class anchors
            target_anchors = global_anchors[target]
            loss_cg = mse_loss(features, target_anchors)

            loss = loss_ce + mu * loss_cg
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

    # 3. Calculate Local Anchors (Average features per class)
    model.eval()
    local_anchors = {}

    sum_features = torch.zeros((num_classes, feature_dim), device=device)
    sum_counts = torch.zeros(num_classes, device=device)

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            _, features = model(data)

            sum_features.index_add_(0, target, features)
            sum_counts.index_add_(
                0, target, torch.ones_like(target, dtype=torch.float32)
            )

    # Average and convert to CPU
    active_classes = torch.where(sum_counts > 0)[0]
    for c in active_classes:
        c_item = int(c.item())
        # We store raw features, normalization happens in loss
        local_anchors[c_item] = (sum_features[c_item] / sum_counts[c_item]).cpu()

    avg_loss = total_loss / num_batches
    model_state = {k: v.cpu() for k, v in model.state_dict().items()}

    return [avg_loss, model_state, local_anchors]


class Server(BaseServer):
    def __init__(self, args: argparse.Namespace):
        super().__init__(False, args)
        self.clients_state = [self.model.state_dict() for _ in range(self.num_clients)]
        self.global_anchors = torch.zeros((self.num_class, self.args.feature_dim))

    def fit(self):
        num_join_clients = int(self.num_clients * self.args.join_ratio)
        num_join_clients = max(1, num_join_clients)

        for r in range(self.rounds):
            t0 = time.time()
            print(f"\n--- FedFM Round {r + 1}/{self.rounds} ---")

            selected_clients = np.random.choice(
                self.num_clients, num_join_clients, replace=False
            )
            print(f"Selected clients: {selected_clients}")

            p = [
                [
                    i,
                    self.client_gpu[i],
                    self.model.state_dict(),
                    self.global_anchors,
                    self.train_sets[i],
                    self.args.model,
                    self.args.dataset,
                    self.args.lr,
                    self.args.batch_size,
                    self.args.epochs,
                    self.args.mu,
                    self.num_class,
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
            all_local_anchors = []

            for i in selected_clients:
                client_loss, client_state, client_anchor = results[i]
                total_loss += client_loss
                selected_states.append(client_state)
                all_local_anchors.append(client_anchor)
            self.loss.append(total_loss / num_join_clients)
            self.aggregate(selected_states)

            # Aggregate Anchors
            self.aggregate_anchors(all_local_anchors)

            self.evaluate()
            print(
                f"Global Accuracy: {self.acc[-1]:.2f}%, Avg Loss: {self.loss[-1]:.4f}"
            )
            print(f"Round finished in {time.time() - t0:.2f} seconds")

    def save(self):
        f = {
            "acc": self.acc,
            "loss": self.loss,
            "state_dict": self.model.state_dict(),
            "global_anchors": self.global_anchors,
        }
        self.deal_save(f)

    def aggregate_anchors(self, all_local_protos):
        """
        Aggregate local prototypes (dicts) into a single global prototype Tensor.
        """
        counts = torch.zeros(self.num_class, dtype=torch.float32)
        for local_protos in all_local_protos:
            for label, proto in local_protos.items():
                if counts[label] == 0:
                    self.global_anchors[label].zero_()
                self.global_anchors[label] += proto
                counts[label] += 1

        # Average
        mask = counts > 0
        self.global_anchors[mask] /= counts[mask].unsqueeze(1)
