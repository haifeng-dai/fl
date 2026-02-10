import argparse
import copy
import time
import os
import torch
import torch.nn.functional as F
import numpy as np

from .utils import BaseServer, ce_loss, get_model, run_parallel_clients


def add_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("FedProc Specific Arguments")
    group.add_argument(
        "--mu", type=float, default=1.0, help="Weight for prototypical contrastive loss"
    )
    group.add_argument(
        "--temperature", type=float, default=0.5, help="Temperature for contrastive loss"
    )
    return parser


def get_path(args):
    args.file_name = f"{args.name_pre}_{args.mu}_{args.temperature}"
    return os.path.join(args.log_path, f"{args.file_name}_{args.times}.log")


def client_worker(params):
    (
        client_id,
        device,
        model_state,
        global_protos,
        train_set,
        model_name,
        dataset_name,
        lr,
        batch_size,
        epochs,
        mu,
        temperature,
        num_classes,
        feature_dim,
    ) = params

    # 1. Initialize Model
    model = get_model(model_name, dataset_name, feature_dim).to(device)
    model.load_state_dict(model_state)
    global_protos = global_protos.data.clone().to(device)

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

            # Prototypical Contrastive Loss
            # Normalize features and prototypes
            features_norm = F.normalize(features, dim=1)
            protos_norm = F.normalize(global_protos, dim=1)

            # InfoNCE-like loss
            # Similarity [Batch, NumClasses]
            logits_con = torch.matmul(features_norm, protos_norm.T) / temperature

            # Target is the class index
            loss_con = F.cross_entropy(logits_con, target)

            loss = loss_ce + mu * loss_con
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

    # 3. Calculate Local Prototypes
    model.eval()
    local_protos = {}
    sum_features = torch.zeros((num_classes, feature_dim), device=device)
    sum_counts = torch.zeros(num_classes, device=device)

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            _, features = model(data)

            sum_features.index_add_(0, target, features)
            sum_counts.index_add_(0, target, torch.ones_like(target, dtype=torch.float32))

    active_classes = torch.where(sum_counts > 0)[0]
    for c in active_classes:
        c_item = int(c.item())
        local_protos[c_item] = (sum_features[c_item] / sum_counts[c_item]).cpu()

    avg_loss = total_loss / num_batches
    model_state = {k: v.cpu() for k, v in model.state_dict().items()}

    return [avg_loss, model_state, local_protos]


class Server(BaseServer):
    def __init__(self, args: argparse.Namespace):
        super().__init__(False, args)
        self.clients_state = [self.model.state_dict() for _ in range(self.num_clients)]
        self.global_protos = torch.zeros((self.num_class, self.args.feature_dim))

    def fit(self):
        num_join_clients = int(self.num_clients * self.args.join_ratio)
        num_join_clients = max(1, num_join_clients)

        for r in range(self.rounds):
            t0 = time.time()
            print(f"\n--- FedProc Round {r + 1}/{self.rounds} ---")

            selected_clients = np.random.choice(
                self.num_clients, num_join_clients, replace=False
            )
            print(f"Selected clients: {selected_clients}")

            p = [
                [
                    i,
                    self.client_gpu[i],
                    self.clients_state[i],
                    self.global_protos,
                    self.train_sets[i],
                    self.args.model,
                    self.args.dataset,
                    self.args.lr,
                    self.args.batch_size,
                    self.args.epochs,
                    self.args.mu,
                    self.args.temperature,
                    self.num_class,
                    self.args.feature_dim,
                ]
                for i in selected_clients
            ]

            p = [
                [
                    i,
                    self.client_gpu[i],
                    self.model.state_dict(),
                    self.global_protos,
                    self.train_sets[i],
                    self.args.model,
                    self.args.dataset,
                    self.args.lr,
                    self.args.batch_size,
                    self.args.epochs,
                    self.args.mu,
                    self.args.temperature,
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
            model_states = []
            all_local_protos = []

            for idx, i in enumerate(selected_clients):
                loss, state, local_protos = results[idx]
                total_loss += loss
                model_states.append(state)
                all_local_protos.append(local_protos)

            self.loss.append(total_loss / num_join_clients)

            # Aggregate Model
            self.aggregate(model_states)

            # Aggregate Prototypes
            self.global_protos = self.aggregate_protos(all_local_protos)

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
            "global_protos": self.global_protos,
        }
        self.deal_save(f)

    def aggregate_protos(self, all_local_protos):
        new_protos = torch.zeros_like(self.global_protos)
        counts = torch.zeros(self.num_class)

        for local_protos in all_local_protos:
            for label, proto in local_protos.items():
                new_protos[label] += proto
                counts[label] += 1

        mask = counts > 0
        new_protos[mask] /= counts[mask].unsqueeze(1)
        # Momentum update could be used, but simple average is standard for basic impl
        new_protos[~mask] = self.global_protos[~mask]

        return new_protos
