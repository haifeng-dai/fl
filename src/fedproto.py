import argparse
import copy
import time, os

import numpy as np
import torch

from src.utils import evaluate_prototype, mse_loss

from .utils import (
    BaseServer,
    ce_loss,
    evaluate_model,
    get_model,
    run_parallel_clients,
)


def add_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("FedProto Specific Arguments")
    group.add_argument(
        "--mu", type=float, default=1.0, help="Weight for prototype loss"
    )
    return parser


def get_path(args):
    args.file_name = f"{args.name_pre}_{args.mu}"
    return os.path.join(args.log_path, f"{args.file_name}_{args.times}.log")


import torch.nn.functional as F


def client_worker(params):
    """
    FedProto local training with prototype regularization.
    """
    (
        _,
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
        num_classes,
        feature_dim,
    ) = params

    # 1. Initialize Model
    model = get_model(model_name, dataset_name, feature_dim).to(device)
    model.load_state_dict(model_state)

    # Pre-move global prototypes to GPU to avoid frequent data transfer
    # global_protos is now a Tensor [C, D]
    global_protos = global_protos.to(device)
    # Pre-compute valid mask [C] to avoid O(BxD) norm calc in loop
    # Check which classes have non-zero prototypes
    proto_sums = torch.sum(torch.abs(global_protos), dim=1)
    valid_proto_mask = proto_sums > 1e-6

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    # 2. Train Model
    total_loss = 0.0
    num_batches = 0
    for _ in range(epochs):
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output, features = model(data)
            loss_ce = ce_loss(output, target)

            # Prototype Loss: Regularize features towards global
            mask = valid_proto_mask[target]

            loss_proto = 0.0
            if mask.any():
                # Filter input features and target prototypes
                features_filtered = features[mask]
                target_filtered = target[mask]
                protos_filtered = global_protos[target_filtered]

                loss_proto = mse_loss(features_filtered, protos_filtered)

            loss = loss_ce + mu * loss_proto
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

    # 3. Calculate Local Prototypes (Average features per class)
    model.eval()
    local_protos: dict[int, torch.Tensor] = {}

    # Initialize tensors on GPU for accumulation to avoid Python loops
    sum_protos = torch.zeros((num_classes, feature_dim), device=device)
    sum_counts = torch.zeros(num_classes, device=device)

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            _, features = model(data)

            # Use index_add_ for fast vectorized summation
            sum_protos.index_add_(0, target, features)
            sum_counts.index_add_(
                0, target, torch.ones_like(target, dtype=torch.float32, device=device)
            )

    # Average and convert to CPU dictionary
    active_classes: torch.Tensor = torch.where(sum_counts > 0)[0]
    for c in active_classes:
        c_item = int(c.item())
        local_protos[c_item] = (sum_protos[c_item] / sum_counts[c_item]).cpu()

    # Return avg_loss, new_model_state, local_protos
    avg_loss = total_loss / num_batches
    model_state = {k: v.cpu() for k, v in model.state_dict().items()}
    return [avg_loss, model_state, local_protos]


class Server(BaseServer):
    def __init__(self, args: argparse.Namespace):
        super().__init__(True, args)
        # Initialize personalized models for each client
        self.clients_state = [self.model.state_dict() for _ in range(self.num_clients)]
        # Global prototypes stored as a Tensor [num_classes, feature_dim] on CPU, or None if not available
        self.global_protos = torch.zeros((self.num_class, self.args.feature_dim), dtype=torch.float32)
        self.acc_p: list[float] = []

    def fit(self):
        num_join_clients = int(self.num_clients * self.args.join_ratio)
        num_join_clients = max(1, num_join_clients)

        for r in range(self.rounds):
            t0 = time.time()
            print(f"\n--- FedProto Round {r + 1}/{self.rounds} ---")

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
                    self.num_class,
                    self.args.feature_dim,
                ]
                for i in selected_clients
            ]

            t_train_start = time.time()
            results = run_parallel_clients(
                client_worker=client_worker,
                parameters=p,
                gpu_pools=self.gpu_pools,
                mp=self.mp,
            )
            print(f"  [Time] Parallel Training: {time.time() - t_train_start:.2f}s")

            total_loss = 0.0
            all_local_protos = []
            for i in selected_clients:
                total_loss += results[i][0]
                self.clients_state[i] = results[i][1]
                all_local_protos.append(results[i][2])
            self.loss.append(total_loss / num_join_clients)
            self.aggregate_protos(all_local_protos)

            self.evaluate()
            print(
                f"Global Accuracy: {self.acc[-1]:.2f}%, "
                f"Proto Accuracy: {self.acc_p[-1]:.2f}%, "
                f"Avg Loss: {self.loss[-1]:.4f}"
            )
            print(f"Round finished in {time.time() - t0:.2f} seconds")

    def aggregate_protos(self, all_local_protos):
        """
        Aggregate local prototypes (dicts) into a single global prototype Tensor.
        Returns: torch.Tensor [num_classes, feature_dim] (CPU)
        """
        counts = torch.zeros(self.num_class, dtype=torch.float32)
        for local_protos in all_local_protos:
            for label, proto in local_protos.items():
                if counts[label] == 0:
                    self.global_protos[label].zero_()

                # proto is typically CPU tensor from client_worker
                self.global_protos[label] += proto
                counts[label] += 1

        # Average
        mask = counts > 0
        self.global_protos[mask] /= counts[mask].unsqueeze(1)

    def evaluate(self):
        # Evaluate each client's personalized model on its local test set
        accs = []
        acc_ps = []
        for i in range(self.num_clients):
            self.model.load_state_dict(self.clients_state[i])
            acc = evaluate_model(self.model, self.test_set[i], self.device)
            acc_p = evaluate_prototype(
                self.model, self.global_protos, self.test_set[i], self.device
            )
            accs.append(acc)
            acc_ps.append(acc_p)

        self.acc.append(sum(accs) / self.num_clients)
        self.acc_p.append(sum(acc_ps) / self.num_clients)
        self.model.cpu()

    def save(self):
        f = {
            "acc": {"model": self.acc, "proto": self.acc_p},
            "loss": self.loss,
            "state_dict": {
                "model": self.clients_state,
                "proto": self.global_protos,
            },
        }
        self.deal_save(f)
