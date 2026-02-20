import argparse, os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .utils import BaseServer, get_model, run_parallel_clients, mse_loss, ce_loss


def add_args(parser: argparse.ArgumentParser):
    """Add FedTGP specific arguments"""
    group = parser.add_argument_group("FedTGP Specific Arguments")
    group.add_argument(
        "--lamda_",
        type=float,
        default=10.0,
        help="Weight for prototype matching loss (default: 10.0)",
    )
    group.add_argument(
        "--server_epochs",
        type=int,
        default=10,
        help="Number of server-side TGP training epochs (default: 10)",
    )
    group.add_argument(
        "--server_lr",
        type=float,
        default=0.01,
        help="Learning rate for server-side TGP training (default: 0.01)",
    )
    group.add_argument(
        "--margin_threshold",
        type=float,
        default=1.0,
        help="Margin threshold for TGP training (default: 1.0)",
    )
    return parser


def get_path(args):
    args.file_name = f"{args.name_pre}_{args.lamda_}_{args.server_epochs}_{args.server_lr}_{args.margin_threshold}"
    return os.path.join(args.log_path, f"{args.file_name}_{args.times}.log")


class TGP(nn.Module):
    """Trainable Global Prototypes module"""

    def __init__(self, num_classes, hidden_dim, feature_dim, device):
        super().__init__()
        self.device = device
        self.num_classes = num_classes

        self.embeddings = nn.Embedding(num_classes, feature_dim)
        self.middle = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.ReLU())
        self.fc = nn.Linear(hidden_dim, feature_dim)

    def forward(self, class_ids):
        """
        Args:
            class_ids: tensor of class indices or list of class indices
        """
        if isinstance(class_ids, list):
            class_ids = torch.tensor(class_ids, device=self.device)
        elif not isinstance(class_ids, torch.Tensor):
            class_ids = torch.tensor(class_ids, device=self.device)

        class_ids = class_ids.to(self.device)

        emb = self.embeddings(class_ids)
        mid = self.middle(emb)
        out = self.fc(mid)
        return out


def proto_cluster(protos_list):
    """Cluster prototypes from multiple clients"""
    proto_clusters = defaultdict(list)
    for protos in protos_list:
        for k, v in protos.items():
            proto_clusters[k].append(v)

    avg_protos = {}
    for k, v in proto_clusters.items():
        protos = torch.stack(v)
        avg_protos[k] = torch.mean(protos, dim=0).detach()

    return avg_protos


def client_worker(params):
    """
    FedTGP client worker with prototype-based training.
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
        lamda_,
        global_protos,
        num_classes,
        feature_dim,
    ) = params

    # Initialize model
    model = get_model(model_name, dataset_name, feature_dim).to(device)
    model.load_state_dict(model_state)

    # Setup
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    model.train()
    total_loss = 0.0
    num_batches = 0

    # Pre-process global prototypes for efficient GPU access
    global_protos_tensor = None
    if global_protos is not None:
        # Create a tensor of shape [num_classes, feature_dim]
        # We need to determine feature_dim from the first available prototype
        first_proto = next(iter(global_protos.values()))
        feature_dim = first_proto.shape[0]
        global_protos_tensor = torch.zeros(num_classes, feature_dim, device=device)

        for label, proto in global_protos.items():
            global_protos_tensor[label] = proto.to(device)

    # Local training
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            output, features = model(x)
            loss = ce_loss(output, y)

            # Prototype matching loss
            if global_protos_tensor is not None:
                target_protos = global_protos_tensor[y]
                loss += mse_loss(features, target_protos) * lamda_

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

    # Collect local prototypes
    model.eval()

    # Initialize accumulators on device for vectorized operation
    proto_sum = torch.zeros(num_classes, feature_dim, device=device)
    proto_count = torch.zeros(num_classes, device=device)

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            _, features = model(x)

            proto_sum.index_add_(0, y, features)
            ones = torch.ones_like(y, dtype=torch.float)
            proto_count.index_add_(0, y, ones)

    # Calculate average prototypes
    local_protos_avg = {}
    # Only process classes that appeared in the local dataset
    present_classes = torch.nonzero(proto_count).squeeze(1)

    for cls_idx in present_classes:
        # Calculate mean: sum / count
        avg = proto_sum[cls_idx] / proto_count[cls_idx]
        local_protos_avg[cls_idx.item()] = avg.cpu()

    # Return results (move to CPU)
    model_state = {k: v.cpu() for k, v in model.state_dict().items()}

    return [avg_loss, model_state, local_protos_avg]


class Server(BaseServer):
    def __init__(self, args: argparse.Namespace):
        # FedTGP is a personalized FL method
        super().__init__(True, args)
        self.clients_state = [self.model.state_dict() for _ in range(self.num_clients)]

        # Use model's actual feature dimension if available, otherwise use args
        if hasattr(self.model, "feature_dim"):
            self.feature_dim = self.model.feature_dim
        else:
            self.feature_dim = args.feature_dim

        # Initialize TGP module
        self.tgp = TGP(
            num_classes=self.num_class,
            hidden_dim=self.feature_dim,
            feature_dim=self.feature_dim,
            device=self.device,
        ).to(self.device)

        self.global_protos = None
        self.gap = torch.ones(self.num_class, device=self.device) * 1e9

    def fit(self):
        num_join_clients = int(self.num_clients * self.args.join_ratio)
        num_join_clients = max(1, num_join_clients)

        for r in range(self.rounds):
            t0 = time.time()
            print(f"\n--- FedTGP Round {r + 1}/{self.rounds} ---")

            # Select clients
            selected_clients = np.random.choice(
                self.num_clients, num_join_clients, replace=False
            )
            print(f"Selected clients: {selected_clients}")

            # Prepare parameters for parallel execution
            global_protos_cpu = None
            if self.global_protos is not None:
                global_protos_cpu = {k: v.cpu() for k, v in self.global_protos.items()}

            p = [
                [
                    i,
                    self.client_gpu[i],
                    self.clients_state[i],
                    self.train_sets[i],
                    self.args.model,
                    self.args.dataset,
                    self.args.lr,
                    self.args.batch_size,
                    self.args.epochs,
                    self.args.lamda_,
                    global_protos_cpu,
                    self.num_class,
                    self.args.feature_dim,
                ]
                for i in selected_clients
            ]

            # Run parallel client training
            results = run_parallel_clients(
                client_worker=client_worker,
                parameters=p,
                gpu_pools=self.gpu_pools,
                mp=self.mp,
            )

            # Process results
            total_loss = 0.0
            selected_states = []
            selected_protos = []
            for i in selected_clients:
                client_loss, client_state, client_proto = results[i]
                total_loss += client_loss
                self.clients_state[i] = client_state
                selected_states.append(client_state)
                selected_protos.append(client_proto)
            self.loss.append(total_loss / num_join_clients)

            uploaded_protos = []
            for p in selected_protos:
                for label, proto in p.items():
                    uploaded_protos.append((proto.to(self.device), label))

            # Calculate class-wise minimum distance (gap)
            self.calculate_gap(selected_protos)

            # Update TGP on server
            self.update_tgp(uploaded_protos)

            # Evaluate personalized models
            self.evaluate_personalized()

            print(
                f"Personalized Accuracy: {self.acc[-1]:.2f}%, Avg Loss: {self.loss[-1]:.4f}"
            )
            print(f"Round finished in {time.time() - t0:.2f} seconds")

    def calculate_gap(self, protos_per_client):
        """Calculate class-wise minimum distance between prototypes"""
        self.gap = torch.ones(self.num_class, device=self.device) * 1e9

        # Average prototypes across clients
        avg_protos = proto_cluster(protos_per_client)

        # Calculate pairwise distances
        for k1 in avg_protos.keys():
            for k2 in avg_protos.keys():
                if k1 > k2:
                    dis = torch.norm(
                        avg_protos[k1].to(self.device) - avg_protos[k2].to(self.device),
                        p=2,
                    )
                    self.gap[k1] = torch.min(self.gap[k1], dis)
                    self.gap[k2] = torch.min(self.gap[k2], dis)

        min_gap = torch.min(self.gap)
        for i in range(len(self.gap)):
            if self.gap[i] > 1e8:
                self.gap[i] = min_gap

        max_gap = torch.max(self.gap)
        # print(f"Class-wise minimum distance: {self.gap.cpu().numpy()}")
        print(f"Min gap: {min_gap:.4f}, Max gap: {max_gap:.4f}")

    def update_tgp(self, uploaded_protos):
        """Update Trainable Global Prototypes"""
        self.tgp.train()
        optimizer = torch.optim.SGD(self.tgp.parameters(), lr=self.args.server_lr)

        for epoch in range(self.args.server_epochs):
            proto_loader = DataLoader(
                uploaded_protos,
                batch_size=self.args.batch_size,
                shuffle=True,
                drop_last=False,
            )

            epoch_loss = 0.0
            num_batches = 0

            for proto_batch, labels_batch in proto_loader:
                proto_batch = proto_batch.to(self.device)
                labels_batch = labels_batch.to(self.device, dtype=torch.long)

                # Generate prototypes for all classes
                proto_gen = self.tgp(list(range(self.num_class)))

                # Calculate distances
                dist = torch.cdist(proto_batch, proto_gen, p=2.0)

                # Add margin for true class
                one_hot = F.one_hot(labels_batch, self.num_class).to(self.device)
                margin = min(torch.max(self.gap).item(), self.args.margin_threshold)
                dist = dist + one_hot * margin

                # Loss: use negative distance as logits
                loss = ce_loss(-dist, labels_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

        # Generate global prototypes
        self.tgp.eval()
        self.global_protos = {}
        with torch.no_grad():
            for class_id in range(self.num_class):
                self.global_protos[class_id] = self.tgp(
                    torch.tensor(class_id, device=self.device)
                ).data.clone()

    def evaluate_personalized(self):
        """Evaluate personalized client models using global prototypes"""
        from .utils.evaluate import evaluate_model

        total_correct = 0
        total_samples = 0

        # Use global prototypes for evaluation if available
        if self.global_protos is not None:
            # Pre-process global prototypes into a tensor for vectorized calculation
            # Use 'inf' to handle missing classes so they are never selected
            global_protos_tensor = torch.zeros(
                self.num_class, self.args.feature_dim, device=self.device
            )
            global_protos_tensor.fill_(1e9)

            for k, v in self.global_protos.items():
                if k < self.num_class:
                    global_protos_tensor[k] = v.to(self.device)

            for i in range(self.num_clients):
                client_model = get_model(
                    self.args.model, self.args.dataset, self.args.feature_dim
                ).to(self.device)
                client_model.load_state_dict(self.clients_state[i])
                client_model.eval()

                test_loader = DataLoader(self.test_set[i], batch_size=64, shuffle=False)

                correct = 0
                total = 0

                with torch.no_grad():
                    for x, y in test_loader:
                        x, y = x.to(self.device), y.to(self.device)
                        _, features = client_model(x)

                        # Minimizing L2 distance is equivalent to minimizing MSE
                        dists = torch.cdist(features, global_protos_tensor, p=2.0)

                        # Predict class with minimum distance
                        pred = torch.argmin(dists, dim=1)
                        correct += (pred == y).sum().item()
                        total += y.shape[0]

                total_correct += correct
                total_samples += total
        else:
            # Fallback to standard evaluation
            for i in range(self.num_clients):
                client_model = get_model(
                    self.args.model, self.args.dataset, self.args.feature_dim
                ).to(self.device)
                client_model.load_state_dict(self.clients_state[i])

                acc = evaluate_model(client_model, self.test_set[i], self.device)
                test_size = len(self.test_set[i])

                total_correct += acc * test_size / 100
                total_samples += test_size

        avg_acc = (total_correct / total_samples) * 100 if total_samples > 0 else 0.0
        self.acc.append(avg_acc)
        self.model.cpu()

    def save(self):
        f = {
            "acc": self.acc,
            "loss": self.loss,
            "state_dict": {
                "global": self.model.state_dict(),
                "client": self.clients_state,
                "tgp": self.tgp.state_dict(),
                "proto": self.global_protos,
            },
        }
        super().deal_save(f)
