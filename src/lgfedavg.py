import argparse
import copy
import time, os

import numpy as np
import torch

from .utils import (
    BaseServer,
    ce_loss,
    get_model,
    param_aggregate,
    run_parallel_clients,
)


def get_path(args):
    args.file_name = f"{args.name_pre}"
    return os.path.join(args.log_path, f"{args.file_name}_{args.times}.log")


def client_worker(params):
    """
    LG-FedAvg Local Worker:
    - Takes local extractor state and global classifier state.
    - Trains full model.
    - Returns updated extractor (for local storage) and classifier (for global aggregation).
    """
    (
        _,
        device,
        local_body_state,
        global_head_state,
        train_set,
        model_name,
        dataset_name,
        lr,
        batch_size,
        epochs,
        feature_dim,
    ) = params

    model = get_model(model_name, dataset_name, feature_dim).to(device)

    # Load sub-modules (Key values in these dicts should be prefix-free)
    model.extractor.load_state_dict(local_body_state)
    model.classifier.load_state_dict(global_head_state)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    model.train()
    total_loss = 0.0
    num_batches = 0
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            output, _ = model(x)
            loss = ce_loss(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    # Return split states (prefix-free)
    new_body = {k: v.cpu() for k, v in model.extractor.state_dict().items()}
    new_head = {k: v.cpu() for k, v in model.classifier.state_dict().items()}
    return [avg_loss, new_body, new_head]


class Server(BaseServer):
    def __init__(self, args: argparse.Namespace):
        # pfl=True uses local test sets
        super().__init__(True, args)
        # Override BaseServer initialization: LG-FedAvg only stores Extractor states
        self.clients_state = [
            self.model.extractor.state_dict() for _ in range(self.num_clients)
        ]

    def fit(self):
        num_join_clients = max(1, int(self.num_clients * self.args.join_ratio))

        for r in range(self.rounds):
            t0 = time.time()
            print(f"\n--- LG-FedAvg Round {r + 1}/{self.rounds} ---")
            selected_clients = np.random.choice(self.num_clients, num_join_clients, replace=False)

            # Global shared head
            global_head_state = self.model.classifier.state_dict()

            p = [[i, self.client_gpu[i], self.clients_state[i], global_head_state,
                  self.train_sets[i], self.args.model, self.args.dataset, self.args.lr,
                  self.args.batch_size, self.args.epochs, self.args.feature_dim]
                 for i in selected_clients]

            results = run_parallel_clients(client_worker, p, self.gpu_pools, self.mp)

            total_loss = 0.0
            new_heads = []
            current_weights = []
            for i in selected_clients:
                client_loss, client_body, client_head = results[i]
                total_loss += client_loss
                # Store local body back
                self.clients_state[i] = client_body
                new_heads.append(client_head)
                current_weights.append(self.weights[i])

            self.loss.append(total_loss / num_join_clients)
            norm_weights = [w / sum(current_weights) for w in current_weights]

            # Aggregate Head Only
            self.model.classifier.load_state_dict(param_aggregate(new_heads, norm_weights))

            self.evaluate()
            print(f"Accuracy: {self.acc[-1]:.2f}%, Loss: {self.loss[-1]:.4f}")
            print(f"Round finished in {time.time() - t0:.2f} seconds")

    def evaluate(self):
        """
        Build full state dicts with proper prefixes for BaseServer.evaluate().
        This ensures correct load_state_dict behavior during personalized testing.
        """
        full_states = []
        # Get latest global classifier and add prefix
        global_head_kv = {f"classifier.{k}": v for k, v in self.model.classifier.state_dict().items()}

        for i in range(self.num_clients):
            # Get this client's local extractor and add prefix
            local_body_kv = {f"extractor.{k}": v for k, v in self.clients_state[i].items()}
            # Merge
            full_state = local_body_kv
            full_state.update(global_head_kv)
            full_states.append(full_state)

        # Call smart evaluation from base class
        super().evaluate(model_states=full_states)

    def save(self):
        # Prepare full model state_dicts for saving/analysis
        client_states_full = []
        global_head_kv = {f"classifier.{k}": v for k, v in self.model.classifier.state_dict().items()}
        for i in range(self.num_clients):
            full_state = {f"extractor.{k}": v for k, v in self.clients_state[i].items()}
            full_state.update(global_head_kv)
            client_states_full.append(full_state)

        f = {
            "acc": self.acc,
            "loss": self.loss,
            "state_dict": {
                "global": self.model.classifier.state_dict(), # Global part
                "client": client_states_full,                 # Full personalized models
            },
        }
        self.deal_save(f)
