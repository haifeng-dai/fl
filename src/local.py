import argparse
import time
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from .utils import BaseServer, ce_loss, evaluate_model, get_model, run_parallel_clients


def get_path(args):
    args.file_name = f"{args.name_pre}_local"
    return os.path.join(args.log_path, f"{args.file_name}_{args.times}.log")


def client_worker(params):
    """
    Pure local training - independent training without communication.
    Each client trains from scratch on its own data.
    """
    (
        _,
        device,
        train_set,
        model_name,
        dataset_name,
        lr,
        batch_size,
        epochs,
        feature_dim,
    ) = params

    model = get_model(model_name, dataset_name, feature_dim).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    total_loss = 0.0
    num_batches = 0
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)

            loss = ce_loss(logits, y)

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
        super().__init__(True, args)

        self.clients_state = [
            get_model(args.model, args.dataset, args.feature_dim).state_dict()
            for _ in range(self.num_clients)
        ]

        self.mp = False

    def fit(self):
        num_join_clients = int(self.num_clients * self.args.join_ratio)
        num_join_clients = max(1, num_join_clients)

        for r in range(self.rounds):
            t0 = time.time()
            print(f"\n--- Local Training Round {r + 1}/{self.rounds} ---")

            selected_clients = np.random.choice(
                self.num_clients, num_join_clients, replace=False
            )
            print(f"Selected clients: {selected_clients}")

            p = [
                [
                    i,
                    self.client_gpu[i],
                    self.train_sets[i],
                    self.args.model,
                    self.args.dataset,
                    self.args.lr,
                    self.args.batch_size,
                    self.args.epochs,
                    self.args.feature_dim,
                ]
                for i in selected_clients
            ]

            results = run_parallel_clients(
                client_worker=client_worker,
                parameters=p,
                gpu_pools={},
                mp=False,
            )

            total_loss = 0.0
            for i in selected_clients:
                client_loss, client_state = results[i]
                total_loss += client_loss
                self.clients_state[i] = client_state
            self.loss.append(total_loss / num_join_clients)

            self.evaluate_personalized()
            print(
                f"Personalized Accuracy: {self.acc[-1]:.2f}%, Avg Loss: {self.loss[-1]:.4f}"
            )
            print(f"Round finished in {time.time() - t0:.2f} seconds")

    def evaluate_personalized(self):
        total_correct = 0
        total_samples = 0

        for i in range(self.num_clients):
            client_model = get_model(
                self.args.model, self.args.dataset, self.args.feature_dim
            ).to(self.device)
            client_model.load_state_dict(self.clients_state[i])

            acc = evaluate_model(client_model, self.test_set[i], self.device)
            test_size = len(self.test_set[i])

            total_correct += acc * test_size / 100
            total_samples += test_size

        avg_acc = (total_correct / total_samples) * 100
        self.acc.append(avg_acc)
        self.model.cpu()

    def save(self):
        f = {
            "acc": self.acc,
            "loss": self.loss,
            "state_dict": self.clients_state,
        }
        self.deal_save(f)
