import argparse
import time

import numpy as np
import torch

from .utils import BaseServer, ce_loss, get_model, run_parallel_clients


def client_worker(params):
    """
    Standard FedAvg local training.
    """
    (
        device,
        model_state,
        train_set,
        model_name,
        dataset_name,
        lr,
        batch_size,
        epochs,
    ) = params

    # 1. Initialize model and load global state
    model = get_model(model_name, dataset_name).to(device)
    model.load_state_dict(model_state)

    # 2. Setup optimizer and data loader
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
    )

    # 3. Local training loop
    total_loss = 0.0
    num_batches = 0
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)

            # Standard Cross Entropy Loss
            batch_loss = ce_loss(logits, y)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            total_loss += batch_loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches

    # 4. Prepare return values (move to CPU)
    # Move state_dict to CPU to avoid CUDA IPC warnings
    model_state = {k: v.cpu() for k, v in model.state_dict().items()}
    return [avg_loss, model_state]


class Server(BaseServer):
    def __init__(self, args: argparse.Namespace):
        super().__init__(False, args)

    def fit(self):
        num_join_clients = int(self.num_clients * self.args.join_ratio)
        # Ensure at least one client participates
        num_join_clients = max(1, num_join_clients)

        for r in range(self.rounds):
            t0 = time.time()
            print(f"\n--- FedAvg Round {r + 1}/{self.rounds} ---")

            selected_clients = np.random.choice(
                self.num_clients, num_join_clients, replace=False
            )
            print(f"Selected clients: {selected_clients}")

            p = [
                [
                    self.client_gpu[i],
                    self.model.state_dict(),
                    self.train_sets[i],
                    self.args.model,
                    self.args.dataset,
                    self.args.lr,
                    self.args.batch_size,
                    self.args.epochs,
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

            # Calculate average loss using incremental summation
            total_loss = 0.0
            for i in range(num_join_clients):
                total_loss += results[i][0]
            avg_loss = total_loss / num_join_clients
            self.loss.append(avg_loss)

            self.clients_state = [results[i][1] for i in range(num_join_clients)]

            # Calculate weights for selected clients
            current_weights = [self.weights[i] for i in selected_clients]
            sum_weights = sum(current_weights)
            # Normalize weights to sum to 1
            norm_weights = [w / sum_weights for w in current_weights]

            self.aggregate(self.clients_state, weights=norm_weights)
            self.evaluate()
            print(f"Global Accuracy: {self.acc[-1]:.2f}%, Avg Loss: {avg_loss:.4f}")
            print(f"Round finished in {time.time() - t0:.2f} seconds")

    def save(self):
        f = {
            "acc": self.acc,
            "loss": self.loss,
            "state_dict": self.model.state_dict(),
        }
        super().deal_save(f)
