import argparse
import time

import torch

from .utils import BaseServer, ce_loss, get_model, run_parallel_clients


def client_worker(params):
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

    model = get_model(model_name, dataset_name).to(device)
    model.load_state_dict(model_state)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
    )
    total_loss = 0.0
    num_batches = 0
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            batch_loss = ce_loss(logits, y)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            total_loss += batch_loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    return [avg_loss, model.state_dict()]


class Server(BaseServer):
    def __init__(self, args: argparse.Namespace):
        super().__init__(False, args)

    def fit(self):
        for r in range(self.rounds):
            t0 = time.time()
            print(f"\n--- FedAvg Round {r + 1}/{self.rounds} ---")

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
                for i in range(self.num_clients)
            ]

            results = run_parallel_clients(
                client_worker=client_worker,
                num_clients=self.num_clients,
                parameters=p,
                gpu_pools=self.gpu_pools,
                mp=self.mp,
            )

            # Calculate average loss using incremental summation
            total_loss = 0.0
            for i in range(self.num_clients):
                total_loss += results[i][0]
            avg_loss = total_loss / self.num_clients
            self.loss.append(avg_loss)

            clients_params = [results[i][1] for i in range(self.num_clients)]
            self.aggregate(clients_params)
            self.evaluate()
            print(f"Global Accuracy: {self.acc[-1]:.2f}%, Avg Loss: {avg_loss:.4f}")
            print(f"Round finished in {time.time() - t0:.2f} seconds")

    def save(self, test):
        f = {
            "acc": self.acc,
            "loss": self.loss,
            "state_dict": self.model.state_dict(),
        }
        super().deal_save(test, f)
