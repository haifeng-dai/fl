import argparse

import torch

from .utils import BaseServer, ce_loss, get_model, run_parallel_clients


def client_worker(params):
    device = params[0]
    model_state = params[1]
    train_set = params[2]
    model_name = params[3]
    dataset_name = params[4]
    lr = params[5]
    batch_size = params[6]
    epochs = params[7]

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
        model = get_model(args.model, args.dataset)
        super().__init__(model, False, args)

    def fit(self):
        for r in range(self.rounds):
            print(f"\n--- FedAvg Round {r + 1}/{self.rounds} ---")
            global_params = {k: v.cpu() for k, v in self.model.state_dict().items()}

            p = [
                [
                    v,
                    global_params,
                    self.train_sets[k],
                    self.args.model,
                    self.args.dataset,
                    self.args.lr,
                    self.args.batch_size,
                    self.args.epochs,
                ]
                for k, v in self.client_gpu.items()
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

    def save(self, test):
        f = {
            "acc": self.acc,
            "loss": self.loss,
            "state_dict": self.model.state_dict(),
        }
        super().deal_save(test, f)
