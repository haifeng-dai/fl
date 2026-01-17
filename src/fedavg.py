import argparse
import os

import torch

from .utils import BaseClient, BaseServer, run_parallel_clients


class Client(BaseClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(self):
        self.model.train()
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.lr,
        )
        loss_ = []
        train_loader = self.build_train_loader()
        for _ in range(self.epochs):
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output, _ = self.model(data)
                loss = self.ce(output, target)
                loss.backward()
                optimizer.step()
                loss_.append(loss.item())
        model_state = {
            k: v.detach().clone().cpu() for k, v in self.model.state_dict().items()
        }
        return sum(loss_) / len(loss_), model_state

    def set_client(self, parameters):
        self.model.load_state_dict(parameters)


class Server(BaseServer):
    def __init__(self, model: torch.nn.Module, args: argparse.Namespace):
        super().__init__(model, False, args)
        for i in range(args.num_clients):
            self.clients[i] = Client(
                client_id=i, model=model, train_set=self.train_sets[i], args=args
            )

    def fit(self):
        for r in range(self.rounds):
            print(f"\n--- FedAvg Round {r + 1}/{self.rounds} ---")
            global_params = {k: v.cpu() for k, v in self.model.state_dict().items()}

            parameters_per_client = [global_params] * self.num_clients

            results = run_parallel_clients(
                clients=self.clients,
                parameters=parameters_per_client,
                gpu_pools=self.gpu_pools,
                no_mp=self.no_mp,
            )
            loss_epoch = [res[0] for res in results]
            client_dicts = [res[1] for res in results]

            avg_loss = sum(loss_epoch) / len(loss_epoch)
            self.loss.append(avg_loss)
            self.aggregate(client_dicts, weights=self.weights)
            acc = self.evaluate()
            self.acc.append(acc)
            print(f"Global Accuracy: {acc:.2f}%, Avg Loss: {avg_loss:.4f}")

    def save(self, test):
        f = {
            "acc": self.acc,
            "loss": self.loss,
            "state_dict": self.model.state_dict(),
        }
        super().deal_save(test, f)
