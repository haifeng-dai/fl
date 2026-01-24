import argparse
import time
import torch
from .utils import BaseServer, ce_loss, get_model, run_parallel_clients


def add_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--mu", type=float, default=0.01, help="Proximal term coefficient for FedProx"
    )


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
        mu,
    ) = params

    model = get_model(model_name, dataset_name).to(device)
    model.load_state_dict(model_state)

    # Keep a copy of global model weights for the proximal term
    # Since model_state is passed in, we can convert it to device tensors once
    global_model_params = {k: v.to(device) for k, v in model_state.items()}

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

            # Calculate Cross Entropy Loss
            loss = ce_loss(logits, y)

            # Calculate Proximal Term
            prox_term = sum(
                ((param - global_model_params[name]) ** 2).sum()
                for name, param in model.named_parameters()
                if name in global_model_params
            )

            # Total Loss
            loss += (mu / 2) * prox_term

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    return [avg_loss, model.state_dict()]


class Server(BaseServer):
    def __init__(self, args: argparse.Namespace):
        super().__init__(False, args)

    def fit(self):
        print(f"FedProx with mu={self.args.mu}")
        for r in range(self.rounds):
            t0 = time.time()
            print(f"\n--- FedProx Round {r + 1}/{self.rounds} ---")
            global_params = {k: v.cpu() for k, v in self.model.state_dict().items()}

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
                    self.args.mu,
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

            # Process results
            avg_loss = sum(res[0] for res in results) / self.num_clients
            self.loss.append(avg_loss)

            clients_params = [res[1] for res in results]
            self.aggregate(clients_params)
            self.evaluate()
            print(f"Global Accuracy: {self.acc[-1]:.2f}%, Avg Loss: {avg_loss:.4f}")
            print(f"Round finished in {time.time() - t0:.2f} seconds")

    def save(self, test):
        f = {
            "acc": self.acc,
            "loss": self.loss,
            "state_dict": self.model.state_dict(),
            "mu": self.args.mu
        }
        super().deal_save(test, f)
