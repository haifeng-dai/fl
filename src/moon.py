import copy
import time

import torch
import argparse

from .utils import (
    BaseServer,
    run_parallel_clients,
    ce_loss,
    get_model,
)


def add_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("MOON Specific Arguments")
    group.add_argument(
        "--mu", type=float, default=1.0, help="Weight for contrastive loss"
    )
    group.add_argument(
        "--tau",
        type=float,
        default=0.5,
        help="Temperature parameter for contrastive loss",
    )
    return parser


def client_worker(params):
    device = params[0]
    global_state = params[1]
    prev_state = params[2]
    train_set = params[3]
    model_name = params[4]
    dataset_name = params[5]
    lr = params[6]
    batch_size = params[7]
    epochs = params[8]
    mu = params[9]
    tau = params[10]

    model = get_model(model_name, dataset_name).to(device)
    model.load_state_dict(global_state)

    global_model = copy.deepcopy(model).to(device)
    global_model.eval()

    prev_model = get_model(model_name, dataset_name).to(device)
    prev_model.load_state_dict(prev_state)
    prev_model.eval()

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    ce_moon = torch.nn.CosineSimilarity(dim=-1)

    total_loss = 0.0
    num_batches = 0
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            output, z = model(x)
            with torch.no_grad():
                _, z_glob = global_model(x)
                _, z_prev = prev_model(x)

            loss_ce = ce_loss(output, y)

            pos_sim = ce_moon(z, z_glob)
            neg_sim = ce_moon(z, z_prev)
            logits = torch.cat([pos_sim.reshape(-1, 1), neg_sim.reshape(-1, 1)], dim=1)
            logits /= tau
            labels = torch.zeros(z.size(0)).to(device).long()
            loss_con = ce_loss(logits, labels)

            loss = loss_ce + mu * loss_con
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    return [avg_loss, model.state_dict()]


class Server(BaseServer):
    def __init__(self, args: argparse.Namespace):
        model = get_model(args.model, args.dataset)
        super().__init__(model, False, args)
        # Initialize previous model states for all clients with the initial global model
        self.client_prev_states = [
            copy.deepcopy(self.model.state_dict()) for _ in range(self.num_clients)
        ]

    def fit(self):
        for r in range(self.rounds):
            t0 = time.time()
            print(f"\n--- MOON Round {r + 1}/{self.rounds} ---")
            global_params = {k: v.cpu() for k, v in self.model.state_dict().items()}

            p = [
                [
                    self.client_gpu[i],
                    global_params,
                    self.client_prev_states[i],
                    self.train_sets[i],
                    self.args.model,
                    self.args.dataset,
                    self.args.lr,
                    self.args.batch_size,
                    self.args.epochs,
                    self.args.mu,
                    self.args.tau,
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
            for res in results:
                total_loss += res[0]
            avg_loss = total_loss / self.num_clients
            self.loss.append(avg_loss)

            clients_params = [res[1] for res in results]

            # Update previous states with the newly trained models
            for i, state in enumerate(clients_params):
                self.client_prev_states[i] = {k: v.cpu() for k, v in state.items()}

            self.aggregate(clients_params)
            self.evaluate()
            print(f"Global Accuracy: {self.acc[-1]:.2f}%, Avg Loss: {avg_loss:.4f}")
            print(f"Round finished in {time.time() - t0:.2f} seconds")

    def save(self, test):
        file_name: str = f"{self.args.mu}_{self.args.tau}"
        f = {
            "acc": self.acc,
            "loss": self.loss,
            "state_dict": self.model.state_dict()
        }
        super().deal_save(test, f, file_name)
