import torch
import torch.nn as nn
import argparse
import copy
from .utils import BaseClient, BaseServer, run_parallel_clients


def add_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("FedProto Specific Arguments")
    group.add_argument(
        "--mu", type=float, default=1.0, help="Weight for prototype loss"
    )
    return parser


class Client(BaseClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mu = self.args.mu

        self.local_protos: dict[int, torch.Tensor] = {}
        self.global_protos: torch.Tensor

    def train(self):
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        loss_list = []
        train_loader = self.build_train_loader()
        for _ in range(self.epochs):
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output, features = self.model(data)
                loss_ce = self.ce(output, target)

                loss_proto = torch.tensor(0.0).to(self.device)
                if self.global_protos is not None and len(self.global_protos) > 0:
                    classes_in_batch = torch.unique(target)
                    for c in classes_in_batch:
                        if c.item() in self.global_protos:
                            c_features = features[target == c]
                            c_global_proto = self.global_protos[c.item()].to(
                                self.device
                            )
                            loss_proto += self.mse(
                                c_features,
                                c_global_proto.expand(c_features.shape[0], -1),
                            )

                loss = loss_ce + self.mu * loss_proto
                loss.backward()
                optimizer.step()
                loss_list.append(loss.item())

        # After training, calculate local prototypes
        self.model.eval()
        local_protos = {}
        counts = {}
        with torch.no_grad():
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                _, features = self.model(data)
                for i in range(len(target)):
                    label = target[i].item()
                    if label not in local_protos:
                        local_protos[label] = features[i].cpu().clone()
                        counts[label] = 1
                    else:
                        local_protos[label] += features[i].cpu()
                        counts[label] += 1

        # Average features for each class
        for label in local_protos:
            local_protos[label] /= counts[label]
        for label, proto in local_protos.items():
            self.local_protos[label] = proto.data.clone()

        return sum(loss_list) / len(loss_list), local_protos

    def set_client(self, global_protos):
        """Receive global prototypes from server."""
        for label, proto in global_protos.items():
            self.local_protos[label] = proto.data.clone()


class Server(BaseServer):
    def __init__(self, model: torch.nn.Module, args: argparse.Namespace):
        super().__init__(model, True, args)
        for i in range(args.num_clients):
            self.clients[i] = Client(
                client_id=i, model=model, train_set=self.train_sets[i], args=args
            )
        self.global_protos: dict[int, torch.Tensor] = {}
        self.acc_p: list[float] = []
        self.loss_p: list[float] = []

    def fit(self):
        for r in range(self.rounds):
            print(f"\n--- FedProto Round {r + 1}/{self.rounds} ---")

            for i in range(self.num_clients):
                self.clients[i].set_client(self.global_protos)

            parameters_per_client = [self.global_protos] * self.num_clients

            results = run_parallel_clients(
                clients=self.clients,
                parameters=parameters_per_client,
                gpu_pools=self.gpu_pools,
                no_mp=self.no_mp,
            )

            # results is a list of (avg_loss, local_protos)
            losses = [res[0] for res in results]
            all_local_protos = [res[1] for res in results]

            avg_loss = sum(losses) / self.num_clients
            self.loss.append(avg_loss)

            # Aggregate prototypes from all clients
            self.global_protos = self.aggregate_protos(all_local_protos)

            self.evaluate()
            print(f"Global Accuracy: {self.acc[-1]:.2f}%, Avg Loss: {avg_loss:.4f}")

    def aggregate_protos(self, all_local_protos):
        global_protos = {}
        counts = {}
        for local_protos in all_local_protos:
            for label, proto in local_protos.items():
                if label not in global_protos:
                    global_protos[label] = proto.clone()
                    counts[label] = 1
                else:
                    global_protos[label] += proto
                    counts[label] += 1

        for label in global_protos:
            global_protos[label] /= counts[label]

        return global_protos

    def evaluate(self):
        # Override evaluate to use local models on local test sets
        accs = []
        for client_id in self.clients:
            acc_ = self.clients[client_id].evaluate(self.test_set[client_id])
            accs.append(acc_)
        acc = sum(accs) / len(accs)
        self.acc.append(acc)

    def save(self, test):
        f = {
            "acc": self.acc,
            "loss": self.loss,
            "global_protos": self.global_protos,
        }
        super().deal_save(test, f)
