import argparse
import copy
import time

import numpy as np
import torch

from src.utils.evaluate import evaluate_prototype

from .utils import (
    BaseServer,
    ce_loss,
    evaluate_model,
    get_model,
    run_parallel_clients,
)


def add_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("FedProto Specific Arguments")
    group.add_argument(
        "--mu", type=float, default=1.0, help="Weight for prototype loss"
    )
    return parser


def client_worker(params):
    (
        device,
        model_state,
        global_protos,
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

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    mse_loss = torch.nn.MSELoss()
    loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    total_loss = 0.0
    num_batches = 0
    for _ in range(epochs):
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output, features = model(data)
            loss_ce = ce_loss(output, target)

            loss_proto = torch.tensor(0.0).to(device)
            if global_protos and len(global_protos) > 0:
                classes_in_batch = torch.unique(target)
                for c in classes_in_batch:
                    if c.item() in global_protos:
                        c_features = features[target == c]
                        c_global_proto = global_protos[c.item()].to(device)
                        loss_proto += mse_loss(
                            c_features,
                            c_global_proto.expand(c_features.shape[0], -1),
                        )

            loss = loss_ce + mu * loss_proto
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

    # After training, calculate local prototypes
    model.eval()
    local_protos = {}
    counts = {}
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            _, features = model(data)
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

    # Return client_id, avg_loss, new_model_state, local_protos
    avg_loss = total_loss / num_batches
    model_state = {k: v.cpu() for k, v in model.state_dict().items()}
    return [avg_loss, model_state, local_protos]


class Server(BaseServer):
    def __init__(self, args: argparse.Namespace):
        super().__init__(True, args)
        # Initialize personalized models for each client
        self.client_states = [
            copy.deepcopy(self.model.state_dict()) for _ in range(self.num_clients)
        ]
        self.global_protos: dict[int, torch.Tensor] = {}
        self.acc_p: list[float] = []

    def fit(self):
        num_join_clients = int(self.num_clients * self.args.join_ratio)
        num_join_clients = max(1, num_join_clients)

        for r in range(self.rounds):
            t0 = time.time()
            print(f"\n--- FedProto Round {r + 1}/{self.rounds} ---")

            selected_clients = np.random.choice(
                self.num_clients, num_join_clients, replace=False
            )
            print(f"Selected clients: {selected_clients}")

            p = [
                [
                    self.client_gpu[i],
                    self.client_states[i],
                    self.global_protos,
                    self.train_sets[i],
                    self.args.model,
                    self.args.dataset,
                    self.args.lr,
                    self.args.batch_size,
                    self.args.epochs,
                    self.args.mu,
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

            # results: [avg_loss, model_state, local_protos]
            new_model_states = [res[1] for res in results]
            all_local_protos = [res[2] for res in results]

            # Calculate average loss using incremental summation
            total_loss = 0.0
            for res in results:
                total_loss += res[0]
            avg_loss = total_loss / num_join_clients
            self.loss.append(avg_loss)

            # Update client models
            for i, state in enumerate(new_model_states):
                client_idx = selected_clients[i]
                self.client_states[client_idx] = {k: v.cpu() for k, v in state.items()}

            self.global_protos = self.aggregate_protos(all_local_protos)
            self.evaluate()
            print(
                f"Global Accuracy: {self.acc[-1]:.2f}%, "
                f"Proto Accuracy: {self.acc_p[-1]:.2f}%, "
                f"Avg Loss: {avg_loss:.4f}"
            )
            print(f"Round finished in {time.time() - t0:.2f} seconds")

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
        # Evaluate each client's personalized model on its local test set
        accs = []
        acc_ps = []

        # Prepare prototype tensor from dict for evaluation
        proto_tensor = None
        if self.global_protos:
            max_label = max(self.global_protos.keys())
            dim = next(iter(self.global_protos.values())).shape[0]
            proto_tensor = torch.zeros(max_label + 1, dim).to(self.device)
            for label, proto in self.global_protos.items():
                proto_tensor[label] = proto.to(self.device)

        for i in range(self.num_clients):
            # Load client i's model state into self.model for evaluation
            self.model.load_state_dict(self.client_states[i])

            acc = evaluate_model(self.model, self.test_set[i], self.device)
            accs.append(acc)

            acc_p = evaluate_prototype(
                self.model, proto_tensor, self.test_set[i], self.device
            )
            acc_ps.append(acc_p)

        self.acc.append(sum(accs) / (len(accs) if accs else 1))
        self.acc_p.append(sum(acc_ps) / (len(acc_ps) if acc_ps else 1))

    def save(self):
        file_name = f"{self.args.mu}"
        f = {
            "acc": {"model": self.acc, "proto": self.acc_p},
            "loss": self.loss,
            "state_dict": {
                "model": self.client_states,
                "proto": self.global_protos,
            },
        }
        super().deal_save(f, file_name)
