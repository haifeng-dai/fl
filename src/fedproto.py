import torch
import argparse
import copy
from .utils import (
    BaseServer,
    run_parallel_clients,
    ce_loss,
    evaluate_model,
    get_model,
)


def add_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("FedProto Specific Arguments")
    group.add_argument(
        "--mu", type=float, default=1.0, help="Weight for prototype loss"
    )
    return parser


def client_worker(client_id, params):
    device = params[0]
    model_state = params[1]
    global_protos = params[2]
    train_set = params[3]
    model_name = params[4]
    dataset_name = params[5]
    lr = params[6]
    batch_size = params[7]
    epochs = params[8]
    mu = params[9]

    model = get_model(model_name, dataset_name).to(device)
    model.load_state_dict(model_state)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    mse_loss = torch.nn.MSELoss()
    loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True
    )

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
    return client_id, [avg_loss, model.state_dict(), local_protos]


class Server(BaseServer):
    def __init__(self, args: argparse.Namespace):
        model = get_model(args.model, args.dataset)
        super().__init__(model, True, args)
        # Initialize personalized models for each client
        self.client_model_states = [
            copy.deepcopy(self.model.state_dict()) for _ in range(self.num_clients)
        ]
        self.global_protos: dict[int, torch.Tensor] = {}

    def fit(self):
        for r in range(self.rounds):
            print(f"\n--- FedProto Round {r + 1}/{self.rounds} ---")

            parameters_per_client = []
            for i in range(self.num_clients):
                p = [
                    self.client_gpu[i],
                    self.client_model_states[i],
                    self.global_protos,
                    self.train_sets[i],
                    self.args.model,
                    self.args.dataset,
                    self.args.lr,
                    self.args.batch_size,
                    self.args.epochs,
                    self.args.mu,
                ]
                parameters_per_client.append(p)

            results = run_parallel_clients(
                client_worker=client_worker,
                num_clients=self.num_clients,
                parameters=parameters_per_client,
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
            avg_loss = total_loss / self.num_clients
            self.loss.append(avg_loss)

            # Update client models
            for i, state in enumerate(new_model_states):
                self.client_model_states[i] = {k: v.cpu() for k, v in state.items()}

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
        # Evaluate each client's personalized model on its local test set
        accs = []
        for i in range(self.num_clients):
            # Load client i's model state into self.model for evaluation
            self.model.load_state_dict(self.client_model_states[i])
            # evaluate_model handles device movement, we pass self.device (Server's device)
            acc = evaluate_model(self.model, self.test_set[i], self.device)
            accs.append(acc)

        acc = sum(accs) / len(accs)
        self.acc.append(acc)

    def save(self, test):
        f = {
            "acc": self.acc,
            "loss": self.loss,
            "global_protos": self.global_protos,
        }
        super().deal_save(test, f)
