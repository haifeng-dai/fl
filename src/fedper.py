import copy
import argparse
import time
import torch

from .utils import (
    BaseServer,
    run_parallel_clients,
    param_aggregate,
    ce_loss,
    get_model,
    evaluate_model,
)


def client_worker(params):
    device = params[0]
    global_body_state = params[1]
    local_head_state = params[2]
    train_set = params[3]
    model_name = params[4]
    dataset_name = params[5]
    lr = params[6]
    batch_size = params[7]
    epochs = params[8]

    # Initialize model
    model = get_model(model_name, dataset_name).to(device)

    # Load parameters directly into sub-modules
    # 1. Load global body into extractor
    model.extractor.load_state_dict(global_body_state)
    # 2. Load local head into classifier
    if local_head_state is not None:
        model.classifier.load_state_dict(local_head_state)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    model.train()
    total_loss = 0.0
    num_batches = 0

    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            output, _ = model(x)
            loss = ce_loss(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches

    # Extract body and head directly from sub-modules
    # Move them to CPU
    new_body_state = {k: v.cpu() for k, v in model.extractor.state_dict().items()}
    new_head_state = {k: v.cpu() for k, v in model.classifier.state_dict().items()}

    return [avg_loss, new_body_state, new_head_state]


class Server(BaseServer):
    def __init__(self, args: argparse.Namespace):
        model = get_model(args.model, args.dataset)
        super().__init__(model, True, args)
        assert isinstance(self.model.extractor, torch.nn.Sequential) and isinstance(self.model.classifier, torch.nn.Linear)

        # Initialize local heads for each client using the initial classifier state
        initial_head = {k: v.cpu() for k, v in self.model.classifier.state_dict().items()}  #

        self.client_head_states = [
            copy.deepcopy(initial_head) for _ in range(self.num_clients)
        ]

    def fit(self):
        assert isinstance(self.model.extractor, torch.nn.Sequential) and isinstance(self.model.classifier, torch.nn.Linear)
        for r in range(self.rounds):
            t0 = time.time()
            print(f"\n--- FedPer Round {r + 1}/{self.rounds} ---")

            # Prepare global body state (extractor)
            global_body = {k: v.cpu() for k, v in self.model.extractor.state_dict().items()}

            p = [
                [
                    self.client_gpu[i],
                    global_body,
                    self.client_head_states[i],
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

            total_loss = 0.0
            new_bodies = []

            for i, res in enumerate(results):
                loss = res[0]
                new_body = res[1]
                new_head = res[2]

                total_loss += loss
                new_bodies.append(new_body)
                self.client_head_states[i] = new_head

            avg_loss = total_loss / self.num_clients
            self.loss.append(avg_loss)

            # Aggregate Body Only
            aggregated_body = param_aggregate(new_bodies, self.weights)
            # Load aggregated body directly into extractor
            self.model.extractor.load_state_dict(aggregated_body)

            # Evaluate
            self.evaluate()
            print(f"Global Accuracy (Avg Personal): {self.acc[-1]:.2f}%", f"Avg Loss: {avg_loss:.4f}")
            print(f"Round finished in {time.time() - t0:.2f} seconds")

    def evaluate(self):
        assert isinstance(self.model.extractor, torch.nn.Sequential) and isinstance(self.model.classifier, torch.nn.Linear)
        accs = []
        # Get current global body
        global_body = {k: v.cpu() for k, v in self.model.extractor.state_dict().items()}

        for i in range(self.num_clients):
            # Load global body and local head into self.model for evaluation
            self.model.extractor.load_state_dict(global_body)
            self.model.classifier.load_state_dict(self.client_head_states[i])

            acc = evaluate_model(self.model, self.test_set[i], self.device)
            accs.append(acc)

        self.acc.append(sum(accs) / len(accs))

    def save(self, test):
        file_name: str = f"{self.args.epochs}_{self.args.batch_size}_{self.args.lr}.pt"
        f = {
            "acc": self.acc,
            "loss": self.loss,
            "state_dict": self.model.state_dict()
        }
        super().deal_save(test, f, file_name)