import argparse
import copy
import time
import os

import numpy as np
import torch

from .utils import (
    BaseServer,
    ce_loss,
    evaluate_model,
    get_model,
    param_aggregate,
    run_parallel_clients,
)


def client_worker(params):
    """
    FedPer local training.
    """
    (
        device,
        global_body_state,
        local_head_state,
        train_set,
        model_name,
        dataset_name,
        lr,
        batch_size,
        epochs,
    ) = params

    # 1. Initialize model
    model = get_model(model_name, dataset_name).to(device)

    # 2. Load parameters directly into sub-modules
    # Load global body (extractor) - shared
    model.extractor.load_state_dict(global_body_state)
    # Load local head (classifier) - personalized
    if local_head_state is not None:
        model.classifier.load_state_dict(local_head_state)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    # 3. Training Loop
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

    # 4. Extract body and head directly from sub-modules
    # Move them to CPU
    # Body is sent to server for aggregation
    new_body_state = {k: v.cpu() for k, v in model.extractor.state_dict().items()}
    # Head is kept local
    new_head_state = {k: v.cpu() for k, v in model.classifier.state_dict().items()}

    return [avg_loss, new_body_state, new_head_state]


class Server(BaseServer):
    def __init__(self, args: argparse.Namespace):
        super().__init__(True, args)
        initial_head = {
            k: v.cpu() for k, v in self.model.classifier.state_dict().items()
        }

        self.client_head_states = [
            copy.deepcopy(initial_head) for _ in range(self.num_clients)
        ]

    def fit(self):
        num_join_clients = int(self.num_clients * self.args.join_ratio)
        num_join_clients = max(1, num_join_clients)

        for r in range(self.rounds):
            t0 = time.time()
            print(f"\n--- FedPer Round {r + 1}/{self.rounds} ---")

            selected_clients = np.random.choice(
                self.num_clients, num_join_clients, replace=False
            )
            print(f"Selected clients: {selected_clients}")

            p = [
                [
                    self.client_gpu[i],
                    self.model.extractor.state_dict(),
                    self.client_head_states[i],
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

            total_loss = 0.0
            new_bodies = []

            for i, res in enumerate(results):
                loss = res[0]
                new_body = res[1]
                new_head = res[2]

                total_loss += loss
                new_bodies.append(new_body)

                # Update local head state for the selected client
                client_idx = selected_clients[i]
                self.client_head_states[client_idx] = new_head

            avg_loss = total_loss / num_join_clients
            self.loss.append(avg_loss)

            # Calculate weights for selected clients
            current_weights = [self.weights[i] for i in selected_clients]
            sum_weights = sum(current_weights)
            norm_weights = [w / sum_weights for w in current_weights]

            # Aggregate Body Only
            aggregated_body = param_aggregate(new_bodies, norm_weights)
            # Load aggregated body directly into extractor
            self.model.extractor.load_state_dict(aggregated_body)

            # Evaluate
            self.evaluate()
            print(
                f"Global Accuracy (Avg Personal): {self.acc[-1]:.2f}%",
                f"Avg Loss: {avg_loss:.4f}",
            )
            print(f"Round finished in {time.time() - t0:.2f} seconds")

    def evaluate(self):
        assert isinstance(self.model.extractor, torch.nn.Sequential) and isinstance(
            self.model.classifier, torch.nn.Linear
        )
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

    def save(self):
        client_states = []
        global_body = self.model.extractor.state_dict()
        for i in range(self.num_clients):
            full_state = copy.deepcopy(global_body)
            full_state.update(self.client_head_states[i])
            client_states.append(full_state)

        f = {"acc": self.acc, "loss": self.loss, "client_states": client_states}
        self.deal_save(f)

    def get_log_path(self):
        self.file_name = f"{self.save_name_pre}"
        return os.path.join(self.log_path, f"{self.file_name}.log")
