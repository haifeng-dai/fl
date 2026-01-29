import argparse
import random
import time
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .utils import BaseServer, ce_loss, get_model, run_parallel_clients, evaluate_model


def add_args(parser: argparse.ArgumentParser):
    """Add FedALA specific arguments"""
    group = parser.add_argument_group("FedALA Specific Arguments")
    group.add_argument(
        "--eta",
        type=float,
        default=1.0,
        help="ALA weight learning rate (default: 1.0)",
    )
    group.add_argument(
        "--rand_percent",
        type=int,
        default=80,
        help="Percentage of local data to sample for ALA weight learning (default: 80)",
    )
    group.add_argument(
        "--layer_idx",
        type=int,
        default=2,
        help="Number of higher layers to apply ALA. 0 means all layers (default: 2)",
    )
    group.add_argument(
        "--ala_threshold",
        type=float,
        default=0.1,
        help="Convergence threshold for ALA weight learning (default: 0.1)",
    )
    group.add_argument(
        "--num_pre_loss",
        type=int,
        default=10,
        help="Number of recent losses to calculate std for ALA convergence (default: 10)",
    )
    return parser


class ALA:
    """Adaptive Local Aggregation module for FedALA"""

    def __init__(
        self,
        client_id: int,
        train_data,
        batch_size: int,
        model_name: str,
        dataset_name: str,
        rand_percent: int,
        layer_idx: int = 0,
        eta: float = 1.0,
        device: str = "cpu",
        threshold: float = 0.1,
        num_pre_loss: int = 10,
    ):
        self.client_id = client_id
        self.train_data = train_data
        self.batch_size = batch_size
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.rand_percent = rand_percent
        self.layer_idx = layer_idx
        self.eta = eta
        self.threshold = threshold
        self.num_pre_loss = num_pre_loss
        self.device = device

        self.weights = None  # Learnable local aggregation weights
        self.start_phase = True

    def adaptive_local_aggregation(
        self, global_model: nn.Module, local_model: nn.Module
    ):
        """
        Apply adaptive local aggregation to initialize local model.

        Args:
            global_model: The received global model
            local_model: The current local model
        """
        # Randomly sample partial local training data
        rand_ratio = self.rand_percent / 100
        rand_num = int(rand_ratio * len(self.train_data))
        rand_idx = random.randint(0, len(self.train_data) - rand_num)

        # Use Subset to create a proper Dataset for DataLoader
        # Slicing the dataset directly might return a tuple of tensors (x, y)
        # which DataLoader interprets incorrectly as two samples
        indices = list(range(rand_idx, rand_idx + rand_num))
        subset = torch.utils.data.Subset(self.train_data, indices)

        rand_loader = DataLoader(subset, self.batch_size, drop_last=False)

        # Get parameter references
        params_g = list(global_model.parameters())
        params = list(local_model.parameters())

        # Deactivate ALA at the 1st communication iteration
        if torch.sum(params_g[0] - params[0]) == 0:
            return

        # Preserve all updates in the lower layers
        if self.layer_idx > 0:
            for param, param_g in zip(
                params[: -self.layer_idx], params_g[: -self.layer_idx]
            ):
                param.data = param_g.data.clone()

        # Temp local model only for weight learning
        model_t = get_model(self.model_name, self.dataset_name).to(self.device)
        model_t.load_state_dict(local_model.state_dict())
        params_t = list(model_t.parameters())

        # Only consider higher layers
        if self.layer_idx > 0:
            params_p = params[-self.layer_idx :]
            params_gp = params_g[-self.layer_idx :]
            params_tp = params_t[-self.layer_idx :]
        else:
            # If layer_idx == 0, apply ALA to all layers
            params_p = params
            params_gp = params_g
            params_tp = params_t

        # Freeze lower layers to reduce computational cost
        if self.layer_idx > 0:
            for param in params_t[: -self.layer_idx]:
                param.requires_grad = False

        # Optimizer for weight learning (lr=0 as we manually update)
        optimizer = torch.optim.SGD(params_tp, lr=0)

        # Initialize weights to all ones in the beginning
        if self.weights is None:
            self.weights = [
                torch.ones_like(param.data).to(self.device) for param in params_p
            ]

        # Initialize higher layers in temp local model
        for param_t, param, param_g, weight in zip(
            params_tp, params_p, params_gp, self.weights
        ):
            param_t.data = param + (param_g - param) * weight

        # Weight learning loop
        loss_t = []
        losses = []
        while True:
            for x, y in rand_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()

                # Forward pass
                output, _ = model_t(x)

                loss = ce_loss(output, y)
                loss.backward()

                # Update weights based on gradients
                for param_t, param, param_g, weight in zip(
                    params_tp, params_p, params_gp, self.weights
                ):
                    weight.data = torch.clamp(
                        weight - self.eta * (param_t.grad * (param_g - param)), 0, 1
                    )

                # Update temp local model with new weights
                for param_t, param, param_g, weight in zip(
                    params_tp, params_p, params_gp, self.weights
                ):
                    param_t.data = param + (param_g - param) * weight

                loss_t.append(loss.item())
            losses.append(np.mean(loss_t))

            # Only train one epoch in subsequent iterations
            if not self.start_phase:
                break

            # Train until convergence in the first iteration
            if (
                len(losses) > self.num_pre_loss
                and np.std(losses[-self.num_pre_loss :]) < self.threshold
            ):
                break

        self.start_phase = False

        # Apply learned aggregation to local model
        for param, param_t in zip(params_p, params_tp):
            param.data = param_t.data.clone()


def client_worker(params):
    """
    FedALA client worker with adaptive local aggregation.
    """
    (
        client_id,
        device,
        global_model_state,
        local_model_state,
        train_set,
        model_name,
        dataset_name,
        lr,
        batch_size,
        epochs,
        eta,
        rand_percent,
        layer_idx,
        ala_threshold,
        num_pre_loss,
    ) = params

    # Initialize models
    global_model = get_model(model_name, dataset_name).to(device)
    global_model.load_state_dict(global_model_state)

    local_model = get_model(model_name, dataset_name).to(device)
    local_model.load_state_dict(local_model_state)

    # Initialize ALA module
    ala = ALA(
        client_id=client_id,
        train_data=train_set,
        batch_size=batch_size,
        model_name=model_name,
        dataset_name=dataset_name,
        rand_percent=rand_percent,
        layer_idx=layer_idx,
        eta=eta,
        device=device,
        threshold=ala_threshold,
        num_pre_loss=num_pre_loss,
    )

    # Apply adaptive local aggregation
    ala.adaptive_local_aggregation(global_model, local_model)

    # Standard local training
    optimizer = torch.optim.SGD(local_model.parameters(), lr=lr)
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    total_loss = 0.0
    num_batches = 0
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            # Forward pass
            output, _ = local_model(x)

            loss = ce_loss(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

    # Return results (move to CPU)
    model_state = {k: v.cpu() for k, v in local_model.state_dict().items()}
    return [avg_loss, model_state]


class Server(BaseServer):
    def __init__(self, args: argparse.Namespace):
        # FedALA is a personalized FL method
        super().__init__(True, args)
        self.clients_state = {
            i: self.model.state_dict() for i in range(self.num_clients)
        }

    def fit(self):
        num_join_clients = int(self.num_clients * self.args.join_ratio)
        num_join_clients = max(1, num_join_clients)

        for r in range(self.rounds):
            t0 = time.time()
            print(f"\n--- FedALA Round {r + 1}/{self.rounds} ---")

            # Select clients
            selected_clients = np.random.choice(
                self.num_clients, num_join_clients, replace=False
            )
            print(f"Selected clients: {selected_clients}")

            # Prepare parameters for parallel execution
            global_model_state_cpu = {
                k: v.cpu() for k, v in self.model.state_dict().items()
            }

            p = [
                [
                    i,
                    self.client_gpu[i],
                    global_model_state_cpu,
                    self.clients_state[i],
                    self.train_sets[i],
                    self.args.model,
                    self.args.dataset,
                    self.args.lr,
                    self.args.batch_size,
                    self.args.epochs,
                    self.args.eta,
                    self.args.rand_percent,
                    self.args.layer_idx,
                    self.args.ala_threshold,
                    self.args.num_pre_loss,
                ]
                for i in selected_clients
            ]

            # Run parallel client training
            results = run_parallel_clients(
                client_worker=client_worker,
                parameters=p,
                gpu_pools=self.gpu_pools,
                mp=self.mp,
            )

            # Process results
            # Calculate total loss, get states and weights of selected clients
            total_loss = 0.0
            selected_states = []
            current_weights = []
            for i in selected_clients:
                total_loss += results[i][0]
                self.clients_state[i] = results[i][1]
                selected_states.append(results[i][1])
                current_weights.append(self.weights[i])
            self.loss.append(total_loss / num_join_clients)
            sum_weights = sum(current_weights)
            norm_weights = [w / sum_weights for w in current_weights]

            self.aggregate(selected_states, weights=norm_weights)
            self.evaluate_personalized()

            print(
                f"Personalized Accuracy: {self.acc[-1]:.2f}%, Avg Loss: {self.loss[-1]:.4f}"
            )
            print(f"Round finished in {time.time() - t0:.2f} seconds")

    def evaluate_personalized(self):
        """Evaluate personalized client models"""

        total_correct = 0
        total_samples = 0

        for i in range(self.num_clients):
            # Load client model
            client_model = get_model(self.args.model, self.args.dataset).to(self.device)
            client_model.load_state_dict(self.clients_state[i])

            # Evaluate on client's test set
            acc = evaluate_model(client_model, self.test_set[i], self.device)
            test_size = len(self.test_set[i])  # type: ignore

            total_correct += acc * test_size / 100
            total_samples += test_size

        avg_acc = (total_correct / total_samples) * 100
        self.acc.append(avg_acc)

    def save(self):
        f = {
            "acc": self.acc,
            "loss": self.loss,
            "state_dict": self.clients_state,
        }
        self.deal_save(f)

    def get_log_path(self):
        self.file_name = f"{self.save_name_pre}_{self.args.eta}_{self.args.rand_percent}_{self.args.layer_idx}_{self.args.ala_threshold}_{self.args.num_pre_loss}"
        return os.path.join(self.log_path, f"{self.file_name}.log")
