import argparse
import copy
import time
import os
import torch
import numpy as np
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from .utils import BaseServer, ce_loss, get_model, run_parallel_clients


def add_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("FedDyn Specific Arguments")
    group.add_argument(
        "--alpha_coef", type=float, default=0.01, help="Regularization coefficient (alpha)"
    )
    return parser


def get_path(args):
    args.file_name = f"{args.name_pre}_alpha{args.alpha_coef}"
    return os.path.join(args.log_path, f"{args.file_name}_{args.times}.log")


def client_worker(params):
    (
        client_id,
        device,
        model_state,
        grad_prev,  # Local gradient history (nabla L_k(w^{t-1}))
        global_model_vector,  # Flattened global model
        train_set,
        model_name,
        dataset_name,
        lr,
        batch_size,
        epochs,
        alpha_coef,
        feature_dim,
    ) = params

    # 1. Initialize model
    model = get_model(model_name, dataset_name, feature_dim).to(device)
    model.load_state_dict(model_state)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    # Move vectors to device
    if grad_prev is not None:
        grad_prev = grad_prev.to(device)
    if global_model_vector is not None:
        global_model_vector = global_model_vector.to(device)

    total_loss = 0.0
    num_batches = 0

    model.train()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)

            # Task Loss
            task_loss = ce_loss(logits, y)

            # FedDyn Regularization
            # L = L_task - <grad_prev, w> + (alpha/2) * ||w - w_global||^2
            curr_params = parameters_to_vector(model.parameters())

            # Linear penalty: - <grad_prev, w>
            lin_penalty = 0.0
            if grad_prev is not None:
                lin_penalty = -torch.dot(grad_prev, curr_params)

            # Quadratic penalty: (alpha/2) * ||w - w_global||^2
            quad_penalty = 0.0
            if global_model_vector is not None:
                diff = curr_params - global_model_vector
                quad_penalty = (alpha_coef / 2.0) * torch.sum(diff ** 2)

            loss = task_loss + lin_penalty + quad_penalty

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += task_loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

    # Return updated model state (cpu)
    return_state = {k: v.cpu() for k, v in model.state_dict().items()}
    return [avg_loss, return_state]


class Server(BaseServer):
    def __init__(self, args: argparse.Namespace):
        super().__init__(False, args)

        # FedDyn State
        # h: global gradient history (vector)
        # Use parameters_to_vector to get the shape and initial zero vector
        self.h = parameters_to_vector(self.model.parameters()).detach().clone().zero_()

        # Local gradients history (nabla L_k)
        # Stored on CPU to save GPU memory
        self.local_grads = {
            i: torch.zeros_like(self.h) for i in range(self.num_clients)
        }

    def fit(self):
        num_join_clients = int(self.num_clients * self.args.join_ratio)
        num_join_clients = max(1, num_join_clients)

        # Global model vector for next round
        global_model_vector = parameters_to_vector(self.model.parameters()).detach().clone()

        for r in range(self.rounds):
            t0 = time.time()
            print(f"\n--- FedDyn Round {r + 1}/{self.rounds} ---")

            selected_clients = np.random.choice(
                self.num_clients, num_join_clients, replace=False
            )
            print(f"Selected clients: {selected_clients}")

            p = [
                [
                    i,
                    self.client_gpu[i],
                    self.model.state_dict(),
                    self.local_grads[i],
                    global_model_vector,
                    self.train_sets[i],
                    self.args.model,
                    self.args.dataset,
                    self.args.lr,
                    self.args.batch_size,
                    self.args.epochs,
                    self.args.alpha_coef,
                    self.args.feature_dim,
                ]
                for i in selected_clients
            ]

            results = run_parallel_clients(
                client_worker=client_worker,
                parameters=p,
                gpu_pools=self.gpu_pools,
                mp=self.mp,
            )

            total_loss = 0.0
            sum_model_params = torch.zeros_like(global_model_vector)

            # Process results
            for idx, i in enumerate(selected_clients):
                loss, client_state_dict = results[idx]
                total_loss += loss

                # Convert client state to vector
                # Faster: manually flatten the dict values in order
                # Safe way: load into self.model (cpu) then flatten.
                self.model.load_state_dict(client_state_dict)
                client_flat = parameters_to_vector(self.model.parameters()).detach()

                sum_model_params += client_flat

                # Update local grad history:
                # nabla L_k(w^{t+1}) approx nabla L_k(w^t) - alpha * (w^{t+1} - w^t)
                model_diff = client_flat - global_model_vector
                self.local_grads[i] -= self.args.alpha_coef * model_diff

            self.loss.append(total_loss / num_join_clients)

            # 1. Average Client Models
            avg_model_params = sum_model_params / num_join_clients

            # 2. Update Global History h
            # h_{t+1} = h_t - alpha * (w_{avg} - w_t)
            self.h -= self.args.alpha_coef * (avg_model_params - global_model_vector)

            # 3. Update Global Model
            # w_{t+1} = w_{avg} - (1/alpha) * h_{t+1}
            new_global_vector = avg_model_params - (1.0 / self.args.alpha_coef) * self.h

            # Load back to model
            vector_to_parameters(new_global_vector, self.model.parameters())
            global_model_vector = new_global_vector

            self.evaluate()
            print(
                f"Global Accuracy: {self.acc[-1]:.2f}%, Avg Loss: {self.loss[-1]:.4f}"
            )
            print(f"Round finished in {time.time() - t0:.2f} seconds")

    def save(self):
        f = {
            "acc": self.acc,
            "loss": self.loss,
            "state_dict": self.model.state_dict(),
        }
        self.deal_save(f)
