import argparse
import copy
import time
import os
import torch
import torch.optim as optim
import numpy as np

from .utils import BaseServer, ce_loss, get_model, run_parallel_clients, param_aggregate


def add_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("SCAFFOLD Specific Arguments")
    group.add_argument(
        "--global_lr", type=float, default=1.0, help="Global learning rate"
    )
    return parser


def get_path(args):
    args.file_name = f"{args.name_pre}_glr{args.global_lr}"
    return os.path.join(args.log_path, f"{args.file_name}_{args.times}.log")


class SCAFFOLDOptimizer(optim.Optimizer):
    def __init__(self, params, lr, weight_decay):
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super(SCAFFOLDOptimizer, self).__init__(params, defaults)

    def step(self, c_global, c_local):
        for group in self.param_groups:
            for p, c_g, c_l in zip(group["params"], c_global, c_local):
                if p.grad is None:
                    continue
                d_p = p.grad.data
                p.data.add_(d_p + c_g.data - c_l.data, alpha=-group["lr"])


def client_worker(params):
    (
        client_id,
        device,
        model_state,
        c_global_state,
        c_local_state,
        train_set,
        model_name,
        dataset_name,
        lr,
        batch_size,
        epochs,
        feature_dim,
    ) = params

    # 1. Initialize model
    model = get_model(model_name, dataset_name, feature_dim).to(device)
    model.load_state_dict(model_state)

    # 2. Prepare control variates
    trainable_names = [n for n, p in model.named_parameters()]

    if c_global_state is None:
        c_global_dict = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
        c_local_dict = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    else:
        c_global_dict = {k: v.to(device) for k, v in c_global_state.items()}
        c_local_dict = {k: v.to(device) for k, v in c_local_state.items()}

    # Flatten for optimizer
    c_global_list = [c_global_dict[n] for n in trainable_names]
    c_local_list = [c_local_dict[n] for n in trainable_names]

    optimizer = SCAFFOLDOptimizer(model.parameters(), lr=lr, weight_decay=0.0)
    loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    # 3. Training
    model.train()
    steps = 0
    total_loss = 0.0

    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            loss = ce_loss(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step(c_global_list, c_local_list)

            total_loss += loss.item()
            steps += 1

    c_delta_dict = {}
    c_local_new_dict = {}

    global_state_device = {k: v.to(device) for k, v in model_state.items()}
    current_state = model.state_dict()

    scaling = 1.0 / (steps * lr)
    for name, param in model.named_parameters():
        c_l = c_local_dict[name]
        c_g = c_global_dict[name]
        w_g = global_state_device[name]
        w_l = param.data

        c_new = c_l - c_g + (w_g - w_l) * scaling
        c_local_new_dict[name] = c_new.cpu()
        c_delta_dict[name] = (c_new - c_l).cpu()

    avg_loss = total_loss / steps if steps > 0 else 0

    # Return: loss, model_state, c_delta, c_local_new
    return [
        avg_loss,
        {k: v.cpu() for k, v in current_state.items()},
        c_delta_dict,
        c_local_new_dict,
    ]


class Server(BaseServer):
    def __init__(self, args: argparse.Namespace):
        super().__init__(False, args)

        # Get parameter names
        self.param_names = [n for n, p in self.model.named_parameters()]

        self.c_global = {
            n: torch.zeros_like(p) for n, p in self.model.named_parameters()
        }
        self.c_local = [
            {n: torch.zeros_like(p) for n, p in self.model.named_parameters()}
            for _ in range(self.num_clients)
        ]

    def fit(self):
        num_join_clients = int(self.num_clients * self.args.join_ratio)
        num_join_clients = max(1, num_join_clients)

        for r in range(self.rounds):
            t0 = time.time()
            print(f"\n--- SCAFFOLD Round {r + 1}/{self.rounds} ---")

            selected_clients = np.random.choice(
                self.num_clients, num_join_clients, replace=False
            )
            print(f"Selected clients: {selected_clients}")

            p = [
                [
                    i,
                    self.client_gpu[i],
                    self.model.state_dict(),
                    self.c_global,
                    self.c_local[i],
                    self.train_sets[i],
                    self.args.model,
                    self.args.dataset,
                    self.args.lr,
                    self.args.batch_size,
                    self.args.epochs,
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

            # Process results
            total_loss = 0.0
            total_delta_c = {
                n: torch.zeros_like(self.c_global[n]) for n in self.param_names
            }

            selected_states = []
            for i in selected_clients:
                client_loss, client_state, client_delta_c, client_c_local = results[i]
                total_loss += client_loss
                selected_states.append(client_state)
                # Accumulate delta_c for global update
                for n in self.param_names:
                    total_delta_c[n] += client_delta_c[n]
                # Update local control variate stored on server
                self.c_local[i] = client_c_local
            self.loss.append(total_loss / num_join_clients)

            # Aggregate model
            # Use uniform weights for SCAFFOLD
            weights = [1.0 / len(selected_states)] * len(selected_states)
            avg_state = param_aggregate(selected_states, weights)

            if self.args.global_lr == 1.0:
                self.model.load_state_dict(avg_state)
            else:
                # Custom aggregation with global learning rate
                current_state = self.model.state_dict()
                for k, v in current_state.items():
                    if k in avg_state:
                        v.mul_(1 - self.args.global_lr).add_(
                            avg_state[k], alpha=self.args.global_lr
                        )

            factor = 1.0 / self.num_clients
            for n in self.param_names:
                self.c_global[n] += total_delta_c[n] * factor

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
