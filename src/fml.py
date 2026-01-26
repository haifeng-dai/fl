import argparse
import copy
import time

import numpy as np
import torch

from .utils import (
    BaseServer,
    ce_loss,
    evaluate_model,
    get_model,
    kl_loss,
    param_aggregate,
    run_parallel_clients,
)


def add_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("FML Specific Arguments")
    group.add_argument(
        "--alpha_fml",
        type=float,
        default=1.0,
        help="Weight for KL Divergence Loss (Global to Local)",
    )
    group.add_argument(
        "--beta_fml",
        type=float,
        default=1.0,
        help="Weight for KL Divergence Loss (Local to Global)",
    )
    return parser


def client_worker(params):
    (
        device,
        global_state,
        local_state,
        train_set,
        model_name,
        dataset_name,
        lr,
        batch_size,
        epochs,
        alpha,
        beta,
    ) = params

    # 1. Initialize Global Model (MEME)
    global_model = get_model(model_name, dataset_name).to(device)
    global_model.load_state_dict(global_state)

    # 2. Initialize Local Model (Personalized)
    local_model = get_model(model_name, dataset_name).to(device)
    local_model.load_state_dict(local_state)

    # Optimizers
    opt_g = torch.optim.SGD(global_model.parameters(), lr=lr)
    opt_l = torch.optim.SGD(local_model.parameters(), lr=lr)

    loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    global_model.train()
    local_model.train()

    total_loss_g = 0.0
    total_loss_l = 0.0
    num_batches = 0

    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            # Forward pass
            out_g, _ = global_model(x)
            out_l, _ = local_model(x)

            # Cross Entropy Loss
            ce_g = ce_loss(out_g, y)
            ce_l = ce_loss(out_l, y)

            # Mutual Learning (KL Divergence)
            # KL(P || Q) -> P is target (detach), Q is input (log_softmax)

            # Loss for Global: CE + beta * KL(Local || Global)
            # We want Global to resemble Local
            loss_kl_g = kl_loss(out_g, out_l.detach())

            # Loss for Local: CE + alpha * KL(Global || Local)
            # We want Local to resemble Global
            loss_kl_l = kl_loss(out_l, out_g.detach())

            loss_g = ce_g + beta * loss_kl_g
            loss_l = ce_l + alpha * loss_kl_l

            # Update Global
            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()

            # Update Local
            opt_l.zero_grad()
            loss_l.backward()
            opt_l.step()

            total_loss_g += loss_g.item()
            total_loss_l += loss_l.item()
            num_batches += 1

    avg_loss_g = total_loss_g / num_batches
    avg_loss_l = total_loss_l / num_batches

    # Return: client_id, [avg_loss, new_global_state, new_local_state]
    global_state = {k: v.cpu() for k, v in global_model.state_dict().items()}
    local_state = {k: v.cpu() for k, v in local_model.state_dict().items()}
    return [avg_loss_l, avg_loss_g, global_state, local_state]


class Server(BaseServer):
    def __init__(self, args: argparse.Namespace):
        super().__init__(True, args)

        # Initialize local models for each client
        self.client_states = [
            copy.deepcopy(self.model.state_dict()) for _ in range(self.num_clients)
        ]
        self.loss_g = []

    def fit(self):
        num_join_clients = int(self.num_clients * self.args.join_ratio)
        num_join_clients = max(1, num_join_clients)

        for r in range(self.rounds):
            t0 = time.time()
            print(f"\n--- FML Round {r + 1}/{self.rounds} ---")

            selected_clients = np.random.choice(
                self.num_clients, num_join_clients, replace=False
            )
            print(f"Selected clients: {selected_clients}")

            p = [
                [
                    self.client_gpu[i],
                    self.model.state_dict(),
                    self.client_states[i],
                    self.train_sets[i],
                    self.args.model,
                    self.args.dataset,
                    self.args.lr,
                    self.args.batch_size,
                    self.args.epochs,
                    self.args.alpha_fml,
                    self.args.beta_fml,
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

            # results: [avg_loss_l, avg_loss_g, global_state, local_state]

            total_loss = 0.0
            total_loss_g = 0.0
            global_states_to_agg = []

            for i, res in enumerate(results):
                loss_l = res[0]
                loss_g = res[1]
                new_g_state = res[2]
                new_l_state = res[3]

                total_loss += loss_l
                total_loss_g += loss_g
                global_states_to_agg.append(new_g_state)

                # Update stored local state (move to CPU)
                client_idx = selected_clients[i]
                self.client_states[client_idx] = {
                    k: v.cpu() for k, v in new_l_state.items()
                }

            self.loss.append(total_loss / num_join_clients)
            self.loss_g.append(total_loss_g / num_join_clients)

            # Calculate weights for selected clients
            current_weights = [self.weights[i] for i in selected_clients]
            sum_weights = sum(current_weights)
            norm_weights = [w / sum_weights for w in current_weights]

            # Aggregate Global Models
            self.model.load_state_dict(
                param_aggregate(global_states_to_agg, norm_weights)
            )

            self.evaluate()
            print(
                f"Global Accuracy: {self.acc[-1]:.2f}%, Avg Local Loss: {self.loss[-1]:.4f}"
            )
            print(f"Round finished in {time.time() - t0:.2f} seconds")

    def evaluate(self):
        accs = []
        for i in range(self.num_clients):
            self.model.load_state_dict(self.client_states[i])
            acc = evaluate_model(self.model, self.test_set[i], self.device)
            accs.append(acc)

        avg_acc = sum(accs) / len(accs)
        self.acc.append(avg_acc)

    def save(self):
        file_name: str = f"{self.args.alpha_fml}_{self.args.beta_fml}"
        f = {
            "acc": self.acc,
            "loss": self.loss,
            "state_dict": {
                "global": self.model.state_dict(),
                "local": self.client_states,
            },
        }
        super().deal_save(f, file_name)
