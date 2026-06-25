import math
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .utils import (
    BaseServer,
    _fmt_num,
    ce_loss,
    compute_mh_weights,
    generate_adjacency_matrix,
    get_model,
)


def get_path(args):
    adj_suffix = f"{args.adj_type}"
    if args.adj_type == "random":
        adj_suffix += f"_{_fmt_num(args.edge_p)}"
    elif args.adj_type == "small_world":
        adj_suffix += f"_{_fmt_num(args.k_small_world)}_{_fmt_num(args.edge_p)}"
    elif args.adj_type == "scale_free":
        adj_suffix += f"_{_fmt_num(args.m_scale_free)}"

    args.file_name = (
        f"{args.common_name}_{adj_suffix}"
        f"_r{_fmt_num(args.event_r)}_bw{_fmt_num(args.bandwidth_mean)}"
    )
    return os.path.join(args.log_path, f"{args.file_name}_{args.cur_time}.log")


def train(params):
    (
        _,
        device,
        model_state,
        train_set,
        model_name,
        dataset_name,
        lr,
        batch_size,
        epochs,
        feature_dim,
        hat_state,
        r,
        rho_i,
        gamma_k,
    ) = params

    model = get_model(model_name, dataset_name, feature_dim).to(device)
    model.load_state_dict(model_state)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    total_loss = 0.0
    num_batches = 0
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = ce_loss(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches

    new_state = model.state_dict()
    n_params = 0
    diff_sq = 0.0
    for k in new_state:
        diff = new_state[k].cpu() - hat_state[k].cpu()
        diff_sq += (diff.norm() ** 2).item()
        n_params += diff.numel()
    change = math.sqrt(diff_sq / n_params)

    new_state = {k: v.cpu().detach().clone() for k, v in new_state.items()}
    return {"loss": avg_loss, "state": new_state, "change": change}


class Server(BaseServer):
    def __init__(self, args):
        super().__init__(pfl=True, args=args)

        self.adj_matrix = generate_adjacency_matrix(args)
        self.mh_weights = compute_mh_weights(self.adj_matrix, device=self.device)

        b_M = getattr(args, "bandwidth_mean", 5000)
        sigma_N = getattr(args, "bandwidth_std", 0.9)
        low = (1.0 - sigma_N) * b_M
        high = (1.0 + sigma_N) * b_M
        rng = np.random.default_rng(42)
        self.bandwidths = rng.uniform(low, high, size=self.num_clients)
        self.rho = [1.0 / b for b in self.bandwidths]
        self.r = getattr(args, "event_r", 250)

        init_sd = self.model.state_dict()
        self.clients_state = [
            {k: v.cpu().clone() for k, v in init_sd.items()}
            for _ in range(self.num_clients)
        ]
        self.hat_states = [
            {k: v.cpu().clone() for k, v in init_sd.items()}
            for _ in range(self.num_clients)
        ]

        self.client_changes = {}
        self.triggered_log = []
        self.triggered_ids_log = []
        self.changes_log = []

        self._build_param_info()

    def _build_param_info(self):
        sd = self.model.state_dict()
        self.param_keys = list(sd.keys())
        self.param_shapes = [sd[k].shape for k in self.param_keys]
        self.param_sizes = [sd[k].numel() for k in self.param_keys]
        self.param_offsets = []
        offset = 0
        for sz in self.param_sizes:
            self.param_offsets.append(offset)
            offset += sz
        self.total_params = offset

    def _flatten_states(self, states_list):
        N = len(states_list)
        flat = torch.zeros(N, self.total_params, dtype=torch.float32)
        for i in range(N):
            for j, k in enumerate(self.param_keys):
                start = self.param_offsets[j]
                end = start + self.param_sizes[j]
                flat[i, start:end] = states_list[i][k].cpu().view(-1)
        return flat

    def _unflatten_to_states(self, flat):
        states = []
        for i in range(flat.shape[0]):
            sd = {}
            for j, k in enumerate(self.param_keys):
                start = self.param_offsets[j]
                end = start + self.param_sizes[j]
                sd[k] = flat[i, start:end].view(self.param_shapes[j]).clone()
            states.append(sd)
        return states

    def lr_schedule(self, k):
        return 0.1 / math.sqrt(1.0 + k)

    def gamma(self, k):
        return self.lr_schedule(k)

    def event_trigger_and_aggregate(self, k):
        gamma_k = self.gamma(k)
        triggered_mask = [
            self.client_changes.get(i, 0.0) >= self.r * self.rho[i] * gamma_k
            for i in range(self.num_clients)
        ]

        triggered_ids = [i for i in range(self.num_clients) if triggered_mask[i]]

        if triggered_ids:
            P = torch.eye(self.num_clients, dtype=torch.float32)
            for i in triggered_ids:
                P[i] = self.mh_weights[i].to(self.device)

            W_flat = self._flatten_states(
                [self.clients_state[i] for i in range(self.num_clients)]
            )
            W_new_flat = P @ W_flat
            new_states = self._unflatten_to_states(W_new_flat)

            for i in triggered_ids:
                self.clients_state[i] = new_states[i]
                self.hat_states[i] = {
                    k: v.cpu().detach().clone() for k, v in new_states[i].items()
                }

        n_triggered = len(triggered_ids)
        self.triggered_log.append(n_triggered)
        self.triggered_ids_log.append(triggered_ids)
        return triggered_ids

    def fit(self):
        num_join_clients = int(self.num_clients * self.args.join_ratio)
        num_join_clients = max(1, num_join_clients)

        for r in range(self.rounds):
            t0 = time.time()
            lr = self.lr_schedule(r)
            print(f"\n--- EF-HC Round {r + 1}/{self.rounds} ---")

            selected_clients = np.random.choice(
                self.num_clients, num_join_clients, replace=False
            )

            p = [
                [
                    i,
                    self.client_gpu[i],
                    self.clients_state[i],
                    self.train_sets[i],
                    self.args.model,
                    self.args.dataset,
                    lr,
                    self.args.batch_size,
                    self.args.epochs,
                    self.args.feature_dim,
                    self.hat_states[i],
                    self.r,
                    self.rho[i],
                    self.gamma(r),
                ]
                for i in selected_clients
            ]
            results = self.run_clients(train, p)

            total_loss = 0.0
            for cid, res in results.items():
                total_loss += res["loss"]
                self.clients_state[cid] = res["state"]
                self.client_changes[cid] = res["change"]
            self.loss.append(total_loss / len(results))

            changes_this_round = {
                i: self.client_changes.get(i, 0.0) for i in range(self.num_clients)
            }
            self.changes_log.append(changes_this_round)

            for i in range(self.num_clients):
                thresh = self.r * self.rho[i] * self.gamma(r)
                flag = " *" if self.client_changes.get(i, 0.0) >= thresh else ""
                print(
                    f"  Client {i:2d}: change={changes_this_round[i]:.6f}  "
                    f"threshold={thresh:.6f}  bw={self.bandwidths[i]:.1f}{flag}"
                )

            triggered_ids = self.event_trigger_and_aggregate(r)
            n_triggered = len(triggered_ids)

            print(
                f"Event Triggered: {n_triggered}/{len(selected_clients)} "
                f"clients (IDs: {triggered_ids})"
            )

            self.evaluate()
            print(f"Avg Loss: {self.loss[-1]:.4f}, Acc: {self.acc[-1]:.2f}%")
            print(f"Round finished in {time.time() - t0:.2f}s")

    def save(self):
        f = {
            "acc": self.acc,
            "loss": self.loss,
            "state_dict": self.clients_state,
            "bandwidths": self.bandwidths.tolist(),
            "num_triggered": self.triggered_log,
            "triggered_ids": self.triggered_ids_log,
            "changes_log": self.changes_log,
        }
        self.deal_save(f)
