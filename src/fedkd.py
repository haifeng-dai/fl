import argparse
import time
import os

import numpy as np
import torch

from .utils import (
    BaseServer,
    ce_loss,
    evaluate_model,
    get_model,
    kl_loss,
    mse_loss,
    run_parallel_clients,
)


def add_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("FedKD Specific Arguments")
    group.add_argument(
        "--lr_g",
        type=float,
        default=0.005,
        help="Learning rate for global model (student)",
    )
    group.add_argument(
        "--energy", type=float, default=0.95, help="SVD energy threshold (0-1)"
    )
    return parser


def get_path(args):
    args.file_name = f"{args.name_pre}_{args.lr_g}_{args.energy}"
    return os.path.join(args.log_path, f"{args.file_name}_{args.times}.log")


def decompose_param(param, energy_threshold):
    """
    Decompose a single parameter tensor using SVD based on energy threshold.

    Args:
        param: Parameter tensor
        energy_threshold: Energy threshold (0-1)

    Returns:
        compressed_param: Compressed parameter (dict or tensor)
    """
    # Keep on original device, do not force move to CPU
    param_shape = param.shape

    # Check if decomposition is possible (2D or 4D tensor)
    # Also usually skip embedding layers
    if len(param_shape) not in [2, 4] or "embedding" in str(param.dtype):
        return param.detach().cpu()

    # Reshape to 2D matrix
    if len(param_shape) == 4:
        # Conv layer: (out, in, h, w) -> (out, in*h*w)
        mat = param.view(param_shape[0], -1)
    else:
        mat = param

    # Perform SVD decomposition
    try:
        # Prefer execution on original device (e.g., GPU)
        u, s, vh = torch.linalg.svd(mat, full_matrices=False)
    except RuntimeError:
        # Fallback for SVD failure (e.g., OOM), fallback to CPU
        mat = mat.cpu()
        try:
            u, s, vh = torch.linalg.svd(mat, full_matrices=False)
        except RuntimeError:
            return param.detach().cpu()

    # Determine rank based on energy threshold
    total_energy = torch.sum(s**2)
    if total_energy == 0:
        return param.detach().cpu()

    cumulative_energy = torch.cumsum(s**2, dim=0)
    # Find the first index where cumulative energy exceeds threshold * total
    mask = cumulative_energy > (energy_threshold * total_energy)
    if not mask.any():
        rank = len(s)
    else:
        rank = torch.searchsorted(mask.int(), 1).item() + 1

    return {
        "u": u[:, :rank].detach().cpu(),
        "s": s[:rank].detach().cpu(),
        "vh": vh[:rank, :].detach().cpu(),
        "original_shape": param_shape,
        "is_compressed": True,
    }


def reconstruct_param(compressed_param, device):
    """
    Reconstruct parameter from compressed representation.

    Args:
        compressed_param: Compressed parameter (from decompose_param)
        device: Target device

    Returns:
        Reconstructed parameter tensor
    """
    if isinstance(compressed_param, dict) and compressed_param.get("is_compressed"):
        u = compressed_param["u"].to(device)
        s = compressed_param["s"].to(device)
        vh = compressed_param["vh"].to(device)

        # Reconstruct: U * diag(S) * Vh
        mat = u @ (torch.diag(s) @ vh)

        # Reshape back to original shape
        return mat.view(compressed_param["original_shape"])
    elif isinstance(compressed_param, torch.Tensor):
        return compressed_param.to(device)
    else:
        # Should not happen if data is clean
        raise ValueError(f"Unknown parameter type: {type(compressed_param)}")


def client_worker(params):
    """
    FedKD local training with SVD-based communication compression and mutual knowledge distillation.
    """
    # Safe unpacking
    (
        _,
        device,
        model_name,
        dataset_name,
        feature_dim,
        train_set,
        compressed_params_g,
        prev_local_state,
        wh_state,
        lr,
        lr_g,
        batch_size,
        epochs,
        energy_threshold,
    ) = params

    # 1. Initialize Models
    # Local personalized model
    model = get_model(model_name, dataset_name, feature_dim).to(device)
    # Global proxy model (constructed from compressed SVD params)
    model_g = get_model(model_name, dataset_name, feature_dim).to(device)

    with torch.no_grad():
        # A. Reconstruct and load global proxy parameters from SVD components
        global_state_dict = {}
        for name, param_data in compressed_params_g.items():
            global_state_dict[name] = reconstruct_param(param_data, device)
        model_g.load_state_dict(global_state_dict)

        # B. Load local model parameters
        if prev_local_state is not None:
            model.load_state_dict(prev_local_state)
        else:
            # First round: start from global state
            model.load_state_dict(global_state_dict)

    # 2. Initialize Feature Alignment Layer (W_h)
    W_h = torch.nn.Linear(feature_dim, feature_dim, bias=False, device=device)
    if wh_state is not None:
        W_h.load_state_dict(wh_state)

    # 3. Optimizers
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    optimizer_g = torch.optim.SGD(model_g.parameters(), lr=lr_g)
    optimizer_W = torch.optim.SGD(W_h.parameters(), lr=lr)

    # 4. Training Loop (Mutual Knowledge Distillation)
    loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    model.train()
    model_g.train()
    W_h.train()

    total_loss = 0.0
    num_batches = 0

    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            # Forward pass
            output, rep = model(x)
            output_g, rep_g = model_g(x)

            # Task Loss (Cross Entropy)
            loss_ce = ce_loss(output, y)
            loss_ce_g = ce_loss(output_g, y)

            # Mutual Knowledge Distillation (KL Divergence)
            loss_kd = kl_loss(output, output_g.detach())
            loss_kd_g = kl_loss(output_g, output.detach())

            # Feature Alignment Loss
            loss_h = mse_loss(rep, W_h(rep_g.detach()))
            loss_h_g = mse_loss(rep.detach(), W_h(rep_g))

            # Normalization factor
            scale = loss_ce.item() + loss_ce_g.item() + 1e-8

            # Total Losses
            loss = loss_ce + loss_kd / scale + loss_h / scale
            loss_g = loss_ce_g + loss_kd_g / scale + loss_h_g / scale

            # Optimization Steps
            optimizer.zero_grad()
            optimizer_g.zero_grad()
            optimizer_W.zero_grad()

            loss.backward(retain_graph=True)
            loss_g.backward()

            optimizer.step()
            optimizer_g.step()
            optimizer_W.step()

            total_loss += loss.item()
            num_batches += 1

    # 5. Compress updated global model using SVD for uplink transmission
    avg_loss = total_loss / num_batches
    compressed_params_g_new = {}
    for name, param in model_g.state_dict().items():
        compressed_params_g_new[name] = decompose_param(param, energy_threshold)

    # Prepare return states
    local_state = {k: v.cpu() for k, v in model.state_dict().items()}
    wh_state = {k: v.cpu() for k, v in W_h.state_dict().items()}

    return [avg_loss, compressed_params_g_new, local_state, wh_state]


class Server(BaseServer):
    def __init__(self, args: argparse.Namespace):
        super().__init__(True, args)

        # Initial decomposition
        self.compressed_params = {}
        for name, param in self.model.state_dict().items():
            self.compressed_params[name] = decompose_param(param, args.energy)

        self.client_wh_states = [None for _ in range(self.num_clients)]
        self.clients_state = [self.model.state_dict() for _ in range(self.num_clients)]

    def fit(self):
        num_join_clients = int(self.num_clients * self.args.join_ratio)
        num_join_clients = max(1, num_join_clients)

        for r in range(self.rounds):
            t0 = time.time()
            print(f"\n--- FedKD Round {r + 1}/{self.rounds} ---")

            selected_clients = np.random.choice(
                self.num_clients, num_join_clients, replace=False
            )
            print(f"Selected clients: {selected_clients}")

            # Send compressed global params to clients
            p = [
                [
                    i,
                    self.client_gpu[i],
                    self.args.model,
                    self.args.dataset,
                    self.args.feature_dim,
                    self.train_sets[i],
                    self.compressed_params,
                    self.clients_state[i],
                    self.client_wh_states[i],
                    self.args.lr,
                    self.args.lr_g,
                    self.args.batch_size,
                    self.args.epochs,
                    self.args.energy,
                ]
                for i in selected_clients
            ]

            results = run_parallel_clients(
                client_worker=client_worker,
                parameters=p,
                gpu_pools=self.gpu_pools,
                mp=self.mp,
            )

            # Update server-side stored client local states
            total_loss = 0.0
            client_compressed_params_list = []
            current_weights = []
            for i in selected_clients:
                client_loss, client_compressed, client_body, client_head = results[i]
                total_loss += client_loss
                client_compressed_params_list.append(client_compressed)
                self.clients_state[i] = client_body
                self.client_wh_states[i] = client_head
                current_weights.append(self.weights[i])
            self.loss.append(total_loss / num_join_clients)
            sum_weights = sum(current_weights)
            norm_weights = [w / sum_weights for w in current_weights]

            self.aggregate_svd(client_compressed_params_list, weights=norm_weights)

            self.evaluate()
            print(
                f"Global Accuracy: {self.acc[-1]:.2f}%, Avg Loss: {self.loss[-1]:.4f}"
            )
            print(f"Round finished in {time.time() - t0:.2f} seconds")

    def evaluate(self):
        acc = 0.0
        for i in range(self.num_clients):
            model = get_model(
                self.args.model, self.args.dataset, self.args.feature_dim
            ).to(self.device)
            with torch.no_grad():
                model.load_state_dict(self.clients_state[i])
            acc_i = evaluate_model(model, self.test_set[i], self.device)
            acc += acc_i
        self.acc.append(acc / self.num_clients)
        self.model.cpu()

    def aggregate_svd(self, client_params_list, weights):
        """Aggregate SVD compressed parameters"""
        # 1. Reconstruct all params to CPU
        aggregated_state_dict = {}
        ref_params = client_params_list[0]

        # Initialize with first client
        for name in ref_params.keys():
            param_0 = reconstruct_param(ref_params[name], torch.device("cpu"))
            aggregated_state_dict[name] = param_0 * weights[0]

        # Accumulate remaining clients
        for i in range(1, len(client_params_list)):
            client_params = client_params_list[i]
            for name in client_params.keys():
                param = reconstruct_param(client_params[name], torch.device("cpu"))
                aggregated_state_dict[name] += param * weights[i]

        # 2. Update server model
        self.model.load_state_dict(aggregated_state_dict)

        # 3. Re-compress for next round distribution
        self.compressed_params = {}
        for name, param in self.model.state_dict().items():
            self.compressed_params[name] = decompose_param(param, self.args.energy)

    def save(self):
        f = {
            "acc": self.acc,
            "loss": self.loss,
            "state_dict": {
                "global": self.model.state_dict(),
                "clients": self.clients_state,
                "wh": self.client_wh_states,
            },
        }
        self.deal_save(f)
