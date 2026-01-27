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


def decompose_param(param, energy_threshold):
    """
    使用基于能量阈值的 SVD 分解单个参数张量

    Args:
        param: 参数张量
        energy_threshold: 能量阈值 (0-1)

    Returns:
        compressed_param: 压缩后的参数（字典或张量）
    """
    # 保持在原设备，不强制移到 CPU
    param_shape = param.shape

    # 检查是否可以进行分解（2D 或 4D 张量）
    # 同时通常跳过 embedding 层
    if len(param_shape) not in [2, 4] or "embedding" in str(param.dtype):
        return param.detach().cpu()

    # 重塑为 2D 矩阵
    if len(param_shape) == 4:
        # 卷积层: (out, in, h, w) -> (out, in*h*w)
        mat = param.view(param_shape[0], -1)
    else:
        mat = param

    # 执行 SVD 分解
    try:
        # 优先在原设备（如 GPU）上执行
        u, s, vh = torch.linalg.svd(mat, full_matrices=False)
    except RuntimeError:
        # SVD 失败时的回退方案（如显存不足），回退到 CPU
        mat = mat.cpu()
        try:
            u, s, vh = torch.linalg.svd(mat, full_matrices=False)
        except RuntimeError:
            return param.detach().cpu()

    # 基于能量阈值确定秩
    total_energy = torch.sum(s**2)
    if total_energy == 0:
        return param.detach().cpu()

    cumulative_energy = torch.cumsum(s**2, dim=0)
    # 找到累积能量超过 threshold * total 的第一个索引
    mask = cumulative_energy > (energy_threshold * total_energy)
    if not mask.any():
        rank = len(s)
    else:
        rank = torch.searchsorted(mask.int(), 1).item() + 1

    # 压缩并立即移至 CPU 以节省显存和兼容通信
    u_k = u[:, :rank].detach().cpu()
    s_k = s[:rank].detach().cpu()
    vh_k = vh[:rank, :].detach().cpu()

    return {
        "u": u_k,
        "s": s_k,
        "vh": vh_k,
        "original_shape": param_shape,
        "is_compressed": True,
    }


def reconstruct_param(compressed_param, device):
    """
    从压缩表示重构参数

    Args:
        compressed_param: 压缩后的参数（来自 decompose_param）
        device: 目标设备

    Returns:
        重构后的参数张量
    """
    if isinstance(compressed_param, dict) and compressed_param.get("is_compressed"):
        u = compressed_param["u"].to(device)
        s = compressed_param["s"].to(device)
        vh = compressed_param["vh"].to(device)

        # 重构: U * diag(S) * Vh
        mat = u @ (torch.diag(s) @ vh)

        # 重塑回原始形状
        return mat.view(compressed_param["original_shape"])
    elif isinstance(compressed_param, torch.Tensor):
        return compressed_param.to(device)
    else:
        # 数据干净时不应该发生
        raise ValueError(f"未知参数类型: {type(compressed_param)}")


def client_worker(params):
    """
    FedKD local training with SVD-based communication compression and mutual knowledge distillation.
    """
    # Safe unpacking
    (
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
    model = get_model(model_name, dataset_name).to(device)
    # Global proxy model (constructed from compressed SVD params)
    model_g = get_model(model_name, dataset_name).to(device)

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

        # 初始分解
        self.compressed_params = {}
        for name, param in self.model.state_dict().items():
            self.compressed_params[name] = decompose_param(param, args.energy)

        # 存储每个客户端的私有本地模型状态（用于模拟本地持久化）
        self.client_wh_states = [None for _ in range(self.num_clients)]

        # 预先计算特征维度，避免客户端重复计算
        self.feature_dim = self._get_feature_dim(args.dataset)

    def _get_feature_dim(self, dataset_name):
        """Helper to get feature dimension of the model"""
        if dataset_name == "mnist":
            dummy_input = torch.randn(1, 1, 28, 28)
        else:
            dummy_input = torch.randn(1, 3, 32, 32)

        with torch.no_grad():
            _, feat = self.model(dummy_input)
        return feat.shape[1]

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

            # 向客户端发送压缩后的全局参数
            p = [
                [
                    self.client_gpu[i],
                    self.args.model,
                    self.args.dataset,
                    self.feature_dim,
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
                num_clients=num_join_clients,
                parameters=p,
                gpu_pools=self.gpu_pools,
                mp=self.mp,
            )

            total_loss = sum(res[0] for res in results)
            avg_loss = total_loss / num_join_clients
            self.loss.append(avg_loss)

            # 更新服务器端存储的客户端本地状态
            for i, res in enumerate(results):
                client_idx = selected_clients[i]
                self.clients_state[client_idx] = res[2]
                self.client_wh_states[client_idx] = res[3]

            # 聚合 SVD 参数 (仅聚合 model_g)
            client_compressed_params_list = [res[1] for res in results]

            # Calculate weights for selected clients
            current_weights = [self.weights[i] for i in selected_clients]
            sum_weights = sum(current_weights)
            norm_weights = [w / sum_weights for w in current_weights]

            self.aggregate_svd(client_compressed_params_list, weights=norm_weights)

            self.evaluate()
            print(f"Global Accuracy: {self.acc[-1]:.2f}%, Avg Loss: {avg_loss:.4f}")
            print(f"Round finished in {time.time() - t0:.2f} seconds")

    def evaluate(self):
        acc = 0.0
        for i in range(self.num_clients):
            model = get_model(self.args.model, self.args.dataset).to(self.device)
            with torch.no_grad():
                model.load_state_dict(self.clients_state[i])
            acc_i = evaluate_model(model, self.test_set[i], self.device)
            acc += acc_i
        self.acc.append(acc / self.num_clients)

    def aggregate_svd(self, client_params_list, weights):
        """聚合 SVD 压缩的参数"""
        # 1. 将所有参数重构到 CPU
        aggregated_state_dict = {}
        ref_params = client_params_list[0]

        # 用第一个客户端初始化
        for name in ref_params.keys():
            param_0 = reconstruct_param(ref_params[name], torch.device("cpu"))
            aggregated_state_dict[name] = param_0 * weights[0]

        # 累积其余客户端
        for i in range(1, len(client_params_list)):
            client_params = client_params_list[i]
            for name in client_params.keys():
                param = reconstruct_param(client_params[name], torch.device("cpu"))
                aggregated_state_dict[name] += param * weights[i]

        # 2. 更新服务器模型
        self.model.load_state_dict(aggregated_state_dict)

        # 3. 为下一轮分发重新压缩
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

    def get_log_path(self):
        self.file_name = f"{self.save_name_pre}_{self.args.lr_g}_{self.args.energy}"
        return os.path.join(self.log_path, f"{self.file_name}.log")
