import torch
import argparse
import time

from .utils import (
    BaseServer,
    run_parallel_clients,
    ce_loss,
    mse_loss,
    get_model,
    kl_loss,
    evaluate_model,
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
    # 分离并移到 CPU 进行 SVD
    param_cpu = param.detach().cpu()
    param_shape = param_cpu.shape

    # 检查是否可以进行分解（2D 或 4D 张量）
    # 同时通常跳过 embedding 层
    if len(param_shape) not in [2, 4] or "embedding" in str(param.dtype):
        return param_cpu

    # 重塑为 2D 矩阵
    if len(param_shape) == 4:
        # 卷积层: (out, in, h, w) -> (out, in*h*w)
        mat = param_cpu.view(param_shape[0], -1)
    else:
        mat = param_cpu

    # 执行 SVD 分解
    try:
        u, s, vh = torch.linalg.svd(mat, full_matrices=False)
    except RuntimeError:
        # SVD 失败时的回退方案
        return param_cpu

    # 基于能量阈值确定秩
    total_energy = torch.sum(s**2)
    if total_energy == 0:
        return param_cpu

    cumulative_energy = torch.cumsum(s**2, dim=0)
    # 找到累积能量超过 threshold * total 的第一个索引
    mask = cumulative_energy > (energy_threshold * total_energy)
    if not mask.any():
        rank = len(s)
    else:
        rank = torch.searchsorted(mask.int(), 1).item() + 1

    # 压缩
    u_k = u[:, :rank]
    s_k = s[:rank]
    vh_k = vh[:rank, :]

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
    # 安全解包参数（避免变量名冲突）
    device = params[0]
    model_name = params[1]
    dataset_name = params[2]
    feature_dim = params[3]
    train_set = params[4]
    compressed_params_g = params[5]
    prev_local_state = params[6]
    wh_state = params[7]
    lr = params[8]
    lr_g = params[9]
    batch_size = params[10]
    epochs = params[11]
    energy_threshold = params[12]
    # print(f"Client on device {device} starting training.")

    # 1. 初始化模型
    model = get_model(model_name, dataset_name).to(device)
    model_g = get_model(model_name, dataset_name).to(device)

    with torch.no_grad():
        # A. 重构并加载全局代理模型参数
        global_state_dict = {}
        for name, param_data in compressed_params_g.items():
            global_state_dict[name] = reconstruct_param(param_data, device)
        model_g.load_state_dict(global_state_dict)

        # B. 加载本地模型参数
        if prev_local_state is not None:
            model.load_state_dict(prev_local_state)
        else:
            # 第一轮：本地模型从全局参数起点开始
            model.load_state_dict(global_state_dict)

    # 2. 初始化特征对齐层（W_h）
    W_h = torch.nn.Linear(feature_dim, feature_dim, bias=False, device=device)
    if wh_state is not None:
        W_h.load_state_dict(wh_state)

    # 3. 优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    optimizer_g = torch.optim.SGD(model_g.parameters(), lr=lr_g)
    optimizer_W = torch.optim.SGD(W_h.parameters(), lr=lr)

    # 4. 训练循环
    loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    model.train()
    model_g.train()
    W_h.train()

    total_loss = 0.0
    num_batches = 0

    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            # 前向传播
            output, rep = model(x)
            output_g, rep_g = model_g(x)

            # 基础交叉熵损失
            loss_ce = ce_loss(output, y)
            loss_ce_g = ce_loss(output_g, y)

            # 互学习知识蒸馏（KL 散度）
            loss_kd = kl_loss(output, output_g.detach())
            loss_kd_g = kl_loss(output_g, output.detach())

            # 特征对齐损失
            loss_h = mse_loss(rep, W_h(rep_g.detach()))
            loss_h_g = mse_loss(rep.detach(), W_h(rep_g))

            # 归一化因子
            scale = loss_ce.item() + loss_ce_g.item() + 1e-8

            loss = loss_ce + loss_kd / scale + loss_h / scale
            loss_g = loss_ce_g + loss_kd_g / scale + loss_h_g / scale

            # 优化步骤
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

    # 5. 压缩全局代理模型用于上传
    avg_loss = total_loss / num_batches
    compressed_params_g_new = {}
    for name, param in model_g.state_dict().items():
        compressed_params_g_new[name] = decompose_param(param, energy_threshold)
    local_state = {k: v.cpu() for k, v in model.state_dict().items()}
    wh_state = {k: v.cpu() for k, v in W_h.state_dict().items()}

    return [avg_loss, compressed_params_g_new, local_state, wh_state]


class Server(BaseServer):
    def __init__(self, args: argparse.Namespace):
        model = get_model(args.model, args.dataset)
        super().__init__(model, True, args)

        # 初始分解
        self.compressed_params = {}
        for name, param in self.model.state_dict().items():
            self.compressed_params[name] = decompose_param(param, args.energy)

        # 存储每个客户端的私有本地模型状态（用于模拟本地持久化）
        self.client_states = [None for _ in range(self.num_clients)]
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
        for r in range(self.rounds):
            t0 = time.time()
            print(f"\n--- FedKD Round {r + 1}/{self.rounds} ---")

            # 向客户端发送压缩后的全局参数
            p = [
                [
                    self.client_gpu[i],
                    self.args.model,
                    self.args.dataset,
                    self.feature_dim,
                    self.train_sets[i],
                    self.compressed_params,
                    self.client_states[i],
                    self.client_wh_states[i],
                    self.args.lr,
                    self.args.lr_g,
                    self.args.batch_size,
                    self.args.epochs,
                    self.args.energy,
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

            total_loss = sum(res[0] for res in results)
            avg_loss = total_loss / self.num_clients
            self.loss.append(avg_loss)

            # 更新服务器端存储的客户端本地状态
            for i, res in enumerate(results):
                self.client_states[i] = res[2]
                self.client_wh_states[i] = res[3]

            # 聚合 SVD 参数 (仅聚合 model_g)
            client_compressed_params_list = [res[1] for res in results]
            self.aggregate_svd(client_compressed_params_list)

            self.evaluate()
            print(f"Global Accuracy: {self.acc[-1]:.2f}%, Avg Loss: {avg_loss:.4f}")
            print(f"Round finished in {time.time() - t0:.2f} seconds")

    def evaluate(self):
        acc = 0.0
        for i in range(self.num_clients):
            model = get_model(self.args.model, self.args.dataset).to(self.device)
            with torch.no_grad():
                model.load_state_dict(self.client_states[i])  # type: ignore
            acc_i = evaluate_model(model, self.test_set[i], self.device)
            acc += acc_i
        self.acc.append(acc / self.num_clients)

    def aggregate_svd(self, client_params_list):
        """聚合 SVD 压缩的参数"""
        # 1. 将所有参数重构到 CPU
        aggregated_state_dict = {}
        ref_params = client_params_list[0]

        # 用第一个客户端初始化
        for name in ref_params.keys():
            param_0 = reconstruct_param(ref_params[name], torch.device("cpu"))
            aggregated_state_dict[name] = param_0 * self.weights[0]

        # 累积其余客户端
        for i in range(1, len(client_params_list)):
            client_params = client_params_list[i]
            for name in client_params.keys():
                param = reconstruct_param(client_params[name], torch.device("cpu"))
                aggregated_state_dict[name] += param * self.weights[i]

        # 2. 更新服务器模型
        self.model.load_state_dict(aggregated_state_dict)

        # 3. 为下一轮分发重新压缩
        self.compressed_params = {}
        for name, param in self.model.state_dict().items():
            self.compressed_params[name] = decompose_param(param, self.args.energy)

    def save(self, test):
        file_name: str = f"{self.args.epochs}_{self.args.batch_size}_{self.args.lr}.pt"
        f = {
            "acc": self.acc,
            "loss": self.loss,
            "state_dict": self.model.state_dict()
        }
        super().deal_save(test, f, file_name)
