import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

from .utils import (
    BaseServer,
    run_parallel_clients,
    evaluate_model,
    get_model,
)


def add_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("FedKD Specific Arguments")
    group.add_argument(
        "--mentee_learning_rate", type=float, default=0.005, help="全局模型（学生）的学习率"
    )
    group.add_argument(
        "--learning_rate_decay_gamma", type=float, default=0.99, help="学习率衰减系数"
    )
    group.add_argument(
        "--energy", type=float, default=0.95, help="SVD能量阈值 (0-1)"
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
    # 分离并移到 CPU 进行 SVD（GPU 上的 SVD 对小矩阵可能较慢/不稳定）
    param_cpu = param.detach().cpu()
    param_shape = param_cpu.shape

    # 检查是否可以进行分解（2D 或 4D 张量）
    # 同时通常跳过 embedding 层
    if len(param_shape) not in [2, 4] or 'embedding' in str(param.dtype):
         return param_cpu

    # 重塑为 2D 矩阵
    if len(param_shape) == 4:
        # 卷积层: (out, in, h, w) -> (out, in*h*w)
        mat = param_cpu.view(param_shape[0], -1)
    else:
        mat = param_cpu

    # 执行 SVD 分解
    # full_matrices=False -> U: (M, K), S: (K,), Vh: (K, N) 其中 K=min(M,N)
    try:
        u, s, vh = torch.linalg.svd(mat, full_matrices=False)
    except RuntimeError:
        # SVD 失败时的回退方案
        return param_cpu

    # 基于能量阈值确定秩
    total_energy = torch.sum(s ** 2)
    if total_energy == 0:
        return param_cpu

    cumulative_energy = torch.cumsum(s ** 2, dim=0)
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
        'u': u_k,
        's': s_k,
        'vh': vh_k,
        'original_shape': param_shape,
        'is_compressed': True
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
    if isinstance(compressed_param, dict) and compressed_param.get('is_compressed'):
        u = compressed_param['u'].to(device)
        s = compressed_param['s'].to(device)
        vh = compressed_param['vh'].to(device)

        # 重构: U * diag(S) * Vh
        mat = u @ (torch.diag(s) @ vh)

        # 重塑回原始形状
        return mat.view(compressed_param['original_shape'])
    elif isinstance(compressed_param, torch.Tensor):
        return compressed_param.to(device)
    else:
        # 数据干净时不应该发生
        raise ValueError(f"未知参数类型: {type(compressed_param)}")


def client_worker(client_id, params):
    device = params[0]
    global_compressed_params = params[1]
    train_set = params[2]
    model_name = params[3]
    dataset_name = params[4]
    lr = params[5]
    batch_size = params[6]
    epochs = params[7]
    mentee_lr = params[8]
    lr_decay_gamma = params[9]
    energy_threshold = params[10]

    # 1. 初始化本地模型（学生/Mentee 1）
    model = get_model(model_name, dataset_name).to(device)

    # 2. 初始化全局模型（教师/Mentee 2）
    global_model = get_model(model_name, dataset_name).to(device)

    # 3. 加载全局参数（必要时从 SVD 重构）
    with torch.no_grad():
        state_dict = {}
        for name, param_data in global_compressed_params.items():
            state_dict[name] = reconstruct_param(param_data, device)

        # 两个模型都用相同的全局权重初始化
        model.load_state_dict(state_dict)
        global_model.load_state_dict(state_dict)

    # 4. 初始化特征对齐层（W_h）
    dummy_input = torch.randn(1, 1 if dataset_name == 'mnist' else 3, 32, 32).to(device)
    if dataset_name == 'mnist':
         dummy_input = torch.randn(1, 1, 28, 28).to(device)

    with torch.no_grad():
        _, feat = model(dummy_input)
        feature_dim = feat.shape[1]

    W_h = nn.Linear(feature_dim, feature_dim, bias=False).to(device)

    # 5. 优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    optimizer_g = torch.optim.SGD(global_model.parameters(), lr=mentee_lr)
    optimizer_W = torch.optim.SGD(W_h.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay_gamma)
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optimizer_g, gamma=lr_decay_gamma)
    scheduler_W = torch.optim.lr_scheduler.ExponentialLR(optimizer_W, gamma=lr_decay_gamma)

    # 6. 损失函数
    kl_loss = nn.KLDivLoss(reduction='batchmean')
    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()

    # 7. 训练循环
    loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    model.train()
    global_model.train()

    total_loss = 0.0
    num_batches = 0

    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            # 前向传播
            output, rep = model(x)
            output_g, rep_g = global_model(x)

            # 基础交叉熵损失
            loss_ce = ce_loss(output, y)
            loss_ce_g = ce_loss(output_g, y)

            # 互学习知识蒸馏（KL 散度）
            loss_kd = kl_loss(F.log_softmax(output, dim=1), F.softmax(output_g.detach(), dim=1))
            loss_kd_g = kl_loss(F.log_softmax(output_g, dim=1), F.softmax(output.detach(), dim=1))

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

            # 梯度裁剪
            nn.utils.clip_grad_norm_(model.parameters(), 10)
            nn.utils.clip_grad_norm_(global_model.parameters(), 10)
            nn.utils.clip_grad_norm_(W_h.parameters(), 10)

            optimizer.step()
            optimizer_g.step()
            optimizer_W.step()

            total_loss += loss.item()
            num_batches += 1

    # 更新学习率调度器
    scheduler.step()
    scheduler_g.step()
    scheduler_W.step()

    avg_loss = total_loss / num_batches

    # 8. 用于通信的 SVD 分解
    compressed_params = {}
    for name, param in global_model.named_parameters():
        compressed_params[name] = decompose_param(param, energy_threshold)

    return client_id, [avg_loss, compressed_params]


class Server(BaseServer):
    def __init__(self, args: argparse.Namespace):
        model = get_model(args.model, args.dataset)
        super().__init__(model, False, args)

        # 初始分解
        self.compressed_params = {}
        for name, param in self.model.named_parameters():
            self.compressed_params[name] = decompose_param(param, args.energy)

    def fit(self):
        for r in range(self.rounds):
            print(f"\n--- FedKD Round {r + 1}/{self.rounds} ---")

            # 准备参数
            # 向客户端发送压缩后的全局参数
            parameters_per_client = [
                [
                    self.client_gpu[i],
                    self.compressed_params,
                    self.train_sets[i],
                    self.args.model,
                    self.args.dataset,
                    self.args.lr,
                    self.args.batch_size,
                    self.args.epochs,
                    self.args.mentee_learning_rate,
                    self.args.learning_rate_decay_gamma,
                    self.args.energy
                ]
                for i in range(self.num_clients)
            ]

            results = run_parallel_clients(
                client_worker=client_worker,
                num_clients=self.num_clients,
                parameters=parameters_per_client,
                gpu_pools=self.gpu_pools,
                mp=self.mp,
            )

            total_loss = sum(res[0] for res in results)
            avg_loss = total_loss / self.num_clients
            self.loss.append(avg_loss)

            # 聚合 SVD 参数
            client_compressed_params_list = [res[1] for res in results]
            self.aggregate_svd(client_compressed_params_list, self.weights)

            self.evaluate()
            print(f"Global Accuracy: {self.acc[-1]:.2f}%, Avg Loss: {avg_loss:.4f}")

    def aggregate_svd(self, client_params_list, weights):
        """聚合 SVD 压缩的参数"""
        # 1. 将所有参数重构到 CPU
        aggregated_state_dict = {}
        ref_params = client_params_list[0]

        # 用第一个客户端初始化
        for name in ref_params.keys():
            param_0 = reconstruct_param(ref_params[name], torch.device('cpu'))
            aggregated_state_dict[name] = param_0 * weights[0]

        # 累积其余客户端
        for i in range(1, len(client_params_list)):
            client_params = client_params_list[i]
            w = weights[i]
            for name in client_params.keys():
                param = reconstruct_param(client_params[name], torch.device('cpu'))
                aggregated_state_dict[name] += param * w

        # 2. 更新服务器模型
        self.model.load_state_dict(aggregated_state_dict)

        # 3. 为下一轮分发重新压缩
        self.compressed_params = {}
        for name, param in self.model.named_parameters():
            self.compressed_params[name] = decompose_param(param, self.args.energy)

    def evaluate(self):
        self.acc.append(evaluate_model(self.model, self.test_set, self.device))

    def save(self, test):
        file_name: str = f"{self.args.epochs}_{self.args.batch_size}_{self.args.lr}.pt"
        f = {"acc": self.acc, "loss": self.loss, "state_dict": self.model.state_dict()}
        super().deal_save(test, f, file_name)