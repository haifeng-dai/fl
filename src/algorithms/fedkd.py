import os
import time

import torch

from .utils import (
    BaseServer,
    fmt_num,
    ce_loss,
    get_model,
    kl_loss,
    mse_loss,
)


def get_path(args):
    args.file_name = f"{args.common_name}_{fmt_num(args.lr_g)}_{fmt_num(args.energy)}"
    return os.path.join(args.log_path, f"{args.file_name}_{args.cur_time}.log")


def decompose_param(param, energy_threshold):
    """
    基于能量阈值 (Energy Threshold) 使用 SVD 分解单个参数向量。

    Args:
        param: 待分解的参数向量
        energy_threshold: 能量阈值 (0-1)

    Returns:
        compressed_param: 压缩后的参数 (字典或张量)
    """
    # 保持在原生设备上处理，不用强制移至 CPU
    param_shape = param.shape

    # 检查分解是否可行（仅支持 2D 或 4D tensor）
    # 通常我们也跳过 embedding 层
    if len(param_shape) not in [2, 4] or "embedding" in str(param.dtype):
        return param.detach().cpu()

    # 将其重置为 2D 矩阵
    if len(param_shape) == 4:
        # 卷积层: (out, in, h, w) -> (out, in*h*w)
        mat = param.view(param_shape[0], -1)
    else:
        mat = param

    # 执行 SVD 分解运算
    try:
        # 优先在原始设备上执行 (例如, GPU 显存)
        u, s, vh = torch.linalg.svd(mat, full_matrices=False)
    except RuntimeError:
        # SVD 失败时的回退机制 (例如, 显存溢出 OOM)，将回退至 CPU
        mat = mat.cpu()
        try:
            u, s, vh = torch.linalg.svd(mat, full_matrices=False)
        except RuntimeError:
            return param.detach().cpu()

    # 根据能量阈值决定秩 (Rank)
    total_energy = torch.sum(s**2)
    if total_energy == 0:
        return param.detach().cpu()

    cumulative_energy = torch.cumsum(s**2, dim=0)
    # 找到累计能量超过 (总能量 * 阈值) 的第一个索引
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
    从压缩表示中重建原本的模型参数。

    Args:
        compressed_param: 压缩后的参数 (来自 decompose_param)
        device: 目标设备

    Returns:
        重建后的参数张量
    """
    if isinstance(compressed_param, dict) and compressed_param.get("is_compressed"):
        u = compressed_param["u"].to(device)
        s = compressed_param["s"].to(device)
        vh = compressed_param["vh"].to(device)

        # 重建矩阵: U * diag(S) * Vh
        mat = u @ (torch.diag(s) @ vh)

        # 重塑回原始形状
        return mat.view(compressed_param["original_shape"])
    elif isinstance(compressed_param, torch.Tensor):
        return compressed_param.to(device)
    else:
        # 在数据无损情况下应不会走到这步
        raise ValueError(f"Unknown parameter type: {type(compressed_param)}")


def train(params):
    """
    FedKD 本地训练流程，采用基于 SVD 的通信压缩与相互知识蒸馏机制。
    """
    # 安全解包参数
    (
        _,
        device,
        compressed_params_g,
        train_set,
        model_name,
        dataset_name,
        lr,
        batch_size,
        epochs,
        feature_dim,
        num_class,
        prev_local_state,
        wh_state,
        lr_g,
        energy_threshold,
    ) = params

    # 1. 初始化模型
    # 本地个性化专家模型 (Student)
    model = get_model(model_name, dataset_name, num_class, feature_dim).to(device)
    # 全局代理模型 (从压缩的 SVD 参数重建)
    model_g = get_model(model_name, dataset_name, num_class, feature_dim).to(device)

    with torch.no_grad():
        # A. 从 SVD 参数中重建并加载全局代理模型参数
        global_state_dict = {}
        for name, param_data in compressed_params_g.items():
            global_state_dict[name] = reconstruct_param(param_data, device)
        model_g.load_state_dict(global_state_dict)

        # B. 加载本地模型参数
        if prev_local_state is not None:
            model.load_state_dict(prev_local_state)
        else:
            # 首轮训练：从全局状态起始
            model.load_state_dict(global_state_dict)

    # 2. 初始化特征对齐层 (W_h)
    W_h = torch.nn.Linear(feature_dim, feature_dim, bias=False, device=device)
    if wh_state is not None:
        W_h.load_state_dict(wh_state)

    # 3. 初始化优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    optimizer_g = torch.optim.SGD(model_g.parameters(), lr=lr_g)
    optimizer_W = torch.optim.SGD(W_h.parameters(), lr=lr)

    # 4. 训练循环 (Mutual Knowledge Distillation)
    loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    model.train()
    model_g.train()
    W_h.train()

    total_loss = 0.0
    num_batches = 0

    for _ in range(epochs):
        for x, y, *_ in loader:
            x, y = x.to(device), y.to(device)
            rep = model.extractor(x)
            output = model.classifier(rep)
            rep_g = model_g.extractor(x)
            output_g = model_g.classifier(rep_g)

            # 基础任务预测损失 (Cross Entropy)
            loss_ce = ce_loss(output, y)
            loss_ce_g = ce_loss(output_g, y)

            # 互相知识蒸馏 (KL 散度)
            loss_kd = kl_loss(output, output_g.detach())
            loss_kd_g = kl_loss(output_g, output.detach())

            # 特征对齐损失
            loss_h = mse_loss(rep, W_h(rep_g.detach()))
            loss_h_g = mse_loss(rep.detach(), W_h(rep_g))

            # 放缩归一化因子
            scale = loss_ce.item() + loss_ce_g.item() + 1e-8

            # 最终整体损失
            loss = loss_ce + loss_kd / scale + loss_h / scale
            loss_g = loss_ce_g + loss_kd_g / scale + loss_h_g / scale

            # 反向传播与优化器更新
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

    # 5. 使用 SVD 压缩更新后的全局模型用于上行通信
    avg_loss = total_loss / num_batches
    compressed_params_g_new = {}
    for name, param in model_g.state_dict().items():
        compressed_params_g_new[name] = decompose_param(param, energy_threshold)

    # 准备返回状态数据
    local_state = {k: v.cpu().detach().clone() for k, v in model.state_dict().items()}
    wh_state = {k: v.cpu().detach().clone() for k, v in W_h.state_dict().items()}

    return {
        "loss": avg_loss,
        "compressed": compressed_params_g_new,
        "state": local_state,
        "wh": wh_state,
    }


class Server(BaseServer):
    def __init__(self, args):
        super().__init__(True, args)

        # 初始分解运算（迁移到 GPU 上执行 SVD）
        self.compressed_params = {}
        for name, param in self.model.state_dict().items():
            self.compressed_params[name] = decompose_param(
                param.to(self.device), args.energy
            )

        self.client_wh_states = [None for _ in range(self.num_clients)]

    def fit(self):
        num_join = max(1, int(self.num_clients * self.args.join_ratio))

        for r in range(self.rounds):
            t0 = time.time()
            print(f"\n--- FedKD Round {r + 1}/{self.rounds} ---")

            selected = torch.randperm(self.num_clients)[:num_join].tolist()
            print(f"Selected clients: {selected}")

            p = self.build_base_params(selected)
            for params, i in zip(p, selected):
                params[2] = self.compressed_params
                params.append(self.clients_state[i])
                params.append(self.client_wh_states[i])
                params.append(self.args.lr_g)
                params.append(self.args.energy)
            results = self.run_clients(train, p)

            # 汇集各客户端回传结果并更新服务器端存储的客户端本地状态
            total_loss = 0.0
            client_compressed_params_list = []
            current_weights = []
            for cid, res in results.items():
                total_loss += res["loss"]
                client_compressed_params_list.append(res["compressed"])
                self.clients_state[cid] = res["state"]
                self.client_wh_states[cid] = res["wh"]
                current_weights.append(self.weights[cid])
            self.loss.append(total_loss / num_join)
            sum_weights = sum(current_weights)
            norm_weights = [w / sum_weights for w in current_weights]

            self.aggregate_svd(client_compressed_params_list, weights=norm_weights)

            self.evaluate()
            print(
                f"Global Accuracy: {self.acc[-1]:.2f}%, Avg Loss: {self.loss[-1]:.4f}"
            )
            print(f"Round finished in {time.time() - t0:.2f} seconds")

    def aggregate_svd(self, client_params_list, weights):
        """聚合通过 SVD 压缩的模型参数"""
        # 1. 在 GPU 上重建并加权聚合所有参数
        aggregated_state_dict = {}
        ref_params = client_params_list[0]

        # 从首个客户端开始初始化
        for name in ref_params.keys():
            param_0 = reconstruct_param(ref_params[name], self.device)
            aggregated_state_dict[name] = param_0 * weights[0]

        # 累加剩余的客户端数据
        for i in range(1, len(client_params_list)):
            client_params = client_params_list[i]
            for name in client_params.keys():
                param = reconstruct_param(client_params[name], self.device)
                aggregated_state_dict[name] += param * weights[i]

        # 2. 更新服务端全局模型（CPU 上持有模型）
        self.model.load_state_dict(
            {k: v.cpu() for k, v in aggregated_state_dict.items()}
        )

        # 3. 为下一轮次分发重新进行压缩（迁移到 GPU 上执行 SVD）
        self.compressed_params = {}
        for name, param in self.model.state_dict().items():
            self.compressed_params[name] = decompose_param(
                param.to(self.device), self.args.energy
            )

    def save(self):
        metrics = {"acc": self.acc, "loss": self.loss}
        params = {"global": self.model.state_dict(), "client": self.clients_state, "aux": {"client_wh_states": self.client_wh_states}}
        self.deal_save(metrics, params)
