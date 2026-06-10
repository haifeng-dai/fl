import torch


def param_aggregate(
    state_dicts: list[dict[str, torch.Tensor]],
    weights: list[float],
):
    # 初始化聚合状态字典
    aggregated_state = {
        k: torch.zeros_like(v, device="cpu", dtype=torch.float32)
        for k, v in state_dicts[0].items()
    }

    # 累加参数特征
    with torch.no_grad():
        for i, state_dict in enumerate(state_dicts):
            w = weights[i]
            for key, param in state_dict.items():
                if key in aggregated_state:
                    aggregated_state[key].add_(param.cpu(), alpha=w)

    return aggregated_state


def proto_aggregate(
    local_protos_list: list[torch.Tensor],
    local_counts_list: list[torch.Tensor] = None,
    old_global_protos: torch.Tensor = None,
):
    """
    针对 [C, D] 张量格式的统一原型聚合函数。
    """
    # 1. 堆叠所有客户端的原型并移至 CPU: [N, C, D]
    stack_protos = torch.stack([p.cpu() for p in local_protos_list])

    # 2. 生成掩码：标识哪些类别的原型是非零的 (即该客户端拥有该类别)
    # 假设全零行代表缺失类别
    mask = (torch.norm(stack_protos, dim=2) > 1e-8).float()  # [N, C]

    # 3. 计算权重：优先使用样本计数，否则使用客户端权重
    if local_counts_list is not None:
        # 按样本计数计算权重
        stack_counts = torch.stack(
            [c.cpu().to(torch.float32) for c in local_counts_list]
        )
        # 只在有样本的类别应用计数
        weights_matrix = stack_counts * mask  # [N, C]
    else:
        # 普通的平均聚合：对所有拥有该类别的客户端原型进行简单平均
        weights_matrix = mask  # [N, C]

    # weight_sums: 每个类别收到的总权重 [C, 1]
    weight_sums = weights_matrix.sum(dim=0, keepdim=True).T  # [C, 1]

    # 聚合结果 [C, D]
    # 使用 einsum 进行高效加权求和: n,nc,ncd->cd
    sum_protos = torch.einsum("nc,ncd->cd", weights_matrix, stack_protos)

    # 4. 归一化（加权平均）
    # 避免除以零
    safe_weight_sums = torch.where(
        weight_sums > 0, weight_sums, torch.ones_like(weight_sums)
    )
    new_global_protos = sum_protos / safe_weight_sums

    # 5. 补全逻辑：如果本轮没有任何客户端拥有某类别，则保留旧全局原型
    if old_global_protos is not None:
        old_global_protos = old_global_protos.cpu()
        missing_mask = weight_sums.squeeze() == 0
        new_global_protos[missing_mask] = old_global_protos[missing_mask]

    return new_global_protos


def pushsum_param_aggregate(
    state_dicts: list[dict[str, torch.Tensor]],
    weights: torch.Tensor,
    M: torch.Tensor,
    gossip_rounds: int = 1,
    prefix: str = None,
):
    """
    使用矩阵运算形式的 Push-Sum 机制聚合模型参数字典列表。

    Args:
        state_dicts: 客户端参数字典列表 (state_dict)
        weights: 初始权重张量 [N, 1]
        M: 混合矩阵 [N, N] (列随机)
        gossip_rounds: 迭代轮数
        prefix: 仅聚合以该前缀开头的键（例如 'extractor.'），为 None 则聚合全部

    Returns:
        new_state_dicts: 聚合后的参数字典列表
        new_weights: 演化后的权重张量 [N, 1]
    """
    num_clients = len(state_dicts)
    if num_clients == 0:
        return [], weights

    # 1. 过滤并记录参数结构
    if isinstance(prefix, str):
        target_keys = [k for k in state_dicts[0].keys() if k.startswith(prefix)]
    elif isinstance(prefix, (list, tuple)):
        target_keys = [
            k for k in state_dicts[0].keys() 
            if any(k.startswith(p) for p in prefix)
        ]
    else:
        target_keys = list(state_dicts[0].keys())


    param_info = []
    total_size = 0
    for k in target_keys:
        shape = state_dicts[0][k].shape
        size = state_dicts[0][k].numel()
        param_info.append((k, shape, total_size, total_size + size))
        total_size += size

    # 2. 扁平化所有客户端参数到矩阵 [N, total_size]
    device = M.device
    # 使用 float32 保证精度与 param_aggregate 一致
    S_flat = torch.zeros(num_clients, total_size, device=device, dtype=torch.float32)
    W_flat = weights.to(device).view(num_clients, 1).to(torch.float32)

    for i in range(num_clients):
        vec = torch.cat([state_dicts[i][k].view(-1) for k in target_keys])
        # 核心修复：输入参数是物理参数 theta，需要转换为“质量” S = theta * w 参与 Gossip
        # 否则如果 w < 1，每一轮聚合都会导致参数值被错误地放大 1/w 倍，最终导致 NaN
        S_flat[i] = vec.to(device) * W_flat[i]

    # 3. 矩阵形式执行多轮 Push-Sum 迭代
    with torch.no_grad():
        for _ in range(gossip_rounds):
            S_flat = torch.mm(M, S_flat)
            W_flat = torch.mm(M, W_flat)

    # 4. 计算共识值 (S / W)
    consensus_flat = S_flat / (W_flat + 1e-12)

    # 5. 恢复参数形状并返回
    consensus_flat = consensus_flat.cpu()
    W_flat = W_flat.cpu()

    new_state_dicts = []
    for i in range(num_clients):
        client_state = {}
        for k, shape, start, end in param_info:
            client_state[k] = consensus_flat[i, start:end].view(shape).clone()
        new_state_dicts.append(client_state)

    return new_state_dicts, W_flat


def flattened_matrix_aggregate(
    state_dicts: list[dict[str, torch.Tensor]],
    weight_matrix: torch.Tensor,
    device: torch.device,
) -> list[dict[str, torch.Tensor]]:
    """
    GPU 矩阵化聚合：S' = W @ S，与 CPU param_aggregate 逐比特一致。

    精确复制 CPU param_aggregate 的 add_(alpha=w) 逻辑在 GPU 上执行，
    确保与 CPU 版本输出完全一致。

    Args:
        state_dicts: N 个客户端 CPU 参数字典列表
        weight_matrix: [N, N] 权重矩阵（已在 device 上）
        device: 计算设备

    Returns:
        N 个 CPU 上的聚合后参数字典列表
    """
    N = len(state_dicts)
    if N == 0:
        return []

    keys = list(state_dicts[0].keys())
    total_size = sum(state_dicts[0][k].numel() for k in keys)

    # 1. Flatten → [N, total_size] on GPU
    S_flat = torch.zeros(N, total_size, device=device, dtype=torch.float32)
    for i in range(N):
        offset = 0
        for k in keys:
            t = state_dicts[i][k].to(device, non_blocking=True)
            n = t.numel()
            S_flat[i, offset:offset + n].copy_(t.reshape(-1))
            offset += n

    # 2. 逐客户端累加：new[i] = sum_j W[i,j] * S_flat[j]
    # 使用 add_(alpha=w) 匹配 CPU param_aggregate 的逐比特行为
    Wd = weight_matrix.to(device)
    new_flat = torch.zeros(N, total_size, device=device, dtype=torch.float32)
    for i in range(N):
        for j in range(N):
            new_flat[i].add_(S_flat[j], alpha=Wd[i, j].item())

    # 3. Unflatten → CPU dicts
    new_dicts = []
    for i in range(N):
        d = {}
        offset = 0
        for k in keys:
            shape = state_dicts[0][k].shape
            n = state_dicts[0][k].numel()
            d[k] = new_flat[i, offset:offset + n].view(shape).cpu()
            offset += n
        new_dicts.append(d)

    return new_dicts
