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
    weights: list[float] = None,
    old_global_protos: torch.Tensor = None,
):
    """
    针对 [C, D] 张量格式的统一原型聚合函数。
    支持类别掩码，防止缺失类别导致的数值稀释。
    """
    num_clients = len(local_protos_list)
    device = "cpu"  # 强制在 CPU 执行聚合

    if weights is None:
        weights = torch.full((num_clients,), 1.0 / num_clients, device=device)
    else:
        weights = torch.tensor(weights, device=device)

    # 1. 堆叠所有客户端的原型并移至 CPU: [N, C, D]
    stack_protos = torch.stack([p.cpu() for p in local_protos_list])

    # 2. 生成掩码：标识哪些类别的原型是非零的 (即该客户端拥有该类别)
    # 假设全零行代表缺失类别
    mask = (torch.norm(stack_protos, dim=2) > 1e-8).float()  # [N, C]

    # 3. 计算加权和
    # weights_expanded: [N, 1] -> [N, C]
    weights_matrix = weights.unsqueeze(1) * mask  # [N, C]

    # weight_sums: 每个类别收到的总权重 [C, 1]
    weight_sums = weights_matrix.sum(dim=0, keepdim=True).T  # [C, 1]

    # 聚合结果 [C, D]
    # 使用 einsum 进行高效加权求和: n,nc,ncd->cd
    sum_protos = torch.einsum('nc,ncd->cd', weights_matrix, stack_protos)

    # 4. 归一化（加权平均）
    # 避免除以零
    safe_weight_sums = torch.where(weight_sums > 0, weight_sums, torch.ones_like(weight_sums))
    new_global_protos = sum_protos / safe_weight_sums

    # 5. 补全逻辑：如果本轮没有任何客户端拥有某类别，则保留旧全局原型
    if old_global_protos is not None:
        old_global_protos = old_global_protos.cpu()
        missing_mask = (weight_sums.squeeze() == 0)
        new_global_protos[missing_mask] = old_global_protos[missing_mask]

    return new_global_protos
