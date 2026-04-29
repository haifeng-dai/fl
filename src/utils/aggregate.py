import torch


def param_aggregate(
    state_dicts: list[dict[str, torch.Tensor]],
    weights: list[float],
):
    # 根据首个客户端的数据结构初始化为零的聚合状态字典
    aggregated_state = {
        k: torch.zeros_like(v, device="cpu", dtype=torch.float32)
        for k, v in state_dicts[0].items()
    }

    # 就地累加参数特征: agg += weight * param
    # 这避免了在内存中堆叠存放所有客户端参数而造成的冗余开销 (即将 O(N) 内存占用优化降至 O(1))
    with torch.no_grad():
        for i, state_dict in enumerate(state_dicts):
            w = weights[i]
            for key, param in state_dict.items():
                # 使用带有 alpha 权重的 add_ 方法实现高效的 BLAS AXPY 计算
                if key in aggregated_state:
                    aggregated_state[key].add_(param.cpu(), alpha=w)

    return aggregated_state
