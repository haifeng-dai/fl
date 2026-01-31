import torch


def param_aggregate(
    state_dicts: list[dict[str, torch.Tensor]],
    weights: list[float],
):
    # Initialize aggregated_state with zeros based on the structure of the first client
    # Use CPU to save GPU memory for training
    aggregated_state = {
        k: torch.zeros_like(v, device="cpu", dtype=torch.float32)
        for k, v in state_dicts[0].items()
    }

    # Accumulate parameters in-place: agg += weight * param
    # This avoids creating a large stack of all client parameters (O(N) memory -> O(1) memory)
    with torch.no_grad():
        for i, state_dict in enumerate(state_dicts):
            w = weights[i]
            for key, param in state_dict.items():
                # Use add_ with alpha for efficient BLAS AXPY operation
                if key in aggregated_state:
                    aggregated_state[key].add_(param.cpu(), alpha=w)

    return aggregated_state
