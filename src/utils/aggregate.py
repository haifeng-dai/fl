import torch


def param_aggregate(
    state_dicts: list[dict[str, torch.Tensor]], weights: list[float] | None = None
):
    if weights is None:
        weights = [1.0 / len(state_dicts)] * len(state_dicts)

    weight_tensor = torch.tensor(weights, dtype=torch.float32, device="cpu")
    with torch.no_grad():
        aggregated_state: dict[str, torch.Tensor] = {}
        for key in state_dicts[0].keys():
            stacked = torch.stack([state_dict[key].cpu() for state_dict in state_dicts], dim=0)
            expanded_weights = weight_tensor.view(-1, *([1] * (stacked.ndim - 1)))
            aggregated_state[key] = (stacked * expanded_weights).sum(dim=0)

        return aggregated_state
