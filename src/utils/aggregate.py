import torch


def param_aggregate(
    state_dicts: list[dict[str, torch.Tensor]], weights: list[float] | None = None
):
    if weights is None:
        weights = [1.0 / len(state_dicts)] * len(state_dicts)

    with torch.no_grad():
        temp_state: dict[str, torch.Tensor] = {}
        for key in state_dicts[0].keys():
            temp_state[key] = torch.zeros_like(state_dicts[0][key])

        for key in temp_state.keys():
            temp = torch.zeros_like(temp_state[key])
            for i, state_dict in enumerate(state_dicts):
                temp += state_dict[key].cpu() * weights[i]
            temp_state[key].copy_(temp)

    return temp_state
