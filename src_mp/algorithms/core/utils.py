import torch


def clone_state(state):
    return {key: value.detach().cpu().clone() for key, value in state.items()}


def aggregate_weighted(states, weights):
    aggregate = {
        key: torch.zeros_like(value, dtype=torch.float32)
        for key, value in states[0].items()
    }
    for state, weight in zip(states, weights, strict=True):
        for key, value in state.items():
            aggregate[key].add_(value.cpu(), alpha=weight)
    return aggregate
