from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset


@dataclass
class BaseClientParams:
    client_id: int
    global_state: dict[str, torch.Tensor]
    train_set: Dataset
    payload: Any = None


@dataclass
class ClientResult:
    client_id: int
    loss: float
    state: dict[str, torch.Tensor]
    payload: Any = None


@dataclass
class EvalTask:
    eval_id: int
    model_state: dict[str, torch.Tensor]
    test_set: Dataset
    payload: Any = None


@dataclass(frozen=True)
class EvalResult:
    eval_id: int
    correct: int
    total: int
    payload: Any = None

    @property
    def accuracy(self) -> float:
        return 100.0 * self.correct / self.total if self.total else 0.0


@dataclass(frozen=True)
class ModelParam:
    model: str
    dataset: str
    feature_dim: int
    num_class: int
