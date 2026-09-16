from __future__ import annotations

import abc

import torch

from src.algorithms.utils.input import prepare_input_batch

from .model import build_model
from .protocol import BaseClientParams, ClientResult, EvalResult, EvalTask, ModelParam
from .utils import clone_state


class BaseClientExecutor(abc.ABC):
    """每个 GPU worker 一个实例；复用模型、优化器和客户端 DataLoader。"""

    def __init__(self, args, device: torch.device, num_class: int):
        self.device, self.num_class = device, num_class
        self.is_ssl = args.ssl != "none"
        self.dataset = args.dataset
        self.batch_size: int = args.batch_size
        self.lr: float = args.lr
        self.momentum: float = args.momentum
        self.weight_decay: float = args.weight_decay
        self.epochs: int = args.epochs
        self.model_param = ModelParam(
            args.model,
            args.dataset,
            args.feature_dim,
            num_class,
        )

        self.model = build_model(self.model_param).to(device)
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
        self.loaders: dict[int, torch.utils.data.DataLoader] = {}
        self.eval_loaders = {}
        self.current_task: BaseClientParams | EvalTask
        self.client_id: int
        self.eval_id: int
        self.loader: torch.utils.data.DataLoader
        self.payload = None

    def _loader(self, task: BaseClientParams) -> torch.utils.data.DataLoader:
        if task.client_id not in self.loaders:
            loader = torch.utils.data.DataLoader(
                task.train_set,
                batch_size=self.batch_size,
                shuffle=True,
            )
            self.loaders[task.client_id] = loader
        return self.loaders[task.client_id]

    def _reset_optimizer(self) -> None:
        for group in self.optimizer.param_groups:
            group.update(
                lr=self.lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
        self.optimizer.state.clear()

    def run(self, task: BaseClientParams) -> ClientResult:
        self.current_task = task
        self.client_id = task.client_id
        self.loader = self._loader(task)
        self.payload = task.payload
        self.model.load_state_dict(task.global_state)
        self._reset_optimizer()
        return self.train()

    def evaluate(self, task: EvalTask) -> EvalResult:
        if task.eval_id not in self.eval_loaders:
            self.eval_loaders[task.eval_id] = torch.utils.data.DataLoader(
                task.test_set, batch_size=128, shuffle=False
            )
        self.current_task = task
        self.eval_id = task.eval_id
        self.payload = task.payload
        self.model.load_state_dict(task.model_state)
        self.model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y, *_ in self.eval_loaders[task.eval_id]:
                inputs = x.to(self.device)
                if self.is_ssl:
                    inputs = prepare_input_batch(inputs, self.dataset)
                labels = y.to(self.device)
                predictions = self.model(inputs).argmax(dim=1)
                correct += predictions.eq(labels).sum().item()
                total += y.size(0)
        return EvalResult(task.eval_id, correct, total)

    def check_nan(self, loss):
        if not torch.isfinite(loss):
            raise FloatingPointError(f"client {self.client_id}: non-finite loss")

    def train(self):
        self.model.train()
        total, batches = 0.0, 0
        for _ in range(self.epochs):
            total, batches = self.run_epoch(total, batches)
        return ClientResult(
            self.client_id,
            total / max(1, batches),
            clone_state(self.model.state_dict()),
        )

    @abc.abstractmethod
    def run_epoch(self, total, batches) -> tuple[float, int]: ...
