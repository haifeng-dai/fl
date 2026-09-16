from __future__ import annotations

import abc

import torch

from src.algorithms.utils.input import prepare_input_batch

from .data import SSLDataLoaderRegistry, SSLStream
from .model import build_model
from .protocol import BaseClientParams, ClientResult, EvalResult, EvalTask, ModelParam
from .utils import clone_state


class BaseClientExecutor(abc.ABC):
    """客户端训练执行器基类。

    每个 GPU worker 只创建一个执行器实例，并在多个任务之间复用模型、
    优化器和 DataLoader。Server 每轮把一个 ``BaseClientParams`` 任务交给
    ``run``；子类通常只需要实现 ``run_epoch``，如果算法有额外状态或训练
    阶段，再覆写 ``train``、``evaluate`` 或增加自己的 payload 处理逻辑。

    子类实现时需要注意：

    * ``self.model`` 是当前客户端的本地模型；
    * ``self.loader`` 是当前任务对应客户端的数据加载器；
    * ``self.payload`` 是 Server 通过任务传入的算法专用状态；
    * ``self.client_id`` 是当前任务的客户端编号；
    * ``run`` 会在每次训练前加载全局或个性化模型状态并重置优化器。
    """

    def __init__(
        self,
        args,
        device: torch.device,
        num_class: int,
        train_sets: dict | None = None,
        **kwargs,
    ):
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

        self.train_sets = train_sets
        self.ssl_registry: SSLDataLoaderRegistry | None = None
        if self.is_ssl and train_sets is not None:
            self.ssl_registry = SSLDataLoaderRegistry(
                train_sets=train_sets,
                batch_size=self.batch_size,
                unlabeled_ratio=int(getattr(args, "unlabeled_ratio", 1)),
            )

    def get_ssl_loaders(self) -> SSLStream:
        """获取当前客户端的半监督双流加载器 (SSLStream)。"""
        if self.ssl_registry is None:
            raise RuntimeError(
                f"client {self.client_id}: SSLDataLoaderRegistry 未初始化，请检查是否传入了 train_sets"
            )
        return self.ssl_registry.get(self.client_id)

    def _loader(self, task: BaseClientParams):
        """获取客户端训练 DataLoader，并按 client_id 缓存复用。"""
        if task.client_id not in self.loaders:
            loader = torch.utils.data.DataLoader(
                task.train_set,
                batch_size=self.batch_size,
                shuffle=True,
            )
            self.loaders[task.client_id] = loader
        return self.loaders[task.client_id]

    def reset_optimizer(self):
        """恢复本任务的优化器超参数并清空动量等历史状态。"""
        for group in self.optimizer.param_groups:
            group.update(
                lr=self.lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
        self.optimizer.state.clear()

    def run(self, task: BaseClientParams):
        """执行一个客户端训练任务。

        该方法负责通用的任务装载流程，通常不需要子类覆写。算法专用的
        payload 会保存到 ``self.payload``，随后调用子类的 ``train``。
        """
        self.current_task = task
        self.client_id = task.client_id
        self.loader = self._loader(task)
        self.payload = task.payload
        self.model.load_state_dict(task.global_state)
        self.reset_optimizer()
        return self.train()

    def evaluate(self, task: EvalTask):
        """在测试集上评估模型并返回正确数、样本总数和任务编号。

        SSL 数据在这里会从原始 uint8 图像转换为模型输入；普通数据则沿用
        已处理的输入。需要额外评估指标的算法可以覆写此方法，并把附加结果
        放入 ``EvalResult.payload``。
        """
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
        """检查损失是否为有限值，发现 NaN 或 Inf 时立即终止当前任务。"""
        if not torch.isfinite(loss):
            raise FloatingPointError(f"client {self.client_id}: non-finite loss")

    def train(self):
        """执行默认的本地训练循环。

        默认按 ``self.epochs`` 次调用 ``run_epoch``，并返回本地平均 loss、
        客户端模型状态和空 payload。需要多阶段训练的算法可以覆写此方法，
        但仍应返回 ``ClientResult``。
        """
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
    def run_epoch(self, total, batches) -> tuple[float, int]:
        """执行一个本地 epoch，并返回累计 loss 与 batch 数。"""
        raise NotImplementedError
