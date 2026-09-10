from collections.abc import Iterator
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, RandomSampler, TensorDataset


@dataclass
class FixMatchLoaders:
    """固定 ``B:μB`` 本地训练协议所需的双数据流。"""

    labeled_loader: DataLoader
    unlabeled_loader: DataLoader
    steps_per_epoch: int
    labeled_count: int
    unlabeled_count: int
    labeled_batch_size: int
    unlabeled_batch_size: int


def build_fixmatch_loaders(
    train_set,
    batch_size: int,
    unlabeled_ratio: int,
) -> FixMatchLoaders:
    """从客户端训练集构建固定 ``B:μB`` 的有/无标签数据流。

    有标签流仅使用 ``is_labeled=True`` 的样本；无监督流使用完整本地训练集
    ``L ∪ U``，忽略其中样本的标签。两个 loader 均在
    单次遍历内无放回随机采样，迭代器重启由 :func:`iterate_fixmatch_batches` 处理。

    ``steps_per_epoch`` 以无标签池大小和监督 batch 大小计算，匹配
    SAGE/ProxyFL 等 FixMatch 式实现的本地训练预算。
    """
    if batch_size <= 0:
        raise ValueError("batch_size 必须为正整数")
    if unlabeled_ratio <= 0:
        raise ValueError("unlabeled_ratio 必须为正整数")

    labeled_mask = train_set.is_labeled.bool()
    labeled_indices = torch.where(labeled_mask)[0]
    unlabeled_indices = torch.arange(len(train_set.y))
    labeled_count = len(labeled_indices)
    unlabeled_count = len(unlabeled_indices)
    unlabeled_batch_size = batch_size * unlabeled_ratio

    labeled_data = TensorDataset(
        train_set.x[labeled_indices], train_set.y[labeled_indices]
    )
    unlabeled_data = TensorDataset(
        train_set.x[unlabeled_indices], train_set.y[unlabeled_indices]
    )
    steps_per_epoch = max(1, unlabeled_count // batch_size)
    labeled_sampler = RandomSampler(
        labeled_data,
        replacement=labeled_count < batch_size,
        num_samples=(
            max(batch_size, steps_per_epoch * batch_size)
            if labeled_count < batch_size
            else None
        ),
    )
    unlabeled_sampler = None
    if unlabeled_count < unlabeled_batch_size:
        unlabeled_sampler = RandomSampler(
            unlabeled_data,
            replacement=True,
            num_samples=steps_per_epoch * unlabeled_batch_size,
        )
    labeled_loader = DataLoader(
        labeled_data,
        batch_size=batch_size,
        shuffle=False,
        sampler=labeled_sampler,
        drop_last=True,
    )
    return FixMatchLoaders(
        labeled_loader=labeled_loader,
        unlabeled_loader=DataLoader(
            unlabeled_data,
            batch_size=unlabeled_batch_size,
            shuffle=unlabeled_sampler is None,
            sampler=unlabeled_sampler,
            drop_last=True,
        ),
        steps_per_epoch=steps_per_epoch,
        labeled_count=labeled_count,
        unlabeled_count=unlabeled_count,
        labeled_batch_size=batch_size,
        unlabeled_batch_size=unlabeled_batch_size,
    )


def iterate_fixmatch_batches(
    loaders: FixMatchLoaders,
) -> Iterator[
    tuple[
        tuple[torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor],
    ]
]:
    """产出一个 local epoch 的成对 batch，并在流耗尽后重启迭代器。"""
    labeled_iterator = iter(loaders.labeled_loader)
    unlabeled_iterator = iter(loaders.unlabeled_loader)

    for _ in range(loaders.steps_per_epoch):
        try:
            labeled_batch = next(labeled_iterator)
        except StopIteration:
            labeled_iterator = iter(loaders.labeled_loader)
            labeled_batch = next(labeled_iterator)

        try:
            unlabeled_batch = next(unlabeled_iterator)
        except StopIteration:
            unlabeled_iterator = iter(loaders.unlabeled_loader)
            unlabeled_batch = next(unlabeled_iterator)

        yield labeled_batch, unlabeled_batch
