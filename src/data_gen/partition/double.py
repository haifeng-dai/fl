"""双异质半监督数据划分。

该模块将全局训练集按类别拆分为有标签和无标签两部分，然后分别使用
统一的类别划分逻辑分配到客户端。它不改变 ``sample``、``client`` 或
``sfd`` 场景的行为。
"""

import os

import numpy as np
import torch

from .common import (
    distribute_by_class,
    _ensure_nonempty_client_indices,
    _has_mixed_ssl_clients,
    get_output_dir,
    is_fresh,
    save_client_data,
)
from .label import split_indices_by_class


def split_labeled_unlabeled_by_class(
    train_indices_by_class,
    label_ratio,
    rng,
):
    """按类别比例拆分全局训练索引。

    ``label_ratio`` 表示每个类别的有标签比例。对于非零比例但样本数
    很少的类别，至少保留一个有标签样本；比例为零时不强制保留样本。
    """
    if not (0 < label_ratio < 1):
        raise ValueError("混合 SSL (double) 要求 label_ratio 必须位于 (0, 1) 开区间")

    labeled_by_class = []
    unlabeled_by_class = []
    for class_indices in train_indices_by_class:
        indices = np.asarray(class_indices, dtype=int).copy()
        rng.shuffle(indices)
        if label_ratio == 0 or len(indices) == 0:
            labeled_count = 0
        else:
            labeled_count = max(1, int(len(indices) * label_ratio))
            labeled_count = min(labeled_count, len(indices))
        labeled_by_class.append(indices[:labeled_count])
        unlabeled_by_class.append(indices[labeled_count:])

    return labeled_by_class, unlabeled_by_class


def _merge_client_indices(client_labeled, client_unlabeled):
    """合并两路客户端索引，并生成对应的标签掩码。"""
    client_train = []
    client_is_labeled = []
    for labeled, unlabeled in zip(client_labeled, client_unlabeled):
        labeled = np.asarray(labeled, dtype=int)
        unlabeled = np.asarray(unlabeled, dtype=int)
        client_train.append(np.concatenate((labeled, unlabeled)))
        client_is_labeled.append(
            torch.cat(
                (
                    torch.ones(len(labeled), dtype=torch.bool),
                    torch.zeros(len(unlabeled), dtype=torch.bool),
                )
            )
        )
    return client_train, client_is_labeled


def prepare_double_ssl_data(args, dataset_name, raw_data):
    """准备双异质半监督数据并保存为统一的客户端 ``.pt`` 文件。"""
    if not (0 < args.label_ratio < 1):
        raise ValueError("混合 SSL (double) 要求 label_ratio 必须位于 (0, 1) 开区间")

    X, Y = raw_data["x"], raw_data["y"]
    rng = np.random.default_rng(args.seed)

    train_by_class, test_by_class, num_classes = split_indices_by_class(
        Y, args.test_ratio, rng
    )
    output_dir = get_output_dir(args, dataset_name)
    if not is_fresh(output_dir, args.num_clients) and _has_mixed_ssl_clients(
        output_dir, args.num_clients
    ):
        print(f"-> Double SSL partition already exists at {output_dir}. Skipping.")
        return output_dir
    print(f"-> Partitioning double SSL data ({os.path.basename(output_dir)})...")

    labeled_by_class, unlabeled_by_class = split_labeled_unlabeled_by_class(
        train_by_class, args.label_ratio, rng
    )

    # 两路训练数据使用独立的分配过程，但共享当前项目的统一划分实现。
    labeled_clients = distribute_by_class(
        labeled_by_class,
        args.num_clients,
        args.partition,
        rng,
        args.alpha,
        args.n_class,
    )
    unlabeled_clients = distribute_by_class(
        unlabeled_by_class,
        args.num_clients,
        args.partition,
        rng,
        args.alpha,
        args.n_class,
    )
    test_clients = distribute_by_class(
        test_by_class,
        args.num_clients,
        args.partition,
        rng,
        args.alpha,
        args.n_class,
    )

    client_labeled = [
        np.concatenate(parts) if parts else np.array([], dtype=int)
        for parts in labeled_clients
    ]
    client_unlabeled = [
        np.concatenate(parts) if parts else np.array([], dtype=int)
        for parts in unlabeled_clients
    ]
    client_labeled = _ensure_nonempty_client_indices(
        client_labeled, rng, "double 有标签 train"
    )
    client_unlabeled = _ensure_nonempty_client_indices(
        client_unlabeled, rng, "double 无标签 train"
    )
    client_train, client_is_labeled = _merge_client_indices(
        client_labeled, client_unlabeled
    )
    client_test = [
        np.concatenate(parts) if parts else np.array([], dtype=int)
        for parts in test_clients
    ]
    client_test = _ensure_nonempty_client_indices(client_test, rng, "test")

    save_client_data(
        output_dir,
        X,
        Y,
        client_train,
        client_test,
        num_classes,
        extra_fields={"train": {"is_labeled": client_is_labeled}},
    )

    print(
        f"-> 成功为 {args.num_clients} 个客户端准备了 "
        f"{dataset_name} (double SSL, {args.partition})。"
    )
    return output_dir
