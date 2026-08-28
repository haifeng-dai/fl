import os

import numpy as np

from .common import (
    distribute_by_class,
    _ensure_nonempty_client_indices,
    get_output_dir,
    resolve_n_class,
    save_client_data,
)


# ──────────────────────────────────────────────────────────────────────────
# 类别划分数学（iid / dirichlet / pathological）—— 与数据格式无关，可复用
# ──────────────────────────────────────────────────────────────────────────
def split_indices_by_class(targets, test_ratio, rng):
    targets_np = targets.numpy()
    num_classes = len(np.unique(targets_np))
    indices_by_class = [np.where(targets_np == i)[0] for i in range(num_classes)]

    train_indices_by_class = []
    test_indices_by_class = []

    for c_idx in indices_by_class:
        rng.shuffle(c_idx)
        split = int(len(c_idx) * (1 - test_ratio))
        train_indices_by_class.append(c_idx[:split])
        test_indices_by_class.append(c_idx[split:])

    return train_indices_by_class, test_indices_by_class, num_classes


def prepare_label_data(args, dataset_name, raw_data):
    """类别划分（iid / dirichlet / pathological），承载数据分布异质。

    按类分组后委托 distribute_by_class 完成异质分布（与域内核质 hetero_split
    共用同一份实现）；ssl 掩码由调用方在划分后独立应用，本函数不感知 ssl 语义。
    DG / SFD 不经过此函数。
    """
    partition_method = args.partition
    num_clients = args.num_clients

    X, Y = raw_data["x"], raw_data["y"]
    rng = np.random.default_rng(args.seed)

    tr_idx_by_cls, te_idx_by_cls, num_classes = split_indices_by_class(
        Y, args.test_ratio, rng
    )

    if args.n_class == 0:
        args.n_class = resolve_n_class(args, num_classes)
        print(
            f"-> Adaptive n_class: dataset has {num_classes} classes, "
            f"{num_clients} clients, setting n_class={args.n_class}"
        )

    output_dir = get_output_dir(args, dataset_name)
    print(f"-> Partitioning data ({os.path.basename(output_dir)})...")

    # 训练 / 测试各自按类分组后，共用同一份异质分布实现
    train_client = distribute_by_class(
        tr_idx_by_cls, num_clients, partition_method, rng, args.alpha, args.n_class
    )
    test_client = distribute_by_class(
        te_idx_by_cls, num_clients, partition_method, rng, args.alpha, args.n_class
    )
    cli_tr_idx = [
        np.concatenate(c) if len(c) else np.array([], dtype=int) for c in train_client
    ]
    cli_te_idx = [
        np.concatenate(c) if len(c) else np.array([], dtype=int) for c in test_client
    ]
    cli_tr_idx = _ensure_nonempty_client_indices(cli_tr_idx, rng, "train")
    cli_te_idx = _ensure_nonempty_client_indices(cli_te_idx, rng, "test")

    # 类别划分：无 domains 字段；is_labeled 默认全有标签（ssl 掩码时再写入）
    save_client_data(output_dir, X, Y, cli_tr_idx, cli_te_idx, num_classes)

    print(
        f"-> 成功为 {num_clients} 个客户端准备了 {dataset_name} ({partition_method})。"
    )
    return output_dir
