import os

import numpy as np
import torch

from .common import (
    distribute_by_class,
    ensure_nonempty_client_indices,
    get_output_dir,
    resolve_n_class,
    save_client_data,
)


# ──────────────────────────────────────────────────────────────────────────
# client 半监督模式的客户端标签身份（纯函数，不读写文件）
# ──────────────────────────────────────────────────────────────────────────
def build_client_label_masks(train_indices, label_ratio, seed):
    """为 client 半监督模式生成每客户端的 is_labeled 掩码（纯函数，不读写文件）。

    label_ratio 此处表示【有标签客户端比例】（客户端级），与 mixed.py 中样本级
    的 label_ratio 含义不同。返回长度等于 num_clients 的列表，每个元素是该客户端
    训练样本级别的 bool 向量（全 True 表示该客户端全为标签数据，全 False 表示全无标签）。

    client 模式要求同时存在有标签与无标签客户端，因此 label_ratio 必须位于开区间
    (0, 1)，且 num_clients >= 2。
    """
    num_clients = len(train_indices)
    if num_clients < 2:
        raise ValueError(
            f"client 半监督模式要求 num_clients >= 2，当前为 {num_clients}。"
        )
    if not 0 < label_ratio < 1:
        raise ValueError(
            f"client 半监督模式要求 label_ratio 位于 (0, 1)，当前为 {label_ratio}。"
        )
    rng_client = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(num_clients, generator=rng_client)
    num_labeled_clients = min(
        num_clients - 1,
        max(1, int(num_clients * label_ratio)),
    )
    labeled_clients = set(permutation[:num_labeled_clients].tolist())
    return [
        torch.full(
            (len(train_indices[client_id]),),
            client_id in labeled_clients,
            dtype=torch.bool,
        )
        for client_id in range(num_clients)
    ]


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
    共用同一份实现）。`ssl=client` 模式下，有标签客户端掩码在保存阶段一次性写入，
    不再二次读写缓存；其余模式不写 is_labeled。DG / SFD 不经过此函数。
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
    cli_tr_idx = ensure_nonempty_client_indices(cli_tr_idx, rng, "train")
    cli_te_idx = ensure_nonempty_client_indices(cli_te_idx, rng, "test")

    # client 半监督模式：在保存阶段一次性写入每客户端 is_labeled 掩码，避免二次 IO。
    extra_fields = None
    if args.ssl == "client":
        train_is_labeled = build_client_label_masks(
            cli_tr_idx, args.label_ratio, args.seed
        )
        extra_fields = {"train": {"is_labeled": train_is_labeled}}

    # 类别划分：无 domains 字段；非 client 模式不写 is_labeled
    save_client_data(
        output_dir, X, Y, cli_tr_idx, cli_te_idx, num_classes, extra_fields
    )

    print(
        f"-> 成功为 {num_clients} 个客户端准备了 {dataset_name} ({partition_method})。"
    )
    return output_dir
