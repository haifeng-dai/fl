import os

import numpy as np

from .common import get_output_dir, save_client_data


# ──────────────────────────────────────────────────────────────────────────
# 类别划分数学（iid / dirichlet / pathological）—— 与数据格式无关，可复用
# ──────────────────────────────────────────────────────────────────────────
def split_indices_by_class(targets, test_ratio):
    targets_np = targets.numpy()
    num_classes = len(np.unique(targets_np))
    indices_by_class = [np.where(targets_np == i)[0] for i in range(num_classes)]

    train_indices_by_class = []
    test_indices_by_class = []

    for c_idx in indices_by_class:
        np.random.shuffle(c_idx)
        split = int(len(c_idx) * (1 - test_ratio))
        train_indices_by_class.append(c_idx[:split])
        test_indices_by_class.append(c_idx[split:])

    return train_indices_by_class, test_indices_by_class, num_classes


def iid_partition(train_indices_by_class, test_indices_by_class, num_clients):
    client_train_indices = [[] for _ in range(num_clients)]
    client_test_indices = [[] for _ in range(num_clients)]

    for k in range(len(train_indices_by_class)):
        tr_k = train_indices_by_class[k]
        tr_splits = np.array_split(tr_k, num_clients)
        te_k = test_indices_by_class[k]
        te_splits = np.array_split(te_k, num_clients)

        for i in range(num_clients):
            client_train_indices[i].append(tr_splits[i])
            client_test_indices[i].append(te_splits[i])

    return (
        [np.concatenate(idx) for idx in client_train_indices],
        [np.concatenate(idx) for idx in client_test_indices],
    )


def dirichlet_partition(
    train_indices_by_class, test_indices_by_class, num_clients, alpha=0.5
):
    client_train_indices = [[] for _ in range(num_clients)]
    client_test_indices = [[] for _ in range(num_clients)]
    num_classes = len(train_indices_by_class)

    for k in range(num_classes):
        proportions = np.random.dirichlet([alpha] * num_clients)

        tr_k = train_indices_by_class[k]
        tr_counts = (np.cumsum(proportions) * len(tr_k)).astype(int)[:-1]
        tr_splits = np.split(tr_k, tr_counts)

        te_k = test_indices_by_class[k]
        te_counts = (np.cumsum(proportions) * len(te_k)).astype(int)[:-1]
        te_splits = np.split(te_k, te_counts)

        for i in range(num_clients):
            client_train_indices[i].append(tr_splits[i])
            client_test_indices[i].append(te_splits[i])

    return (
        [np.concatenate(idx) for idx in client_train_indices],
        [np.concatenate(idx) for idx in client_test_indices],
    )


def pathological_partition(
    train_indices_by_class, test_indices_by_class, num_clients, n_classes_per_client=2
):
    num_classes = len(train_indices_by_class)
    client_train_indices = [[] for _ in range(num_clients)]
    client_test_indices = [[] for _ in range(num_clients)]

    total_slots = num_clients * n_classes_per_client

    if total_slots < num_classes:
        raise ValueError(
            f"[Pathological Partition Error] 总需求分片数 ({total_slots}) 小于类别总数 ({num_classes})。"
        )

    shards_per_class_list = [total_slots // num_classes] * num_classes
    remainder = total_slots % num_classes
    for i in range(remainder):
        shards_per_class_list[i] += 1

    train_shards = []
    test_shards = []
    for k in range(num_classes):
        shards_for_this_class = shards_per_class_list[k]
        if shards_for_this_class == 0:
            train_shards.append([])
            test_shards.append([])
            continue
        if len(train_indices_by_class[k]) < shards_for_this_class:
            raise ValueError(
                f"[Pathological Partition Error] 类别 {k} 样本量不足以切分为 {shards_for_this_class} 个分片。"
            )
        train_shards.append(
            np.array_split(train_indices_by_class[k], shards_for_this_class)
        )
        test_shards.append(
            np.array_split(test_indices_by_class[k], shards_for_this_class)
        )

    shard_ids = []
    for k in range(num_classes):
        for s in range(shards_per_class_list[k]):
            shard_ids.append((k, s))
    np.random.shuffle(shard_ids)

    for i in range(num_clients):
        for j in range(n_classes_per_client):
            k, s = shard_ids[i * n_classes_per_client + j]
            client_train_indices[i].append(train_shards[k][s])
            client_test_indices[i].append(test_shards[k][s])

    return (
        [np.concatenate(idx) for idx in client_train_indices],
        [np.concatenate(idx) for idx in client_test_indices],
    )


def prepare_label_data(args, dataset_name, raw_data):
    """类别划分（iid / dirichlet / pathological），承载数据分布异质。

    仅做类别划分并落盘；ssl 掩码（sample/client）由调用方（__init__.py）在划分后
    独立应用，本函数不感知 ssl 语义。DG / SFD 不经过此函数。
    """
    partition_method = args.partition
    num_clients = args.num_clients

    X, Y = raw_data["x"], raw_data["y"]

    tr_idx_by_cls, te_idx_by_cls, num_classes = split_indices_by_class(
        Y, args.test_ratio
    )

    if args.n_class == 0:
        args.n_class = max(2, -(-num_classes // num_clients))
        print(
            f"-> Adaptive n_class: dataset has {num_classes} classes, "
            f"{num_clients} clients, setting n_class={args.n_class}"
        )

    output_dir = get_output_dir(args, dataset_name)

    print(f"-> Partitioning data ({os.path.basename(output_dir)})...")

    if partition_method == "iid":
        cli_tr_idx, cli_te_idx = iid_partition(
            tr_idx_by_cls, te_idx_by_cls, num_clients
        )
    elif partition_method == "dirichlet":
        cli_tr_idx, cli_te_idx = dirichlet_partition(
            tr_idx_by_cls, te_idx_by_cls, num_clients, args.alpha
        )
    elif partition_method == "pathological":
        cli_tr_idx, cli_te_idx = pathological_partition(
            tr_idx_by_cls, te_idx_by_cls, num_clients, args.n_class
        )
    else:
        raise ValueError(f"未知分区方法: {partition_method}")

    # 类别划分：无 domains 字段；is_labeled 默认全有标签（ssl 掩码时再写入）
    save_client_data(output_dir, X, Y, cli_tr_idx, cli_te_idx, num_classes)

    print(
        f"-> 成功为 {num_clients} 个客户端准备了 {dataset_name} ({partition_method})。"
    )
    return output_dir
