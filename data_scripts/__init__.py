import importlib
import os

import numpy as np
import torch

# --- 辅助方法 ---

def split_indices_by_class(targets, test_ratio):
    """
    先按类别对所有索引进行划分，每个类别内部按 test_ratio 分成训练和测试。
    """
    num_classes = len(np.unique(targets))
    indices_by_class = [np.where(targets == i)[0] for i in range(num_classes)]

    train_indices_by_class = []
    test_indices_by_class = []

    for c_idx in indices_by_class:
        np.random.shuffle(c_idx)
        split = int(len(c_idx) * (1 - test_ratio))
        train_indices_by_class.append(c_idx[:split])
        test_indices_by_class.append(c_idx[split:])

    return train_indices_by_class, test_indices_by_class, num_classes

# --- 分区方法 ---

def iid_partition(train_indices_by_class, test_indices_by_class, num_clients):
    client_train_indices = [[] for _ in range(num_clients)]
    client_test_indices = [[] for _ in range(num_clients)]

    for k in range(len(train_indices_by_class)):
        # Train
        tr_k = train_indices_by_class[k]
        tr_splits = np.array_split(tr_k, num_clients)
        # Test
        te_k = test_indices_by_class[k]
        te_splits = np.array_split(te_k, num_clients)

        for i in range(num_clients):
            client_train_indices[i].append(tr_splits[i])
            client_test_indices[i].append(te_splits[i])

    return ([np.concatenate(idx) for idx in client_train_indices],
            [np.concatenate(idx) for idx in client_test_indices])


def dirichlet_partition(train_indices_by_class, test_indices_by_class, num_clients, alpha=0.5):
    client_train_indices = [[] for _ in range(num_clients)]
    client_test_indices = [[] for _ in range(num_clients)]
    num_classes = len(train_indices_by_class)

    for k in range(num_classes):
        proportions = np.random.dirichlet([alpha] * num_clients)

        # 划分训练集
        tr_k = train_indices_by_class[k]
        tr_counts = (np.cumsum(proportions) * len(tr_k)).astype(int)[:-1]
        tr_splits = np.split(tr_k, tr_counts)

        # 划分测试集（使用相同的比例）
        te_k = test_indices_by_class[k]
        te_counts = (np.cumsum(proportions) * len(te_k)).astype(int)[:-1]
        te_splits = np.split(te_k, te_counts)

        for i in range(num_clients):
            client_train_indices[i].append(tr_splits[i])
            client_test_indices[i].append(te_splits[i])

    return ([np.concatenate(idx) for idx in client_train_indices],
            [np.concatenate(idx) for idx in client_test_indices])


def pathological_partition(train_indices_by_class, test_indices_by_class, num_clients, n_classes_per_client=2):
    num_classes = len(train_indices_by_class)
    client_train_indices = [[] for _ in range(num_clients)]
    client_test_indices = [[] for _ in range(num_clients)]

    total_slots = num_clients * n_classes_per_client

    if total_slots < num_classes:
        raise ValueError(
            f"[Pathological Partition Error] 总需求分片数 ({total_slots}) 小于类别总数 ({num_classes})。\n"
            f"请增加 num_clients 或 n_classes_per_client。"
        )

    if total_slots % num_classes != 0:
        raise ValueError(
            f"[Pathological Partition Error] 总需求分片数 ({total_slots}) 无法被类别总数 ({num_classes}) 整除。\n"
            f"请调整参数使得 (num_clients * n_classes_per_client) % {num_classes} == 0。"
        )

    shards_per_class = total_slots // num_classes
    if shards_per_class == 0:
        raise ValueError("[Pathological Partition Error] 计算出的每类分片数为 0。")

    train_shards = []
    test_shards = []
    for k in range(num_classes):
        if len(train_indices_by_class[k]) < shards_per_class:
            raise ValueError(
                f"[Pathological Partition Error] 类别 {k} 的样本量 ({len(train_indices_by_class[k])}) "
                f"不足以切分为 {shards_per_class} 个分片。"
            )

        train_shards.append(np.array_split(train_indices_by_class[k], shards_per_class))
        test_shards.append(np.array_split(test_indices_by_class[k], shards_per_class))

    shard_ids = []
    for k in range(num_classes):
        for s in range(shards_per_class):
            shard_ids.append((k, s))

    np.random.shuffle(shard_ids)

    for i in range(num_clients):
        for j in range(n_classes_per_client):
            k, s = shard_ids[i * n_classes_per_client + j]
            client_train_indices[i].append(train_shards[k][s])
            client_test_indices[i].append(test_shards[k][s])

    return ([np.concatenate(idx) for idx in client_train_indices],
            [np.concatenate(idx) for idx in client_test_indices])


# --- 主要入口点 ---


def prepare_data(dataset_name, partition_method, num_clients, **kwargs):
    raw_dir = "./datasets/raw"
    raw_path = os.path.join(raw_dir, f"{dataset_name}_raw.pt")

    if not os.path.exists(raw_path):
        print(f"-> 未找到 {dataset_name} 的原始数据。正在处理...")
        module = importlib.import_module(f"data_scripts.process_{dataset_name}")
        module.process(raw_dir)

    # 2. 准备分区文件夹名
    if partition_method == "iid":
        part_str = f"iid_n{num_clients}"
    elif partition_method == "dirichlet":
        alpha = kwargs.get("alpha", 0.5)
        part_str = f"dirichlet_n{num_clients}_a{alpha}"
    elif partition_method == "pathological":
        n_classes = kwargs.get("n_classes", 2)
        part_str = f"pathological_n{num_clients}_c{n_classes}"
    else:
        raise ValueError(f"未知分区方法: {partition_method}")

    output_dir = os.path.join("./datasets", dataset_name, part_str)

    if os.path.exists(output_dir) and len(os.listdir(output_dir)) >= num_clients:
        print(f"-> {dataset_name} 的 {part_str} 分区已存在。跳过处理。")

    print(f"-> 正在划分数据 ({part_str})...")
    data = torch.load(raw_path, weights_only=False)
    X, Y = data["x"], data["y"]

    test_ratio = kwargs.get("test_ratio", 0.2)

    # 1. 首先按类别划分训练和测试索引
    tr_idx_by_cls, te_idx_by_cls, num_classes = split_indices_by_class(Y.numpy(), test_ratio)

    # 保存一个全局测试集供服务器使用 (包含所有类的测试部分)
    base_dir = f"./datasets/{dataset_name}"
    if not os.path.exists(os.path.join(base_dir, "test_data.pt")):
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        all_te_idx = np.concatenate(te_idx_by_cls)
        torch.save({"x": X[all_te_idx], "y": Y[all_te_idx]}, os.path.join(base_dir, "test_data.pt"))

    # 2. 执行分区逻辑
    if partition_method == "iid":
        cli_tr_idx, cli_te_idx = iid_partition(tr_idx_by_cls, te_idx_by_cls, num_clients)
    elif partition_method == "dirichlet":
        alpha = kwargs.get("alpha", 0.5)
        cli_tr_idx, cli_te_idx = dirichlet_partition(tr_idx_by_cls, te_idx_by_cls, num_clients, alpha)
    elif partition_method == "pathological":
        n_classes = kwargs.get("n_classes", 2)
        cli_tr_idx, cli_te_idx = pathological_partition(tr_idx_by_cls, te_idx_by_cls, num_clients, n_classes)
    else:
        raise ValueError(f"未知分区方法: {partition_method}")

    # 3. 保存客户端数据
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(num_clients):
        client_data = {
            "train": {"x": X[cli_tr_idx[i]], "y": Y[cli_tr_idx[i]]},
            "test": {"x": X[cli_te_idx[i]], "y": Y[cli_te_idx[i]]},
            "num_classes": num_classes
        }
        torch.save(client_data, os.path.join(output_dir, f"client_{i}.pt"))

    print(f"-> 成功为 {num_clients} 个客户端准备了 {dataset_name} ({partition_method})。")
