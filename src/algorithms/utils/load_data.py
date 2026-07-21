import os

import torch
from torch.utils.data import TensorDataset


def get_partition_path(dataset_name, partition, num_clients, alpha=0.5, n_classes=2):
    if partition == "iid":
        part_str = f"iid_n{num_clients}"
    elif partition == "dirichlet":
        part_str = f"dirichlet_n{num_clients}_a{alpha}"
    elif partition == "pathological":
        part_str = f"pathological_n{num_clients}_c{n_classes}"
    else:
        raise ValueError(f"Unknown partition method: {partition}")
    return os.path.join("./datasets", dataset_name, part_str)


def load_domain_data(
    domain_dataset, domain_partition, num_clients, alpha=0.5, pfl=False
):
    """
    领域感知的数据加载接口。
    返回: (train_datasets, test_dataset, train_counts, num_class, source_test, target_test, domain_labels)
    - train_datasets: dict[int, Dataset]
    - test_dataset: dict[int, Dataset] (如果 pfl=True) 或 Dataset (如果 pfl=False)
    - train_counts: dict[int, int]
    - num_class: int
    - source_test: Dataset (所有源域测试样本合并)
    - target_test: Dataset | None (目标域测试样本，None 表示未设置 target_domain)
    - domain_labels: dict[int, list[str]] (每个客户端的领域分布)
    """
    part_dir = os.path.join("./datasets", domain_dataset, domain_partition)

    train_datasets = {}
    test_datasets = {}
    train_counts = {}
    domain_labels = {}
    data: dict = {}

    for i in range(num_clients):
        data_path = os.path.join(part_dir, f"client_{i}.pt")
        data = torch.load(data_path, weights_only=False)

        train_x = data["train"]["x"]
        train_y = data["train"]["y"]
        train_datasets[i] = TensorDataset(train_x, train_y)
        train_counts[i] = len(train_x)

        test_x = data["test"]["x"]
        test_y = data["test"]["y"]
        test_datasets[i] = TensorDataset(test_x, test_y)

        # 提取该客户端的领域标签
        train_domains = data["train"].get("domains", [])
        domain_labels[i] = train_domains

    num_class = data["num_classes"]

    if not pfl:
        all_test_x, all_test_y = [], []
        for ds in test_datasets.values():
            all_test_x.append(ds.tensors[0])
            all_test_y.append(ds.tensors[1])
        test_datasets = TensorDataset(torch.cat(all_test_x), torch.cat(all_test_y))

    # 加载源域测试集和目标域测试集
    source_test_path = os.path.join(part_dir, "source_test.pt")
    target_test_path = os.path.join(part_dir, "target_test.pt")

    source_test = None
    if os.path.exists(source_test_path):
        st = torch.load(source_test_path, weights_only=False)
        source_test = TensorDataset(st["x"], st["y"])

    target_test = None
    if os.path.exists(target_test_path):
        tt = torch.load(target_test_path, weights_only=False)
        target_test = TensorDataset(tt["x"], tt["y"])

    return (
        train_datasets,
        test_datasets,
        train_counts,
        num_class,
        source_test,
        target_test,
        domain_labels,
    )


def load_data(dataset_name, partition, num_clients, alpha=0.5, n_classes=2, pfl=False):
    """
    统一的数据加载接口。
    返回: (train_datasets, test_dataset, train_counts)
    - train_datasets: dict[int, Dataset]
    - test_dataset: dict[int, Dataset] (如果 pfl=True) 或 Dataset (如果 pfl=False)
    - train_counts: dict[int, int]
    """
    part_dir = get_partition_path(
        dataset_name, partition, num_clients, alpha, n_classes
    )

    train_datasets = {}
    test_datasets = {}
    train_counts = {}

    data = {}
    for i in range(num_clients):
        data_path = os.path.join(part_dir, f"client_{i}.pt")
        data = torch.load(data_path, weights_only=False)

        # 加载训练集并消除多进程 IPC 序列化开销（Ray 独立处理，不使用 PyTorch 共享内存）
        train_x = data["train"]["x"]
        train_y = data["train"]["y"]
        train_datasets[i] = TensorDataset(train_x, train_y)
        train_counts[i] = len(train_x)

        # 加载测试集
        test_x = data["test"]["x"]
        test_y = data["test"]["y"]
        test_datasets[i] = TensorDataset(test_x, test_y)

    num_class = data["num_classes"]

    if not pfl:
        # 非 pFL 模式，聚合所有客户端的测试集作为全局测试集
        all_test_x = []
        all_test_y = []
        for ds in test_datasets.values():
            # TensorDataset.tensors 返回 (x, y) 元组
            all_test_x.append(ds.tensors[0])
            all_test_y.append(ds.tensors[1])

        merged_x = torch.cat(all_test_x, dim=0)
        merged_y = torch.cat(all_test_y, dim=0)

        test_datasets = TensorDataset(merged_x, merged_y)

    return train_datasets, test_datasets, train_counts, num_class
