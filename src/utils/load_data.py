import torch
import os
from torch.utils.data import TensorDataset, ConcatDataset


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

    for i in range(num_clients):
        data_path = os.path.join(part_dir, f"client_{i}.pt")
        data = torch.load(data_path, weights_only=False)

        # 加载训练集
        train_datasets[i] = TensorDataset(data["train"]["x"], data["train"]["y"])
        train_counts[i] = len(data["train"]["x"])

        test_datasets[i] = TensorDataset(data["test"]["x"], data["test"]["y"])

    num_class = data["num_classes"]  # type: ignore

    if not pfl:
        # 非 pFL 模式，聚合所有客户端的测试集作为全局测试集
        test_datasets = ConcatDataset(list(test_datasets.values()))

    return train_datasets, test_datasets, train_counts, num_class
