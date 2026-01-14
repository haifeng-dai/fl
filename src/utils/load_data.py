import torch
import os
from torch.utils.data import DataLoader, TensorDataset

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

def load_test_data(dataset_name):
    test_path = os.path.join("./datasets", dataset_name, "test_data.pt")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Global test data not found at {test_path}.")
    data = torch.load(test_path, weights_only=False)
    dataset = TensorDataset(data["x"], data["y"])
    return DataLoader(dataset, batch_size=128, shuffle=False)

def load_data(dataset_name, partition, num_clients, alpha=0.5, n_classes=2, pfl=False):
    """
    统一的数据加载接口。
    返回: (train_loaders, test_loader)
    - train_loaders: dict[int, DataLoader]
    - test_loader: dict[int, DataLoader] (如果 pfl=True) 或 DataLoader (如果 pfl=False)
    """
    part_dir = get_partition_path(dataset_name, partition, num_clients, alpha, n_classes)
    
    train_loaders = {}
    test_loaders = {} if pfl else None

    for i in range(num_clients):
        data_path = os.path.join(part_dir, f"client_{i}.pt")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data for client {i} not found at {data_path}.")
        
        data = torch.load(data_path, weights_only=False)
        
        # 加载训练集
        train_dataset = TensorDataset(data["train"]["x"], data["train"]["y"])
        train_loaders[i] = DataLoader(train_dataset, batch_size=64, shuffle=True)
        
        # 如果是 pFL，加载每个客户端的本地测试集
        if pfl:
            test_dataset = TensorDataset(data["test"]["x"], data["test"]["y"])
            test_loaders[i] = DataLoader(test_dataset, batch_size=64, shuffle=False)

    if not pfl:
        # 非 pFL 模式，加载全局测试集
        test_loader = load_test_data(dataset_name)
    else:
        test_loader = test_loaders

    return train_loaders, test_loader