import os

import torch

from src.data_gen import get_domain_partition_dir


class MetaDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, domains=None, is_labeled=None):
        self.x = x
        self.y = y
        self.domains = domains
        self.is_labeled = is_labeled

        if domains is not None:
            uniq = sorted(set(domains))
            self._domain_ids = torch.tensor(
                [{d: i for i, d in enumerate(uniq)}[d] for d in domains],
                dtype=torch.long,
            )
            self._domain_map = {d: i for i, d in enumerate(uniq)}
        else:
            self._domain_ids = None
            self._domain_map = None

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (
            self.x[idx],
            self.y[idx],
            self._domain_ids[idx]
            if self._domain_ids is not None
            else torch.tensor(-1, dtype=torch.long),
            self.is_labeled[idx] if self.is_labeled is not None else torch.tensor(True),
        )


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


def load_data(
    dataset_name,
    partition=None,
    num_clients=20,
    alpha=0.5,
    n_classes=2,
    pfl=False,
    domain_partition=None,
    domain_aware=True,
):
    is_domain = domain_partition is not None

    if is_domain:
        part_str = get_domain_partition_dir(
            domain_partition, num_clients, alpha, domain_aware=domain_aware
        )
        part_dir = os.path.join("./datasets", dataset_name, part_str)
    else:
        part_dir = get_partition_path(
            dataset_name, partition, num_clients, alpha, n_classes
        )

    train_datasets = {}
    test_datasets = {}
    train_counts = {}
    domain_labels = {}
    data = {}

    for i in range(num_clients):
        data_path = os.path.join(part_dir, f"client_{i}.pt")
        data = torch.load(data_path, weights_only=False)

        train_x = data["train"]["x"]
        train_y = data["train"]["y"]
        train_domains = data["train"].get("domains", None) if is_domain else None
        train_datasets[i] = MetaDataset(train_x, train_y, domains=train_domains)
        train_counts[i] = len(train_x)

        test_x = data["test"]["x"]
        test_y = data["test"]["y"]
        test_domains = data["test"].get("domains", None) if is_domain else None
        test_datasets[i] = MetaDataset(test_x, test_y, domains=test_domains)

        if is_domain:
            domain_labels[i] = train_domains if train_domains is not None else []

    num_class = data["num_classes"]

    if not pfl:
        all_test_x, all_test_y = [], []
        for ds in test_datasets.values():
            all_test_x.append(ds.x)
            all_test_y.append(ds.y)
        test_datasets = MetaDataset(torch.cat(all_test_x), torch.cat(all_test_y))

    source_test = None
    target_test = None

    if is_domain:
        source_test_path = os.path.join(part_dir, "source_test.pt")
        target_test_path = os.path.join(part_dir, "target_test.pt")

        if os.path.exists(source_test_path):
            st = torch.load(source_test_path, weights_only=False)
            source_test = MetaDataset(st["x"], st["y"])

        if os.path.exists(target_test_path):
            tt = torch.load(target_test_path, weights_only=False)
            target_test = MetaDataset(tt["x"], tt["y"])

    return (
        train_datasets,
        test_datasets,
        train_counts,
        num_class,
        source_test,
        target_test,
        domain_labels,
    )
