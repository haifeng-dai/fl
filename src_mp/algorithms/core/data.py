from torch.utils.data import DataLoader, TensorDataset

from src.algorithms.utils.input import normalize_dataset
from src.algorithms.utils.load_data import load_data
from src.data_gen import prepare_data


class SSLStream:
    """包装有标签与无标签 DataLoader。

    迭代时以 unlabeled_loader 为基准，labeled_loader 自动无限循环补齐。
    同时提供 labeled_loader 和 unlabeled_loader 属性以备特殊用途。
    """

    def __init__(self, labeled_loader: DataLoader, unlabeled_loader: DataLoader):
        self.labeled_loader = labeled_loader
        self.unlabeled_loader = unlabeled_loader

    def __iter__(self):
        def _infinite(loader):
            while True:
                yield from loader

        return zip(_infinite(self.labeled_loader), self.unlabeled_loader)

    def __len__(self) -> int:
        return len(self.unlabeled_loader)


class SSLDataLoaderRegistry:
    """管理并缓存所有客户端的半监督双流加载器 (SSLStream)。"""

    def __init__(self, train_sets: dict, batch_size: int, unlabeled_ratio: int = 1):
        self.streams: dict[int, SSLStream] = {}
        for cid, dataset in train_sets.items():
            labeled_mask = dataset.is_labeled.bool()
            l_loader = DataLoader(
                TensorDataset(dataset.x[labeled_mask], dataset.y[labeled_mask]),
                batch_size=batch_size,
                shuffle=True,
            )
            u_loader = DataLoader(
                TensorDataset(dataset.x[~labeled_mask], dataset.y[~labeled_mask]),
                batch_size=batch_size * unlabeled_ratio,
                shuffle=True,
            )
            self.streams[cid] = SSLStream(l_loader, u_loader)

    def get(self, cid: int) -> SSLStream:
        return self.streams[cid]


def load_federated_data(args, pfl: bool = False, normalize: bool = True):
    prepare_data(args)
    train_sets, test_set, train_counts, num_class = load_data(args, pfl=pfl)
    if normalize:
        for dataset in train_sets.values():
            normalize_dataset(dataset, args.dataset)
        if pfl:
            for dataset in test_set.values():
                normalize_dataset(dataset, args.dataset)
        else:
            normalize_dataset(test_set, args.dataset)
    return train_sets, test_set, train_counts, num_class
