import os

import torch

from src.data_gen.partition.common import get_output_dir


class MetaDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, domains=None, is_labeled=None):
        self.x = x
        self.y = y
        self.domains = domains
        self.is_labeled = is_labeled

        if domains is not None:
            # domains = ["sketch", "photo", "sketch", "photo", "cartoon"]
            uniq = sorted(set(domains))
            # uniq = ["cartoon", "photo", "sketch"]
            self.domain_map = {d: i for i, d in enumerate(uniq)}
            # domain_map = {"cartoon": 0, "photo": 1, "sketch": 2}
            #             ↑ 字符串域 → 整数 ID 的查表
            self.domain_ids = torch.tensor(
                [self.domain_map[d] for d in domains],
                dtype=torch.long,
            )
            # domain_ids = tensor([2, 1, 2, 1, 0])
            #             ↑ 每个样本对应的整数域 ID，可被 DataLoader batch、做 ==/!= 比较
        else:
            self.domain_ids = None
            self.domain_map = None

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        domain_id = (
            self.domain_ids[idx]
            if self.domain_ids is not None
            else torch.tensor(-1, dtype=torch.long)  # 非 DG 时填充 -1
        )  # int
        labeled = (
            self.is_labeled[idx]
            if self.is_labeled is not None
            else torch.tensor(True)  # 默认全有标签
        )  # bool
        return (
            self.x[idx],
            self.y[idx],
            domain_id,
            labeled,
        )  # 样本, 标签, 域 ID, 是否有标签


def load_data(args, pfl=False):
    """加载已划分好的客户端数据。

    - 划分目录由 get_output_dir(args, dataset_name) 唯一确定（sfd/fdg/category/mixed 自动区分）。
    - 所有模式均走同一通用加载路径：每个 client_i.pt 含 train 与 test；
      train 的 is_labeled 透传（sample/double 掩码场景存在该字段，其余缺省全有标签）。
    - ``pfl`` 只决定返回的测试集形态：``pfl=False`` 合并各客户端 test 为单个全局
      MetaDataset；``pfl=True`` 保留每客户端 test 字典。``pfl`` 不参与 SSL 场景判断，
      算法与数据场景是否适配由调用方保证。
    """
    dataset_name = args.dataset
    part_dir = get_output_dir(args, dataset_name)

    train_datasets = {}
    test_datasets = {}
    train_counts = {}
    data = {}

    for i in range(args.num_clients):
        data_path = os.path.join(part_dir, f"client_{i}.pt")
        data = torch.load(data_path, weights_only=False)

        train_x = data["train"]["x"]
        train_y = data["train"]["y"]
        train_is_labeled = data["train"].get("is_labeled", None)
        train_datasets[i] = MetaDataset(train_x, train_y, is_labeled=train_is_labeled)
        train_counts[i] = len(train_x)

        test_x = data["test"]["x"]
        test_y = data["test"]["y"]
        test_domains = data["test"].get("domains", None)
        test_is_labeled = data["test"].get("is_labeled", None)
        test_datasets[i] = MetaDataset(
            test_x, test_y, domains=test_domains, is_labeled=test_is_labeled
        )

    num_class = data["num_classes"]

    if not pfl:
        # 合并各客户端测试集为全局测试集，保留 domains/is_labeled（评估按域切分用）
        all_test_x, all_test_y, all_test_domains, all_test_labeled = [], [], [], []
        for ds in test_datasets.values():
            all_test_x.append(ds.x)
            all_test_y.append(ds.y)
            if ds.domains is not None:
                all_test_domains.extend(ds.domains)
            if ds.is_labeled is not None:
                all_test_labeled.append(ds.is_labeled)
        test_datasets = MetaDataset(
            torch.cat(all_test_x),
            torch.cat(all_test_y),
            domains=all_test_domains if all_test_domains else None,
            is_labeled=torch.cat(all_test_labeled) if all_test_labeled else None,
        )

    return (
        train_datasets,
        test_datasets,
        train_counts,
        num_class,
    )
