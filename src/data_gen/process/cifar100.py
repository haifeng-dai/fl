import os

import torch
from torchvision import datasets, transforms


def process(output_dir="./datasets/raw"):
    """
    下载并处理 CIFAR100 数据集，合并训练集和测试集为 uint8 [0, 255]。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 下载并加载 CIFAR100 训练集与测试集（无变换，保留原始 uint8）
    train_set = datasets.CIFAR100(root=output_dir, train=True, download=True)
    test_set = datasets.CIFAR100(root=output_dir, train=False, download=True)

    # data 是 numpy 数组 (N, H, W, C)，转为 Tensor (N, C, H, W)，dtype 为 torch.uint8
    x_train = torch.from_numpy(train_set.data).permute(0, 3, 1, 2)
    y_train = torch.tensor(train_set.targets, dtype=torch.long)
    x_test = torch.from_numpy(test_set.data).permute(0, 3, 1, 2)
    y_test = torch.tensor(test_set.targets, dtype=torch.long)

    all_x = torch.cat([x_train, x_test], dim=0)
    all_y = torch.cat([y_train, y_test], dim=0)

    assert all_x.dtype == torch.uint8, f"Expected uint8, got {all_x.dtype}"

    processed_data = {
        "x": all_x,
        "y": all_y,
        "num_classes": 100,
    }

    save_path = os.path.join(output_dir, "cifar100_raw.pt")
    torch.save(processed_data, save_path)
    print(f"-> CIFAR100 raw data saved to {save_path}")

    return processed_data
