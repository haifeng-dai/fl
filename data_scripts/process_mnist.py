import os

import torch
from torchvision import datasets, transforms


def process(output_dir="./datasets/raw"):
    """
    下载并处理 MNIST 数据集，合并训练集和测试集。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 定义数据转换：转换为张量并进行标准化
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    # 下载并加载 MNIST 训练集
    train_set = datasets.MNIST(
        root=output_dir, train=True, download=True, transform=transform
    )
    # 下载并加载 MNIST 测试集
    test_set = datasets.MNIST(
        root=output_dir, train=False, download=True, transform=transform
    )

    def get_all_tensors(dataset):
        """
        从数据集中获取所有图像和标签作为单个张量。
        """
        loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset))
        return next(iter(loader))

    # 获取训练集和测试集的所有张量
    x_train, y_train = get_all_tensors(train_set)
    x_test, y_test = get_all_tensors(test_set)

    # 合并训练集和测试集的图像和标签
    all_x = torch.cat([x_train, x_test], dim=0)
    all_y = torch.cat([y_train, y_test], dim=0)

    # 封装处理后的数据
    processed_data = {"x": all_x, "y": all_y, "num_classes": 10}

    return processed_data
