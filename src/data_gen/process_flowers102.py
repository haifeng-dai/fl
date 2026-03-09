import os

import torch
from torchvision import datasets, transforms


def process(output_dir="./datasets/raw"):
    """
    下载并处理 Flowers102 数据集，合并 train, val, test 集合。
    统一 Resize 到 224x224 并进行标准化。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 定义数据转换
    # Flowers102 图片尺寸不一，必须 Resize
    # 使用 ImageNet 的均值和标准差
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    splits = ["train", "val", "test"]
    all_x_list = []
    all_y_list = []

    print("-> Downloading/Processing Flowers102 (this may take a while)...")

    for split in splits:
        dataset = datasets.Flowers102(
            root=output_dir, split=split, download=True, transform=transform
        )

        # 使用 DataLoader 分批读取，避免一次性加载导致的内存压力（虽然数据量不大）
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=128, shuffle=False, num_workers=4
        )

        for x, y in loader:
            all_x_list.append(x)
            all_y_list.append(y)

    # 合并所有数据
    all_x = torch.cat(all_x_list, dim=0)
    all_y = torch.cat(all_y_list, dim=0)

    # 封装处理后的数据
    processed_data = {"x": all_x, "y": all_y, "num_classes": 102}

    # 保存
    save_path = os.path.join(output_dir, "flowers102_raw.pt")
    torch.save(processed_data, save_path)
    print(f"-> Flowers102 raw data saved to {save_path}")
    print(f"   Data shape: {all_x.shape}, Labels shape: {all_y.shape}")

    return processed_data
