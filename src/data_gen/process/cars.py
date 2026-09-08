import os

import torch
from torchvision import datasets, transforms


def process(output_dir="./datasets/raw"):
    """
    使用本地手动下载并解压的 Stanford_Cars.tar 数据集进行处理。
    数据集结构预期为:
    extracted_cars/
        train/
            Class_A/
            Class_B/
        test/
            Class_A/
            Class_B/
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    extracted_path = os.path.join(output_dir, "extracted_cars")
    if not os.path.exists(extracted_path):
        raise FileNotFoundError(f"找不到解压后的数据集目录: {extracted_path}")

    print(f"-> Processing Stanford Cars from manual extraction: {extracted_path}")

    # 定义数据转换
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.PILToTensor(),
        ]
    )

    all_x = []
    all_y = []

    # 直接使用 ImageFolder 加载 train 和 test
    for split in ["train", "test"]:
        split_path = os.path.join(extracted_path, split)
        if os.path.exists(split_path):
            print(f"   Loading {split} split...")
            # ImageFolder 会自动将子文件夹作为标签
            dataset = datasets.ImageFolder(root=split_path, transform=transform)
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=128, shuffle=False, num_workers=4
            )

            for x, y in loader:
                all_x.append(x)
                all_y.append(y)
        else:
            print(f"   Warning: Split {split} not found in {extracted_path}")

    if not all_x:
        raise ValueError("未能加载任何有效图片。")

    # 转为 Tensor
    print("-> Converting to tensors...")
    all_x_tensor = torch.cat(all_x, dim=0)
    all_y_tensor = torch.cat(all_y, dim=0)
    assert all_x_tensor.dtype == torch.uint8, f"Expected uint8, got {all_x_tensor.dtype}"

    # 封装处理后的数据 (Stanford Cars 196 类)
    num_classes = len(torch.unique(all_y_tensor))
    processed_data = {
        "x": all_x_tensor,
        "y": all_y_tensor,
        "num_classes": num_classes,
    }

    # 保存
    save_path = os.path.join(output_dir, "cars_raw.pt")
    torch.save(processed_data, save_path)
    print(f"-> Stanford Cars processed data saved to {save_path}")
    print(f"   Final Shape: {all_x_tensor.shape}, Detected Classes: {num_classes}")

    return processed_data
