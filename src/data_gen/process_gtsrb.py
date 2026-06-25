import os

import kagglehub
import pandas as pd
import torch
from PIL import Image
from torchvision import datasets, transforms

def process(output_dir="./datasets/raw"):
    """
    使用 kagglehub 自动下载并处理 GTSRB (德国交通标志) 数据集。
    适配 meowmeowmeowmeowmeow/gtsrb-german-traffic-sign 的结构。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("-> Downloading/Updating GTSRB dataset via kagglehub...")
    # 自动下载最新版本并返回本地缓存路径
    download_path = kagglehub.dataset_download(
        "meowmeowmeowmeowmeow/gtsrb-german-traffic-sign"
    )
    print(f"-> GTSRB cached at: {download_path}")

    # 定义数据转换 (GTSRB 通常 Resize 为 32x32)
    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.3337, 0.3064, 0.3171], std=[0.2672, 0.2564, 0.2629]
            ),
        ]
    )

    all_x = []
    all_y = []

    # 1. 处理训练集 (Train 文件夹下按 0, 1, ..., 42 分类)
    train_dir = os.path.join(download_path, "train")
    if os.path.exists(train_dir):
        print("-> Processing Training split...")
        # ImageFolder 自动处理子目录作为类标签
        train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=128, num_workers=4
        )

        for x, y in train_loader:
            all_x.append(x)
            all_y.append(y)

    # 2. 处理测试集 (根据 Test.csv 加载)
    test_csv = os.path.join(download_path, "Test.csv")
    test_dir = download_path  # Test.csv 中的路径通常是相对根目录的 'Test/xxxx.png'

    if os.path.exists(test_csv):
        print("-> Processing Test split...")
        df_test = pd.read_csv(test_csv)

        batch_x = []
        batch_y = []
        for index, row in df_test.iterrows():
            img_path = os.path.join(test_dir, row["Path"])
            label = int(row["ClassId"])

            if os.path.exists(img_path):
                try:
                    img = Image.open(img_path).convert("RGB")
                    batch_x.append(transform(img))
                    batch_y.append(torch.tensor(label))
                except Exception as e:
                    print(f"跳过损坏图片 {img_path}: {e}")

            # 分批处理以节省内存并显示进度
            if len(batch_x) >= 1000:
                all_x.append(torch.stack(batch_x))
                all_y.append(torch.tensor(batch_y))
                batch_x = []
                batch_y = []
                print(
                    f"   Loaded {len(all_x) * 1000 if isinstance(all_x[-1], torch.Tensor) else len(all_x)} images..."
                )

        if batch_x:
            all_x.append(torch.stack(batch_x))
            all_y.append(torch.tensor(batch_y))

    if not all_x:
        raise ValueError(
            f"未能从 {download_path} 加载任何数据。请检查 Kaggle 数据集结构。"
        )

    # 转为 Tensor
    print("-> Converting to tensors...")
    all_x_tensor = torch.cat(
        [x if x.dim() == 4 else x.unsqueeze(0) for x in all_x], dim=0
    )
    all_y_tensor = torch.cat(
        [y if y.dim() == 1 else y.unsqueeze(0) for y in all_y], dim=0
    )
    all_x_tensor = torch.cat([x if x.dim() == 4 else x.unsqueeze(0) for x in all_x], dim=0)
    all_y_tensor = torch.cat([y if y.dim() == 1 else y.unsqueeze(0) for y in all_y], dim=0)

    # 封装处理后的数据 (GTSRB 有 43 类)
    processed_data = {"x": all_x_tensor, "y": all_y_tensor, "num_classes": 43}

    # 保存
    save_path = os.path.join(output_dir, "gtsrb_raw.pt")
    torch.save(processed_data, save_path)
    print(f"-> GTSRB processed data saved to {save_path}")
    print(f"   Shape: {all_x_tensor.shape}, Classes: 43")

    return processed_data
