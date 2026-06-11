import os
import zipfile

import requests
import torch
from PIL import Image
from torchvision import transforms


def download_and_extract(root):
    if not os.path.exists(root):
        os.makedirs(root)

    zip_path = os.path.join(root, "tiny-imagenet-200.zip")
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"

    if not os.path.exists(zip_path):
        print(f"-> Downloading Tiny-ImageNet-200 from {url}...")
        r = requests.get(url, stream=True)
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        print("-> Download complete.")

    extract_path = os.path.join(root, "tiny-imagenet-200")
    if not os.path.exists(extract_path):
        print("-> Extracting Tiny-ImageNet-200...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(root)
        print("-> Extraction complete.")

    return extract_path


def process(output_dir="./datasets/raw"):
    """
    下载并处理 Tiny-ImageNet-200 数据集。
    合并 100,000 张训练图和 10,000 张验证图（带标签）。
    """
    dataset_path = download_and_extract(output_dir)

    # 1. 建立类别映射 (WNID -> Label ID)
    wnids_path = os.path.join(dataset_path, "wnids.txt")
    with open(wnids_path, "r") as f:
        wnids = [line.strip() for line in f.readlines()]
    wnid_to_label = {wnid: i for i, wnid in enumerate(wnids)}

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    all_x = []
    all_y = []

    print("-> Loading Training Data (100,000 images)...")
    for wnid in wnids:
        label = wnid_to_label[wnid]
        img_dir = os.path.join(dataset_path, "train", wnid, "images")
        for img_name in os.listdir(img_dir):
            if img_name.endswith(".JPEG"):
                img_path = os.path.join(img_dir, img_name)
                with Image.open(img_path).convert("RGB") as img:
                    all_x.append(transform(img))
                    all_y.append(label)

    print("-> Loading Validation Data (10,000 images)...")
    val_annotations_path = os.path.join(dataset_path, "val", "val_annotations.txt")
    with open(val_annotations_path, "r") as f:
        for line in f.readlines():
            parts = line.strip().split("\t")
            img_name = parts[0]
            wnid = parts[1]
            label = wnid_to_label[wnid]
            img_path = os.path.join(dataset_path, "val", "images", img_name)
            with Image.open(img_path).convert("RGB") as img:
                all_x.append(transform(img))
                all_y.append(label)

    # 转换为张量
    print("-> Converting to Tensors...")
    all_x = torch.stack(all_x)
    all_y = torch.tensor(all_y, dtype=torch.long)

    processed_data = {"x": all_x, "y": all_y, "num_classes": 200}

    # 保存
    save_path = os.path.join(output_dir, "tiny_imagenet_raw.pt")
    torch.save(processed_data, save_path)
    print(f"-> Tiny-ImageNet raw data saved to {save_path}")
    print(f"   Shape: {all_x.shape}, Labels: {all_y.shape}")

    return processed_data
