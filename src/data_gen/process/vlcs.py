import os
import zipfile

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

VLCS_DOMAINS = ["VOC2007", "LabelMe", "Caltech101", "SUN09"]
# VLCS 固定 5 类，跨领域一致
VLCS_CLASSES = ["bird", "car", "chair", "dog", "person"]
VLCS_URL = (
    "https://drive.google.com/uc?export=download&id=1skw4DVQH0SSJmls5gHnZz2Ni3-yO8V6H"
)


def load_images(domain_dir, domain_name, transform):
    dataset = datasets.ImageFolder(root=domain_dir, transform=transform)
    # 只保留 VLCS_CLASSES 中的类别，统一 idx
    class_to_idx = {c: i for i, c in enumerate(VLCS_CLASSES)}
    kept_x, kept_y, kept_domains = [], [], []
    loader = DataLoader(dataset, batch_size=len(dataset))
    all_x, all_y = next(iter(loader))
    for i in range(len(all_y)):
        # 获取原始标签对应的文件夹名
        folder_name = dataset.classes[all_y[i].item()]
        if folder_name in class_to_idx:
            kept_x.append(all_x[i])
            kept_y.append(class_to_idx[folder_name])
            kept_domains.append(domain_name)
    if kept_x:
        return (
            torch.stack(kept_x),
            torch.tensor(kept_y, dtype=torch.long),
            kept_domains,
        )
    return torch.empty(0), torch.empty(0, dtype=torch.long), []


def process(output_dir="./datasets/raw"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    save_path = os.path.join(output_dir, "vlcs_raw.pt")
    if os.path.exists(save_path):
        print(f"-> VLCS raw data already exists at {save_path}")
        return torch.load(save_path, weights_only=False)

    data_dir = os.path.join(output_dir, "VLCS")
    is_extracted = os.path.exists(data_dir)
    if not is_extracted:
        zip_path = os.path.join(output_dir, "VLCS.zip")
        if not os.path.exists(zip_path):
            print("-> VLCS dataset not found. Attempting download...")
            download_vlcs(zip_path)
        print("-> Extracting VLCS...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(data_dir)
        # Handle nested folder
        contents = os.listdir(data_dir)
        if len(contents) == 1 and os.path.isdir(os.path.join(data_dir, contents[0])):
            inner = os.path.join(data_dir, contents[0])
            for d in os.listdir(inner):
                os.rename(os.path.join(inner, d), os.path.join(data_dir, d))
            os.rmdir(inner)
        print("-> Extraction complete.")

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.PILToTensor(),
    ])

    all_x, all_y, all_domains = [], [], []
    for domain in VLCS_DOMAINS:
        domain_path = os.path.join(data_dir, domain)
        if not os.path.isdir(domain_path):
            print(f"  Warning: domain '{domain}' not found at {domain_path}, skipping.")
            continue
        x, y, doms = load_images(domain_path, domain, transform)
        if len(x) == 0:
            print(f"  Warning: domain '{domain}' has 0 valid samples, skipping.")
            continue
        all_x.append(x)
        all_y.append(y)
        all_domains.extend(doms)
        print(f"  Loaded {domain}: {len(x)} samples, {len(torch.unique(y))} classes")

    out_x = torch.cat(all_x, dim=0)
    out_y = torch.cat(all_y, dim=0)
    assert out_x.dtype == torch.uint8, f"Expected uint8, got {out_x.dtype}"

    processed_data = {
        "x": out_x,
        "y": out_y,
        "num_classes": len(VLCS_CLASSES),
        "domains": all_domains,
        "domain_names": [d for d in VLCS_DOMAINS
                         if os.path.isdir(os.path.join(data_dir, d))],
    }
    torch.save(processed_data, save_path)
    print(f"-> VLCS raw data saved to {save_path}")
    print(f"   Total samples: {len(processed_data['x'])}, "
          f"domains: {processed_data['domain_names']}")
    return processed_data


def download_vlcs(zip_path):
    import urllib.request
    try:
        urllib.request.urlretrieve(VLCS_URL, zip_path)
        return
    except OSError as e:
        print(f"  Direct download failed: {e}")

    import subprocess
    try:
        subprocess.run(
            ["wget", "--no-check-certificate", "-O", zip_path, VLCS_URL],
            check=True, capture_output=True,
        )
        return
    except (OSError, subprocess.CalledProcessError):
        print("  wget 下载失败，请参考下方手动下载指引。")

    raise RuntimeError(
        "VLCS dataset download failed. Please download manually:\n"
        f"  1. Install gdown: pip install gdown\n"
        f"  2. Run: gdown {VLCS_URL} -O {zip_path}\n"
        "  Or manually download from a trusted source and place at:\n"
        f"  {zip_path}"
    )
