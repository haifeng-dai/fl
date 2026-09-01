import os
import zipfile

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

PACS_DOMAINS = ["photo", "art_painting", "cartoon", "sketch"]
PACS_URL = "https://drive.google.com/uc?export=download&id=1J0dMEWn3UvBqP9OOmCvLq0NIgCceSZ4c"


def load_images(domain_dir, domain_name, transform):
    dataset = datasets.ImageFolder(root=domain_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=len(dataset))
    x, y = next(iter(loader))
    num_samples = len(x)
    domains = [domain_name] * num_samples
    return x, y, domains


def process(output_dir="./datasets/raw"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    save_path = os.path.join(output_dir, "pacs_raw.pt")
    if os.path.exists(save_path):
        print(f"-> PACS raw data already exists at {save_path}")
        return torch.load(save_path, weights_only=False)

    data_dir = os.path.join(output_dir, "pacs")
    is_extracted = os.path.exists(data_dir)
    if not is_extracted:
        zip_path = os.path.join(output_dir, "pacs.zip")
        if not os.path.exists(zip_path):
            print("-> PACS dataset not found. Attempting download...")
            download_pacs(zip_path)
        print("-> Extracting PACS...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(data_dir)
        # Find actual data directory (zip may have nested folder)
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
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    all_x, all_y, all_domains = [], [], []
    for domain in PACS_DOMAINS:
        domain_path = os.path.join(data_dir, domain)
        if not os.path.isdir(domain_path):
            print(f"  Warning: domain '{domain}' not found at {domain_path}, skipping.")
            continue
        x, y, doms = load_images(domain_path, domain, transform)
        all_x.append(x)
        all_y.append(y)
        all_domains.extend(doms)
        print(f"  Loaded {domain}: {len(x)} samples, {len(torch.unique(y))} classes")

    processed_data = {
        "x": torch.cat(all_x, dim=0),
        "y": torch.cat(all_y, dim=0),
        "num_classes": len(torch.unique(torch.cat(all_y, dim=0))),
        "domains": all_domains,
        "domain_names": PACS_DOMAINS,
    }
    torch.save(processed_data, save_path)
    print(f"-> PACS raw data saved to {save_path}")
    print(f"   Total samples: {len(processed_data['x'])}, domains: {processed_data['domain_names']}")
    return processed_data


def download_pacs(zip_path):
    import urllib.request
    try:
        urllib.request.urlretrieve(PACS_URL, zip_path)
        return
    except OSError as e:
        print(f"  Direct download failed: {e}")

    import subprocess
    try:
        subprocess.run(["wget", "--no-check-certificate", "-O", zip_path, PACS_URL], check=True, capture_output=True)
        return
    except (OSError, subprocess.CalledProcessError):
        print("  wget 下载失败，请参考下方手动下载指引。")

    raise RuntimeError(
        "PACS dataset download failed. Please download manually:\n"
        f"  1. Visit: https://www.kaggle.com/datasets/samjwright/pacs-image-dataset\n"
        f"  2. Download and place the zip at: {zip_path}\n"
        f"  3. Or install gdown: pip install gdown"
    )
