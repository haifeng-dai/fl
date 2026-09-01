import os
import zipfile

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DOMAINNET_DOMAINS = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
# DomainNet 每个领域的 345 类文件夹名是一致的
DOMAINNET_BASE_URL = "http://csr.bu.edu/ftp/visda/2019/multi-source"
NUM_CLASSES = 345


def process(output_dir="./datasets/raw"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    save_path = os.path.join(output_dir, "domainnet_raw.pt")
    if os.path.exists(save_path):
        print(f"-> DomainNet raw data already exists at {save_path}")
        return torch.load(save_path, weights_only=False)

    data_dir = os.path.join(output_dir, "domainnet")
    os.makedirs(data_dir, exist_ok=True)

    all_x, all_y, all_domains = [], [], []
    global_class_map = None

    for domain in DOMAINNET_DOMAINS:
        domain_path = os.path.join(data_dir, domain)
        domain_zip = os.path.join(output_dir, f"{domain}.zip")

        # Download if needed
        if not os.path.isdir(domain_path):
            if not os.path.exists(domain_zip):
                print(f"-> Downloading DomainNet/{domain}...")
                download_domain(domain, domain_zip)
            print(f"-> Extracting {domain}...")
            os.makedirs(domain_path, exist_ok=True)
            with zipfile.ZipFile(domain_zip, "r") as zf:
                # DomainNet ZIP 内直接是类别文件夹
                zf.extractall(domain_path)
            # Handle nested folder
            contents = os.listdir(domain_path)
            if len(contents) == 1 and os.path.isdir(
                os.path.join(domain_path, contents[0])
            ):
                inner = os.path.join(domain_path, contents[0])
                for d in os.listdir(inner):
                    os.rename(os.path.join(inner, d), os.path.join(domain_path, d))
                os.rmdir(inner)
            print(f"  {domain} extraction complete.")

        # QuickDraw 是灰度线条图，特殊处理
        if domain == "quickdraw":
            t = transforms.Compose(
                [
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.Grayscale(num_output_channels=3),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            t = transforms.Compose(
                [
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

        # Build class map from first domain, then remap labels
        if global_class_map is None:
            ref = datasets.ImageFolder(root=domain_path)
            global_class_map = {c: i for i, c in enumerate(sorted(ref.classes))}

        dataset = datasets.ImageFolder(root=domain_path, transform=t)
        original_classes = dataset.classes
        loader = DataLoader(dataset, batch_size=len(dataset))
        x, raw_y = next(iter(loader))

        remapped = [global_class_map[original_classes[y.item()]] for y in raw_y]
        y = torch.tensor(remapped, dtype=torch.long)

        doms = [domain] * len(x)
        all_x.append(x)
        all_y.append(y)
        all_domains.extend(doms)
        print(f"  Loaded {domain}: {len(x)} samples, {len(torch.unique(y))} classes")

        # Clean up zip to save space
        if os.path.exists(domain_zip):
            os.remove(domain_zip)

    processed_data = {
        "x": torch.cat(all_x, dim=0),
        "y": torch.cat(all_y, dim=0),
        "num_classes": len(global_class_map) if global_class_map else NUM_CLASSES,
        "domains": all_domains,
        "domain_names": DOMAINNET_DOMAINS,
    }
    torch.save(processed_data, save_path)
    print(f"-> DomainNet raw data saved to {save_path}")
    print(
        f"   Total samples: {len(processed_data['x'])}, "
        f"domains: {processed_data['domain_names']}"
    )
    return processed_data


def download_domain(domain, save_path):
    url = f"{DOMAINNET_BASE_URL}/groundtruth/{domain}.zip"
    try:
        import urllib.request

        urllib.request.urlretrieve(url, save_path)
        return
    except OSError as e:
        print(f"  urllib download failed: {e}")

    import subprocess

    try:
        subprocess.run(
            ["wget", "--no-check-certificate", "-O", save_path, url],
            check=True,
            capture_output=True,
        )
        return
    except (OSError, subprocess.CalledProcessError):
        print("  wget 下载失败，请参考下方手动下载指引。")

    raise RuntimeError(
        f"DomainNet/{domain} download failed.\n"
        f"  Please manually download from:\n"
        f"  {url}\n"
        f"  And place the zip at: {save_path}"
    )
