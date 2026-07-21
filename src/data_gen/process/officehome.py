import os
import urllib.request
import zipfile

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

OFFICEHOME_DOMAINS = ["Art", "Clipart", "Product", "RealWorld"]
OFFICEHOME_URL = (
    "https://drive.google.com/uc?export=download&id=0B81rNlvGIcedR2VmS3VWVmtyeTA"
)
NUM_CLASSES = 65


def load_images(domain_dir, domain_name, transform, class_to_idx):
    dataset = datasets.ImageFolder(root=domain_dir, transform=transform)
    # Remap folder names to consistent class indices
    remapped_targets = []
    for _, y in dataset.samples:
        folder_name = dataset.classes[y]
        remapped_targets.append(class_to_idx[folder_name])
    loader = DataLoader(dataset, batch_size=len(dataset))
    x, _ = next(iter(loader))
    y = torch.tensor(remapped_targets, dtype=torch.long)
    num_samples = len(x)
    domains = [domain_name] * num_samples
    return x, y, domains


def process(output_dir="./datasets/raw"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    save_path = os.path.join(output_dir, "officehome_raw.pt")
    if os.path.exists(save_path):
        print(f"-> OfficeHome raw data already exists at {save_path}")
        return torch.load(save_path, weights_only=False)

    data_dir = os.path.join(output_dir, "OfficeHome")
    is_extracted = os.path.exists(data_dir)
    if not is_extracted:
        zip_path = os.path.join(output_dir, "OfficeHome.zip")
        if not os.path.exists(zip_path):
            print("-> OfficeHome dataset not found. Attempting download...")
            download_officehome(zip_path)
        print("-> Extracting OfficeHome...")
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

    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Build a unified class mapping from the first domain
    first_domain = OFFICEHOME_DOMAINS[0]
    first_path = os.path.join(data_dir, first_domain)
    ref_dataset = datasets.ImageFolder(root=first_path)
    sorted_classes = sorted(ref_dataset.classes)
    class_to_idx = {cls: i for i, cls in enumerate(sorted_classes)}
    if len(class_to_idx) != NUM_CLASSES:
        print(f"  Warning: found {len(class_to_idx)} classes, expected {NUM_CLASSES}")

    all_x, all_y, all_domains = [], [], []
    for domain in OFFICEHOME_DOMAINS:
        domain_path = os.path.join(data_dir, domain)
        if not os.path.isdir(domain_path):
            print(f"  Warning: domain '{domain}' not found at {domain_path}, skipping.")
            continue
        x, y, doms = load_images(domain_path, domain, transform, class_to_idx)
        all_x.append(x)
        all_y.append(y)
        all_domains.extend(doms)
        print(f"  Loaded {domain}: {len(x)} samples, {len(torch.unique(y))} classes")

    processed_data = {
        "x": torch.cat(all_x, dim=0),
        "y": torch.cat(all_y, dim=0),
        "num_classes": NUM_CLASSES,
        "domains": all_domains,
        "domain_names": OFFICEHOME_DOMAINS,
    }
    torch.save(processed_data, save_path)
    print(f"-> OfficeHome raw data saved to {save_path}")
    print(
        f"   Total samples: {len(processed_data['x'])}, domains: {processed_data['domain_names']}"
    )
    return processed_data


def download_officehome(zip_path):

    try:
        urllib.request.urlretrieve(OFFICEHOME_URL, zip_path)
        return
    except Exception as e:
        print(f"  Direct download failed: {e}")

    import subprocess

    try:
        subprocess.run(
            ["wget", "--no-check-certificate", "-O", zip_path, OFFICEHOME_URL],
            check=True,
            capture_output=True,
        )
        return
    except Exception:
        pass

    raise RuntimeError(
        "OfficeHome dataset download failed. Please download manually:\n"
        f"  1. Visit: https://www.kaggle.com/datasets/charvitgalani/officehome-dataset\n"
        f"  2. Download and place the zip at: {zip_path}\n"
        f"  3. Or install gdown: pip install gdown"
    )
