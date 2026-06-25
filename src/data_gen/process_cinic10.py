import os
import tarfile
import urllib.request

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

CINIC10_URL = (
    "https://datashare.is.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz"
)
CINIC10_SPLITS = ["train", "valid", "test"]


def process(output_dir="./datasets/raw"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    tar_path = os.path.join(output_dir, "CINIC-10.tar.gz")
    if not os.path.exists(tar_path):
        print("-> Downloading CINIC-10 (may take a while)...")
        urllib.request.urlretrieve(CINIC10_URL, tar_path)
        print("-> Download complete.")
    else:
        print("-> CINIC-10 tar.gz already exists, skipping download.")

    extract_dir = os.path.join(output_dir, "cinic10_extracted")
    extracted_flag = os.path.join(extract_dir, "train")
    if not os.path.exists(extracted_flag):
        print("-> Extracting CINIC-10...")
        os.makedirs(extract_dir, exist_ok=True)
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=extract_dir)
        print("-> Extraction complete.")
    else:
        print("-> Already extracted, skipping.")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.47889522, 0.47227842, 0.43047404),
                (0.24205776, 0.23828046, 0.25874835),
            ),
        ]
    )

    all_x, all_y = [], []
    base_dir = extract_dir
    for split in CINIC10_SPLITS:
        split_dataset = datasets.ImageFolder(
            root=os.path.join(base_dir, split), transform=transform
        )
        loader = DataLoader(split_dataset, batch_size=len(split_dataset))
        x, y = next(iter(loader))
        all_x.append(x)
        all_y.append(y)

    all_x = torch.cat(all_x, dim=0)
    all_y = torch.cat(all_y, dim=0)

    processed_data = {"x": all_x, "y": all_y, "num_classes": 10}

    save_path = os.path.join(output_dir, "cinic10_raw.pt")
    torch.save(processed_data, save_path)
    print(f"-> CINIC-10 raw data saved to {save_path}")
    print(f"   Total samples: {len(all_x)}, shape: {all_x.shape}")

    return processed_data
