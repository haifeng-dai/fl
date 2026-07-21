import os

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

# 定义 4 种不同的"增广策略"作为合成领域
DOMAIN_TRANSFORMS = {
    "clean": transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    ),
    "color_jitter": transforms.Compose(
        [
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    ),
    "blur_noise": transforms.Compose(
        [
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    ),
    "rotate_cutout": transforms.Compose(
        [
            transforms.RandomRotation(degrees=30),
            transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    ),
}


def process(output_dir="./datasets/raw"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    save_path = os.path.join(output_dir, "cifar10_dg_raw.pt")
    if os.path.exists(save_path):
        print(f"-> CIFAR-10 DG raw data already exists at {save_path}")
        return torch.load(save_path, weights_only=False)

    print("-> Loading CIFAR-10 (this will download if needed)...")
    full_dataset = datasets.CIFAR10(
        root=output_dir,
        train=True,
        download=True,
        transform=None,
    )
    # Also include test split
    test_dataset = datasets.CIFAR10(
        root=output_dir,
        train=False,
        download=True,
        transform=None,
    )
    # Concatenate train + test
    combined_x = torch.cat(
        [
            torch.tensor(full_dataset.data),
            torch.tensor(test_dataset.data),
        ],
        dim=0,
    )
    combined_y = torch.cat(
        [
            torch.tensor(full_dataset.targets),
            torch.tensor(test_dataset.targets),
        ],
        dim=0,
    )

    # Create a temporary dataset with all samples
    class TempDataset(torch.utils.data.Dataset):
        def __init__(self, x, y, transform):
            self.data = x
            self.targets = y
            self.transform = transform

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            img = self.data[idx]
            img = transforms.ToPILImage()(img.permute(2, 0, 1))
            if self.transform:
                img = self.transform(img)
            return img, self.targets[idx]

    domain_names = list(DOMAIN_TRANSFORMS.keys())
    all_x, all_y, all_domains = [], [], []

    for dname, dtransform in DOMAIN_TRANSFORMS.items():
        temp_ds = TempDataset(combined_x, combined_y, dtransform)
        loader = DataLoader(temp_ds, batch_size=len(temp_ds))
        x, y = next(iter(loader))
        doms = [dname] * len(x)
        all_x.append(x)
        all_y.append(y)
        all_domains.extend(doms)
        print(f"  Generated domain '{dname}': {len(x)} samples")

    processed_data = {
        "x": torch.cat(all_x, dim=0),
        "y": torch.cat(all_y, dim=0),
        "num_classes": 10,
        "domains": all_domains,
        "domain_names": domain_names,
    }
    torch.save(processed_data, save_path)
    print(f"-> CIFAR-10 DG raw data saved to {save_path}")
    print(
        f"   Total samples: {len(processed_data['x'])}, domains: {processed_data['domain_names']}"
    )
    return processed_data
