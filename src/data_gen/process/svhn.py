import os

import torch
from torchvision import datasets, transforms


def process(output_dir="./datasets/raw"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train_set = datasets.SVHN(root=output_dir, split="train", download=True)
    test_set = datasets.SVHN(root=output_dir, split="test", download=True)

    # SVHN data is numpy array (N, 3, 32, 32) uint8
    x_train = torch.from_numpy(train_set.data)
    y_train = torch.tensor(train_set.labels, dtype=torch.long)
    x_test = torch.from_numpy(test_set.data)
    y_test = torch.tensor(test_set.labels, dtype=torch.long)

    all_x = torch.cat([x_train, x_test], dim=0)
    all_y = torch.cat([y_train, y_test], dim=0)
    assert all_x.dtype == torch.uint8, f"Expected uint8, got {all_x.dtype}"

    processed_data = {
        "x": all_x,
        "y": all_y,
        "num_classes": 10,
    }

    save_path = os.path.join(output_dir, "svhn_raw.pt")
    torch.save(processed_data, save_path)
    print(f"-> SVHN raw data saved to {save_path}")

    return processed_data
