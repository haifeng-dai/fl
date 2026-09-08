import os

import torch
from torchvision import datasets, transforms


def process(output_dir="./datasets/raw"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train_set = datasets.EMNIST(root=output_dir, split="byclass", train=True, download=True)
    test_set = datasets.EMNIST(root=output_dir, split="byclass", train=False, download=True)

    # train_set.data is torch.Tensor (N, 28, 28) uint8
    x_train = train_set.data.unsqueeze(1)
    y_train = train_set.targets.clone().detach().to(torch.long)
    x_test = test_set.data.unsqueeze(1)
    y_test = test_set.targets.clone().detach().to(torch.long)

    all_x = torch.cat([x_train, x_test], dim=0)
    all_y = torch.cat([y_train, y_test], dim=0)
    assert all_x.dtype == torch.uint8, f"Expected uint8, got {all_x.dtype}"

    processed_data = {
        "x": all_x,
        "y": all_y,
        "num_classes": 62,
    }

    save_path = os.path.join(output_dir, "femnist_raw.pt")
    torch.save(processed_data, save_path)
    print(f"-> FEMNIST raw data saved to {save_path}")

    return processed_data
