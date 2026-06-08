import os

import torch
from torchvision import datasets, transforms


def process(output_dir="./datasets/raw"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
    ])

    train_set = datasets.SVHN(
        root=output_dir, split="train", download=True, transform=transform
    )
    test_set = datasets.SVHN(
        root=output_dir, split="test", download=True, transform=transform
    )

    def get_all_tensors(dataset):
        loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset))
        return next(iter(loader))

    x_train, y_train = get_all_tensors(train_set)
    x_test, y_test = get_all_tensors(test_set)

    all_x = torch.cat([x_train, x_test], dim=0)
    all_y = torch.cat([y_train, y_test], dim=0)

    processed_data = {"x": all_x, "y": all_y, "num_classes": 10}

    save_path = os.path.join(output_dir, "svhn_raw.pt")
    torch.save(processed_data, save_path)
    print(f"-> SVHN raw data saved to {save_path}")

    return processed_data
