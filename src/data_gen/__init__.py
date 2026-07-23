import os

import torch

from .partition import get_domain_partition_dir, get_partition_path, prepare_domain_data, prepare_label_data
from .process import process_dataset

__all__ = ["get_domain_partition_dir", "prepare_data"]


def prepare_data(args):
    dataset_name = args.dataset
    raw_dir = "./datasets/raw"
    raw_path = os.path.join(raw_dir, f"{dataset_name}_raw.pt")
    if not os.path.exists(raw_path):
        print(f"-> Raw data for {dataset_name} not found. Processing...")
        process_dataset(dataset_name, raw_dir)
    raw_data = torch.load(raw_path, weights_only=False)

    if getattr(args, "domain_partition", None) is not None:
        prepare_domain_data(args, dataset_name, raw_data)
    else:
        prepare_label_data(args, dataset_name, raw_data)
