import os

import torch

from .partition import prepare_domain_data, prepare_label_data
from .process import process_dataset


def load_data(args):
    raw_dir = "./datasets/raw"
    dataset_name = getattr(args, "domain_dataset", None) or args.dataset
    raw_path = os.path.join(raw_dir, f"{dataset_name}_raw.pt")
    if not os.path.exists(raw_path):
        print(f"-> Raw data for {dataset_name} not found. Processing...")
        process_dataset(dataset_name, raw_dir)
    return dataset_name, torch.load(raw_path, weights_only=False)


def prepare_data(args):
    dataset_name, raw_data = load_data(args)
    if getattr(args, "domain_dataset", None) is not None:
        prepare_domain_data(args, dataset_name, raw_data)
    else:
        prepare_label_data(args, dataset_name, raw_data)
