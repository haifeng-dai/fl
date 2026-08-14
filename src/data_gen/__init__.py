import os

import torch

from .partition import (
    apply_label_ratio_client,
    apply_label_ratio_sample,
    prepare_fdg_data,
    prepare_label_data,
    prepare_sfd_data,
)
from .partition.common import get_output_dir, is_fresh, resolve_n_class
from .process import process_dataset

__all__ = [
    "prepare_data",
    "get_output_dir",
]


def prepare_data(args):
    """数据划分总入口（三维模型，优先级 sfd > fdg > category）。

    - sfd=true ：SFD 双域半监督场景（域偏移 + 类异质 + 半监督），
        完整流程内聚于 partition/sfd.py（域划分 + is_labeled 掩码）。
    - fdg=true ：FDG 纯域泛化，源域全训练、目标域作测试，
        完整流程内聚于 partition/fdg.py。
    - 其余     ：类别划分（iid/dirichlet/pathological）+ 可选 ssl 掩码。
    """
    dataset_name = args.dataset
    raw_dir = "./datasets/raw"
    raw_path = os.path.join(raw_dir, f"{dataset_name}_raw.pt")
    if not os.path.exists(raw_path):
        print(f"-> Raw data for {dataset_name} not found. Processing...")
        process_dataset(dataset_name, raw_dir)
    raw_data = torch.load(raw_path, weights_only=False)
    resolve_n_class(args, raw_data["num_classes"])

    if args.ssl == "sfd":
        prepare_sfd_data(args, dataset_name, raw_data)
    elif args.fdg:
        prepare_fdg_data(args, dataset_name, raw_data)
    else:
        output_dir = get_output_dir(args, dataset_name)
        if is_fresh(output_dir, args.num_clients):
            prepare_label_data(args, dataset_name, raw_data)
            if args.ssl == "sample":
                apply_label_ratio_sample(output_dir, args.num_clients, args.label_ratio)
            elif args.ssl == "client":
                apply_label_ratio_client(output_dir, args.num_clients, args.label_ratio)
