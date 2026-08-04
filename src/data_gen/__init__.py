import os

import torch

from .partition import (
    apply_label_ratio_client,
    apply_label_ratio_sample,
    prepare_fdg_data,
    prepare_label_data,
    prepare_sfd_data,
)
from .partition.common import get_output_dir, is_fresh
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

    if args.sfd:
        prepare_sfd_data(args, dataset_name, raw_data)
    elif args.fdg:
        prepare_fdg_data(args, dataset_name, raw_data)
    else:
        # 默认：类别划分（维度二）；ssl 掩码作为完全独立的后处理步骤
        output_dir = get_output_dir(args, dataset_name)
        # 统一判定复用 is_fresh，避免与 domain 分支各自维护一份重复逻辑。
        fresh = is_fresh(output_dir, args.num_clients)
        if fresh:
            prepare_label_data(args, dataset_name, raw_data)
        # ssl 掩码幂等且与当前 config 绑定（apply_label_ratio_* 每次重算 is_labeled），
        # 缓存目录名未编码 ssl/label_ratio，故必须按当前配置重新应用，确保配置即时生效。
        ssl = getattr(args, "ssl", "none")
        if ssl == "sample":
            apply_label_ratio_sample(output_dir, args.num_clients, args.label_ratio)
        elif ssl == "client":
            apply_label_ratio_client(output_dir, args.num_clients, args.label_ratio)
