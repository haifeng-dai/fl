import os

import torch

from .partition import (
    prepare_domain_data,
    prepare_label_data,
)
from .partition.common import get_output_dir
from .process import process_dataset
from .sfd import apply_sfd
from .ssl import (
    apply_label_ratio_client,
    apply_label_ratio_sample,
)

__all__ = [
    "prepare_data",
    "get_output_dir",
]


def prepare_data(args):
    """数据划分总入口（三维模型，优先级 sfd > dg > category）。

    - sfd=true ：维度三 SFD 特定场景（域偏移 + 类异质 + 半监督）。
        域内部做异质切分，再由 apply_sfd 写入 label/unlabel 掩码。
    - dg=true  ：维度一纯域泛化，不同客户端分配不同域，不考虑类数据划分。
    - 其余     ：维度二类别划分（iid/dirichlet/pathological）+ 可选 ssl 掩码。
    """
    dataset_name = args.dataset
    raw_dir = "./datasets/raw"
    raw_path = os.path.join(raw_dir, f"{dataset_name}_raw.pt")
    if not os.path.exists(raw_path):
        print(f"-> Raw data for {dataset_name} not found. Processing...")
        process_dataset(dataset_name, raw_dir)
    raw_data = torch.load(raw_path, weights_only=False)

    if args.sfd:
        # 维度三：SFD 是完全独立的场景——域选择在此完成，域划分复用通用函数
        if args.label_domain is None or args.unlabel_domain is None:
            raise ValueError("SFD 场景必须指定 label_domain 与 unlabel_domain")
        args.selected_domains = f"{args.label_domain},{args.unlabel_domain}"
        output_dir = prepare_domain_data(
            args, dataset_name, raw_data, heterogeneous=True
        )
        apply_sfd(
            output_dir,
            args.num_clients,
            args.label_domain,
            args.unlabel_domain,
            args.label_rate,
        )
    elif args.dg:
        # 维度一：纯域泛化（不考虑类数据划分）
        prepare_domain_data(args, dataset_name, raw_data, heterogeneous=False)
    else:
        # 默认：类别划分（维度二）；ssl 掩码作为完全独立的后处理步骤
        output_dir = prepare_label_data(args, dataset_name, raw_data)
        ssl = getattr(args, "ssl", "none")
        if ssl == "sample":
            apply_label_ratio_sample(output_dir, args.num_clients, args.label_ratio)
        elif ssl == "client":
            apply_label_ratio_client(output_dir, args.num_clients, args.label_ratio)
