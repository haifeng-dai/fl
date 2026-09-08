import os

import torch

from src.algorithms.utils.input import DATASET_SPECS
from .partition import (
    prepare_fdg_data,
    prepare_label_data,
    prepare_mixed_ssl_data,
    prepare_sfd_data,
)
from .partition.common import get_output_dir, is_fresh, resolve_n_class
from .process import process_dataset

__all__ = [
    "get_output_dir",
    "prepare_data",
]


def prepare_data(args):
    """数据划分总入口。

    - sfd=true ：SFD 双域半监督场景（域偏移 + 类异质 + 半监督），
        完整流程内聚于 partition/sfd.py（域划分 + is_labeled 掩码）。
    - fdg=true ：FDG 纯域泛化，源域全训练、目标域作测试，
        完整流程内聚于 partition/fdg.py。
    - sample/double：按类别统一生成 L/U/Test 三池，再决定 L/U 到客户端的
        分配方式（sample 同分布、double 双异质），全局 Test 只保存一份。
    - 其余     ：类别划分（iid/dirichlet/pathological）+ 可选 ssl 掩码。

    配置层已保证 SSL 与 FDG 互斥；以下分支仅负责分派各个互斥场景。
    """
    dataset_name = args.dataset
    raw_dir = "./datasets/raw"
    raw_path = os.path.join(raw_dir, f"{dataset_name}_raw.pt")

    need_process = not os.path.exists(raw_path)
    if not need_process:
        raw_data = torch.load(raw_path, weights_only=False)
        # 若为图像数据集但缓存仍为历史旧版 float32，则自动触发重生成
        spec = DATASET_SPECS.get(dataset_name)
        if spec is not None and spec.get("kind") == "image":
            if raw_data.get("x") is not None and raw_data["x"].dtype != torch.uint8:
                print(f"-> 检测到 {dataset_name} 旧版数据缓存，正在重新生成 uint8 数据...")
                need_process = True

    if need_process:
        print(f"-> Raw data for {dataset_name} not found or outdated. Processing...")
        raw_data = process_dataset(dataset_name, raw_dir)

    resolve_n_class(args, raw_data["num_classes"])

    if args.ssl == "sfd":
        prepare_sfd_data(args, dataset_name, raw_data)
    elif args.fdg:
        prepare_fdg_data(args, dataset_name, raw_data)
    elif args.ssl in ("sample", "double"):
        # sample 与 double 共用同一份基础 L/U/Test 三池，仅 L/U 到客户端的
        # 分配方式不同：sample 同分布、double 双异质（由 args.ssl 唯一表达）。
        prepare_mixed_ssl_data(args, dataset_name, raw_data)
    else:
        output_dir = get_output_dir(args, dataset_name)
        should_partition = is_fresh(output_dir, args.num_clients)

        if should_partition:
            prepare_label_data(args, dataset_name, raw_data)
