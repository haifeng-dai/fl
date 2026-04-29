import argparse
import datetime
import importlib
import random
import sys
import time

import numpy as np
import torch

from src import get_pre_name, prepare_data


def set_seed(seed):
    """设置所有随机种子以确保实验可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_args():
    parser = argparse.ArgumentParser(description="Unified FL Framework", add_help=False)
    parser.add_argument(
        "--algo",
        type=str,
        default="fedavg",
        # choices=[
        #     "fedavg",
        #     "moon",
        #     "fedpln",
        #     "feddpl",
        #     "feddpl1",
        #     "feddpl2",
        #     "feddpl3",
        #     "feddpl4",
        #     "fedproto",
        #     "fedkd",
        #     "fml",
        #     "proxyfl",
        #     "fedper",
        #     "fedprox",
        #     "fedsa",
        #     "fedlsa",
        #     "lgfedavg",
        #     "fedrep",
        #     "fedala",
        #     "fedtgp",
        #     "fedtgp1",
        #     "fedtgp2",
        #     "fedtgp3",
        #     "fedtest",
        #     "feddyn",
        #     "scaffold",
        #     "fedfm",
        #     "fedproc",
        #     "local",
        # ],
    )
    parser.add_argument("--test", type=int, default=0, help="Test or train")
    parser.add_argument(
        "--feature_dim",
        type=int,
        default=512,
        help="Feature dimension for prototypes (default: 512)",
    )
    args, _ = parser.parse_known_args()

    # 2. 构建完整的解析器
    full_parser = argparse.ArgumentParser(parents=[parser])

    # 数据相关参数
    data_group = full_parser.add_argument_group("Data & Partitioning Arguments")
    data_group.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        help="Dataset name",
        choices=[
            "mnist",
            "cifar10",
            "cifar100",
            "flowers102",
            "cars",
            "gtsrb",
            "har",
            "har_feat",
            "tiny_imagenet",
        ],
    )
    data_group.add_argument(
        "--model",
        type=str,
        default="cnn",
        help="Model architecture",
        choices=["cnn", "resnet18", "resnet50", "harcnn", "harmlp"],
    )
    data_group.add_argument(
        "--num_clients", type=int, default=10, help="Number of clients"
    )
    data_group.add_argument(
        "--partition",
        type=str,
        default="iid",
        choices=["iid", "dirichlet", "pathological"],
        help="Data partitioning strategy",
    )
    data_group.add_argument("--alpha", type=float, default=0.5, help="For Dirichlet")
    data_group.add_argument(
        "--n_class",
        type=int,
        default=0,
        help="For Pathological (0 means auto select: cifar10:2, cifar100:10, tiny_imagenet:20)",
    )
    data_group.add_argument(
        "--test_ratio", type=float, default=0.2, help="Ratio of test data"
    )

    # 训练相关参数
    train_group = full_parser.add_argument_group("Training Arguments")
    train_group.add_argument(
        "--join_ratio",
        type=float,
        default=1.0,
        help="Ratio of clients participating in each round",
    )
    train_group.add_argument(
        "--epochs", type=int, default=1, help="Number of local epochs"
    )
    train_group.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    train_group.add_argument(
        "--rounds", type=int, default=5, help="Number of communication rounds"
    )
    train_group.add_argument("--batch_size", type=int, default=64, help="Batch size")
    train_group.add_argument(
        "--gpus", type=str, default="0", help="Comma separated list of GPU ids"
    )
    train_group.add_argument(
        "--mp", type=int, default=0, help="Enable multiprocessing training"
    )
    train_group.add_argument(
        "--max_workers_per_gpu",
        type=int,
        default=1,
        help="Maximum number of parallel workers per GPU to avoid OOM",
    )
    train_group.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    train_group.add_argument(
        "--times", type=int, default=1, help="Number of times to run the experiment"
    )

    # 算法专属参数
    try:
        algo_module = importlib.import_module(f"src.{args.algo}")
    except ModuleNotFoundError:
        raise ValueError(f"Algorithm module src.{args.algo} not found.")

    if hasattr(algo_module, "add_args"):
        algo_module.add_args(full_parser)

    args = full_parser.parse_args()

    # 自动根据数据集设置 n_class (当 n_class 为 0 时)
    if args.n_class == 0:
        if args.dataset == "cifar100":
            args.n_class = 10
        elif args.dataset == "tiny_imagenet":
            args.n_class = 20
        else:
            args.n_class = 2  # 默认回退值

    get_pre_name(args)

    return args, algo_module


def main():
    args, algo_module = get_args()

    # 保存配置的实验总次数和初始随机种子
    total_times = args.times
    base_seed = args.seed

    for t in range(total_times):
        # 更新当前运行的索引和随机种子
        args.times = t
        args.seed = base_seed + t
        a = time.time()

        set_seed(args.seed)

        # 获取日志路径
        log_path = algo_module.get_path(args)

        original_stdout = sys.stdout
        original_stderr = sys.stderr
        log_f = None

        if not args.test:
            # 打开日志文件并将 stdout/stderr 重定向
            log_f = open(log_path, "w", encoding="utf-8", buffering=1)
            sys.stdout = log_f
            sys.stderr = log_f

        try:
            print(f"=== Experiment {t + 1}/{total_times} (Seed: {args.seed}) ===")
            print(
                f"Start time: {datetime.datetime.fromtimestamp(a).strftime('%Y-%m-%d %H:%M:%S')}\n"
            )

            prepare_data(
                dataset_name=args.dataset,
                partition_method=args.partition,
                num_clients=args.num_clients,
                alpha=args.alpha,
                n_classes=args.n_class,
                test_ratio=args.test_ratio,
            )

            server = algo_module.Server(args=args)
            server.fit()
            server.save()

            b = time.time()
            print(
                f"End time: {datetime.datetime.fromtimestamp(b).strftime('%Y-%m-%d %H:%M:%S')}"
            )
            delta = datetime.timedelta(seconds=int(b - a))
            print(f"\nTotal time: {delta}")
        finally:
            # 恢复 stdout/stderr 并关闭日志文件
            if not args.test:
                sys.stdout = original_stdout
                sys.stderr = original_stderr
                if log_f:
                    log_f.close()


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    torch.multiprocessing.set_sharing_strategy("file_system")
    main()
