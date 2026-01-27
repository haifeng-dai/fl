import argparse
import datetime
import importlib
import time
import sys

import torch

from src import prepare_data


def get_args():
    parser = argparse.ArgumentParser(description="Unified FL Framework", add_help=False)
    parser.add_argument(
        "--algo",
        type=str,
        default="fedavg",
        choices=[
            "fedavg",
            "fedavg_stream",
            "moon",
            "fedpln",
            "feddpl",
            "fedproto",
            "fedkd",
            "fml",
            "proxyfl",
            "fedper",
            "fedprox",
            "fedsa",
            "fedlsa",
            "lgfedavg",
            "fedrep",
            "fedala",
            "fedtgp",
        ],
    )
    parser.add_argument("--test", type=int, default=0, help="Test or train")
    args, _ = parser.parse_known_args()

    # 2. Build Full Parser
    full_parser = argparse.ArgumentParser(parents=[parser])

    # Data Args
    data_group = full_parser.add_argument_group("Data & Partitioning Arguments")
    data_group.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        help="Dataset name",
        choices=["mnist", "cifar10", "cifar100", "har", "har_feat"],
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
    data_group.add_argument("--n_class", type=int, default=2, help="For Pathological")
    data_group.add_argument(
        "--test_ratio", type=float, default=0.2, help="Ratio of test data"
    )

    # Training Args
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
        "--parallel_mode",
        type=str,
        default="stream",
        choices=["sequential", "stream", "multi_stream"],
        help="Parallel mode for stream training: sequential, stream (1 per GPU), multi_stream (N per GPU)",
    )

    # Algorithm Specific Args
    try:
        algo_module = importlib.import_module(f"src.{args.algo}")
    except ModuleNotFoundError:
        raise ValueError(f"Algorithm module src.{args.algo} not found.")

    if hasattr(algo_module, "add_args"):
        algo_module.add_args(full_parser)

    args = full_parser.parse_args()
    return args, algo_module


def main():
    args, algo_module = get_args()
    server = algo_module.Server(args=args)
    prepare_data(
        dataset_name=args.dataset,
        partition_method=args.partition,
        num_clients=args.num_clients,
        alpha=args.alpha,
        n_classes=args.n_class,
        test_ratio=args.test_ratio,
    )

    # 仅输出到文件
    log_path = server.get_log_path()
    log_f = open(log_path, "w", encoding="utf-8", buffering=1)
    sys.stdout = log_f
    sys.stderr = log_f

    server.fit()
    server.save()


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    a = time.time()

    main()

    b = time.time()
    delta = datetime.timedelta(seconds=int(b - a))
    print(f"\nTotal time: {delta}\n")
