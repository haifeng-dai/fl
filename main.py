import argparse
import importlib
import time

import torch

from src import CNN, ResNet18, prepare_data


def main():
    # 1. First-pass parsing for algo
    parser = argparse.ArgumentParser(description="Unified FL Framework", add_help=False)
    parser.add_argument(
        "--algo",
        type=str,
        default="fedavg",
        choices=["fedavg", "fedavg_stream", "moon", "fedpln", "feddpl", "fedproto"],
    )
    parser.add_argument("--test", type=bool, default=True, help="Test or train")
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
        choices=["mnist", "cifar10"],
    )
    data_group.add_argument(
        "--model",
        type=str,
        default="cnn",
        help="Model architecture",
        choices=["cnn", "resnet18"],
    )
    data_group.add_argument(
        "--partition",
        type=str,
        default="iid",
        choices=["iid", "dirichlet", "pathological"],
        help="Data partitioning strategy",
    )
    data_group.add_argument(
        "--num_clients", type=int, default=10, help="Number of clients"
    )
    data_group.add_argument(
        "--test_ratio", type=float, default=0.2, help="Ratio of test data"
    )
    data_group.add_argument("--alpha", type=float, default=0.5, help="For Dirichlet")
    data_group.add_argument("--n_classes", type=int, default=2, help="For Pathological")

    # Training Args
    train_group = full_parser.add_argument_group("Training Arguments")
    train_group.add_argument(
        "--rounds", type=int, default=5, help="Number of communication rounds"
    )
    train_group.add_argument(
        "--epochs", type=int, default=1, help="Number of local epochs"
    )
    train_group.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    train_group.add_argument("--batch_size", type=int, default=64, help="Batch size")
    train_group.add_argument(
        "--gpus", type=str, default="0", help="Comma separated list of GPU ids"
    )
    train_group.add_argument(
        "--no_mp", action="store_true", help="Disable multiprocessing training"
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

    # 3. Automatic Data Preparation
    prepare_data(
        dataset_name=args.dataset,
        partition_method=args.partition,
        num_clients=args.num_clients,
        alpha=args.alpha,
        n_classes=args.n_classes,
        test_ratio=args.test_ratio,
    )

    # 4. Instantiate and Run
    server = algo_module.Server(args=args)
    server.fit()
    server.save(args.test)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    a = time.time()
    main()

    b = time.time()
    print(f"Total time: {b - a} seconds")
