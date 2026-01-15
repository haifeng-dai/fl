import argparse
import importlib

import torch

from data_scripts import prepare_data
from src.models import SimpleCNN
from src.utils import load_data, is_pfl


def main():
    # 1. First-pass parsing for algo
    parser = argparse.ArgumentParser(description="Unified FL Framework", add_help=False)
    parser.add_argument(
        "--algo", type=str, default="fedavg", choices=["fedavg", "moon"]
    )
    args, _ = parser.parse_known_args()

    # 2. Build Full Parser
    full_parser = argparse.ArgumentParser(parents=[parser])

    # Data Args
    data_group = full_parser.add_argument_group("Data & Partitioning Arguments")
    data_group.add_argument("--dataset", type=str, default="mnist")
    data_group.add_argument(
        "--partition",
        type=str,
        default="iid",
        choices=["iid", "dirichlet", "pathological"],
    )
    data_group.add_argument("--num_clients", type=int, default=10)
    data_group.add_argument("--test_ratio", type=float, default=0.2)
    data_group.add_argument("--alpha", type=float, default=0.5, help="For Dirichlet")
    data_group.add_argument("--n_classes", type=int, default=2, help="For Pathological")

    # Training Args
    train_group = full_parser.add_argument_group("Training Arguments")
    train_group.add_argument("--rounds", type=int, default=5)
    train_group.add_argument("--epochs", type=int, default=1)
    train_group.add_argument("--lr", type=float, default=0.01)
    train_group.add_argument("--batch_size", type=int, default=64)
    train_group.add_argument("--gpus", type=str, default="0")
    train_group.add_argument("--no_mp", action="store_true", help="Disable multiprocessing training")

    # Algorithm Specific Args
    algo_module = importlib.import_module(f"src.{args.algo}")
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

    # 4. Resource Setup
    gpu_ids = [int(i) for i in args.gpus.split(",")]
    args.cuda = {
        i: torch.device(
            f"cuda:{gpu_ids[i % len(gpu_ids)]}"
            if torch.cuda.is_available()
            else "cpu"
        ) for i in range(args.num_clients)
    }

    # 5. Global Objects
    global_model = SimpleCNN()
    pfl = is_pfl(args.algo)

    # Unified data loading
    train_sets, test_set, train_counts = load_data(
        dataset_name=args.dataset,
        partition=args.partition,
        num_clients=args.num_clients,
        alpha=args.alpha,
        n_classes=args.n_classes,
        pfl=pfl
    )

    # 6. Instantiate and Run
    server = None
    if args.algo == "fedavg":
        from src.fedavg import FedAvgServer

        assert isinstance(test_set, torch.utils.data.Dataset)
        server = FedAvgServer(
            model=global_model,
            pfl=False,
            args=args
        )
    elif args.algo == "moon":
        from src.moon import MOONServer

        assert isinstance(test_set, torch.utils.data.Dataset)
        server = MOONServer(
            model=global_model,
            pfl=False,
            args=args
        )
    if server is None:
        raise ValueError(f"Unsupported algorithm: {args.algo}")

    try:
        server.fit()
    finally:
        if server is not None:
            server.close()


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
