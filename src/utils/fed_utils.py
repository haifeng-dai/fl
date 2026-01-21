import copy
import os
import argparse

import torch
import torch.multiprocessing as mp
from .aggregate import param_aggregate
from .load_data import load_data
from .evaluate import evaluate_model


class BaseServer:
    def __init__(self, model: torch.nn.Module, pfl: bool, args: argparse.Namespace):
        self.model = copy.deepcopy(model).cpu()
        self.args = args
        self.rounds: int = args.rounds
        self.mp: bool = bool(self.args.mp)

        self.num_clients = self.args.num_clients
        self.device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        # self.clients: dict[int, BaseClient] = {}
        self.pfl = pfl
        self.acc: list[float] = []
        self.loss: list[float] = []

        self.train_sets, self.test_set, train_counts, self.num_class = load_data(
            dataset_name=args.dataset,
            partition=args.partition,
            num_clients=args.num_clients,
            alpha=args.alpha,
            n_classes=args.n_classes,
            pfl=self.pfl,
        )
        self.fold_path = os.path.join(
            "results", f"{args.dataset}_{args.partition}_{args.num_clients}"
        )
        if args.partition == "dirichlet":
            self.fold_path += f"_{args.alpha}"
        elif args.partition == "pathological":
            self.fold_path += f"_{args.n_classes}"
        os.makedirs(self.fold_path, exist_ok=True)

        # Use pre-calculated counts for sample weights
        total_samples = sum(train_counts.values())
        self.weights = [
            train_counts[i] / total_samples for i in range(len(train_counts))
        ]

        self._BaseServer__start_pools(args.gpus)

    def _BaseServer__start_pools(self, gpus):
        gpu_ids = [int(i) for i in gpus.split(",")]

        # 根据是否启用并行模式来分配GPU
        if self.mp:
            # 并行模式：将客户端循环分配到多个GPU
            self.client_gpu = {
                i: torch.device(
                    f"cuda:{gpu_ids[i % len(gpu_ids)]}"
                    if torch.cuda.is_available()
                    else "cpu"
                )
                for i in range(self.num_clients)
            }
        else:
            # 非并行模式：所有客户端都使用第一个GPU
            first_gpu = torch.device(
                f"cuda:{gpu_ids[0]}" if torch.cuda.is_available() else "cpu"
            )
            self.client_gpu = {i: first_gpu for i in range(self.num_clients)}

        self.gpu_pools = {}
        if self.mp:
            device_counts = dict.fromkeys(set(self.client_gpu.values()), 0)
            for device in self.client_gpu.values():
                device_counts[device] += 1

            # 获取最大worker数限制（如果设置了的话）
            max_workers = getattr(self.args, 'max_workers_per_gpu', None)

            for device, count in device_counts.items():
                # 限制每个GPU的最大并行worker数，避免OOM
                actual_workers = min(count, max_workers) if max_workers else count
                print(f"-> 为设备 {device} 分配并行池 (Worker: {actual_workers}/{count})")
                self.gpu_pools[device] = mp.Pool(processes=actual_workers)
        else:
            print(f"-> 未启用多进程训练，将使用顺序训练 (设备: {self.client_gpu[0]})")

    def aggregate(
        self, client_state_dicts, weights: list[float] | None = None, *args, **kwargs
    ):
        aggregated_state = param_aggregate(client_state_dicts, weights)
        self.model.load_state_dict(aggregated_state)

    def evaluate(self, *args, **kwargs):
        acc = evaluate_model(self.model, self.test_set, self.device)
        self.acc.append(acc)

    def fit(self, *args, **kwargs):
        raise NotImplementedError

    def close(self):
        """显式关闭并行池，释放 GPU 资源"""
        if hasattr(self, "gpu_pools"):
            for device, pool in self.gpu_pools.items():
                print(f"-> 正在关闭设备 {device} 的并行池...")
                pool.close()
                pool.join()
            # 防止重复关闭
            self.gpu_pools = {}

    def __del__(self):
        self.close()

    def deal_save(self, test, params, file_name: str | None = None):
        new_name = f"{self.args.epochs}_{self.args.batch_size}_{self.args.lr}"
        if file_name:
            new_name += f"_{file_name}"
        path = os.path.join(self.fold_path, f"{new_name}.pt")
        if test:
            print(f"not save to {path}")
        else:
            print(f"saved to {path}")
            torch.save(params, path)
