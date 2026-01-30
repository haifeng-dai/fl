import argparse
import os
import torch
import torch.multiprocessing as mp

from ..models import CNN, ResNet18, ResNet50, HARCNN, HARMLP
from .aggregate import param_aggregate
from .evaluate import evaluate_model
from .load_data import load_data


class BaseServer:
    def __init__(self, pfl: bool, args: argparse.Namespace):
        self.model = get_model(args.model, args.dataset, args.feature_dim).cpu()
        self.args = args
        self.rounds: int = args.rounds
        self.mp: bool = bool(self.args.mp)

        self.num_clients: int = self.args.num_clients
        self.pfl = pfl
        self.acc: list[float] = []
        self.loss: list[float] = []

        self.train_sets, self.test_set, train_counts, self.num_class = load_data(
            dataset_name=args.dataset,
            partition=args.partition,
            num_clients=args.num_clients,
            alpha=args.alpha,
            n_classes=args.n_class,
            pfl=self.pfl,
        )

        total_samples = sum(train_counts.values())
        self.weights = [
            train_counts[i] / total_samples for i in range(len(train_counts))
        ]

        self._BaseServer__start_pools(args.gpus)

    def _BaseServer__start_pools(self, gpus):
        gpu_ids = [int(i) for i in gpus.split(",")]

        # 根据是否启用并行模式来分配GPU
        self.gpu_pools = {}
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
            device_counts = dict.fromkeys(set(self.client_gpu.values()), 0)
            for device in self.client_gpu.values():
                device_counts[device] += 1

            # 获取最大worker数限制（如果设置了的话）
            max_workers = getattr(self.args, "max_workers_per_gpu", None)

            for device, count in device_counts.items():
                # 限制每个GPU的最大并行worker数，避免OOM
                actual_workers = min(count, max_workers) if max_workers else count
                self.gpu_pools[device] = mp.Pool(processes=actual_workers)
        else:
            # 非并行模式：所有客户端都使用第一个GPU
            first_gpu = torch.device(
                f"cuda:{gpu_ids[0]}" if torch.cuda.is_available() else "cpu"
            )
            self.client_gpu = {i: first_gpu for i in range(self.num_clients)}
            print(f"-> 未启用多进程训练，将使用顺序训练 (设备: {self.client_gpu[0]})")

        self.device = gpu_ids[-1]

    def aggregate(
        self, client_state_dicts, weights: list[float] | None = None, *args, **kwargs
    ):
        if weights is None:
            weights = self.weights
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

    def deal_save(self, params):
        path = os.path.join(self.args.save_path, f"{self.args.file_name}.pt")
        if self.args.test:
            print(f"\nnot save to {path}\n")
        else:
            print(f"\nsaved to {path}\n")
            torch.save(params, path)
        self.close()


def get_model(model_name, dataset, feature_dim):
    num_classes = 100 if dataset == "cifar100" else 10
    if model_name == "resnet18":
        global_model = ResNet18(num_classes=num_classes, dataset_name=dataset, feature_dim=feature_dim)
    elif model_name == "resnet50":
        global_model = ResNet50(num_classes=num_classes, dataset_name=dataset, feature_dim=feature_dim)
    elif model_name == "harcnn":
        global_model = HARCNN(in_channels=9, num_classes=6, feature_dim=feature_dim)
    elif model_name == "harmlp":
        global_model = HARMLP(input_dim=561, num_classes=6, feature_dim=feature_dim)
    elif model_name == "cnn":
        # 根据数据集选择输入通道数
        input_channels = 1 if dataset == "mnist" else 3
        global_model = CNN(input_channels=input_channels, num_classes=num_classes, feature_dim=feature_dim)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    return global_model
