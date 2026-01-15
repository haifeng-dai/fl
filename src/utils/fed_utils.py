import copy
from collections import defaultdict
import argparse

import torch
import torch.multiprocessing as mp

from .aggregate import param_aggregate
from .evaluate import evaluate_model
from torch.utils.data import DataLoader

from .load_data import load_data


class BaseClient:
    def __init__(
            self,
            client_id: int,
            model: torch.nn.Module,
            train_set: torch.utils.data.Dataset,
            args: argparse.Namespace
    ):
        self.client_id = client_id
        self.model = copy.deepcopy(model).to(args.cuda[client_id])
        self.train_set = train_set
        self.lr: float = args.lr
        self.batch_size: int = args.batch_size
        self.epochs: int = args.epochs
        self.device: torch.device = args.cuda[client_id]

        self.ce = torch.nn.CrossEntropyLoss()
        self.mse = torch.nn.MSELoss()
        self.KL = torch.nn.KLDivLoss(reduction="batchmean")

    def build_train_loader(self) -> DataLoader:
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

    def train(self, *args, **kwargs):
        raise NotImplementedError

    def set_client(self, *args, **kwargs):
        raise NotImplementedError

    def evaluate(self, test_set, *args, **kwargs):
        acc = evaluate_model(self.model, test_set, self.device)
        return acc


class BaseServer:
    def __init__(
            self,
            model: torch.nn.Module,
            pfl: bool,
            args: argparse.Namespace
    ):
        self.model = model.cpu()
        self.args = args
        self.rounds: int = args.rounds
        self.no_mp: bool = self.args.no_mp

        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.clients = dict[int, BaseClient]()
        self.pfl = pfl

        self.gpu_pools = {}
        if not self.no_mp:
            device_counts = defaultdict(int)
            for _, device in self.args.cuda.items():
                device_counts[device] += 1

            for device, count in device_counts.items():
                print(f"-> 为设备 {device} 分配并行池 (Worker: {count})")
                self.gpu_pools[device] = mp.Pool(processes=count)
        else:
            print("-> 禁用多进程训练，将使用顺序训练")

        self.train_sets, self.test_set, train_counts = load_data(
            dataset_name=args.dataset,
            partition=args.partition,
            num_clients=args.num_clients,
            alpha=args.alpha,
            n_classes=args.n_classes,
            pfl=self.pfl
        )

        # Use pre-calculated counts for sample weights
        total_samples = sum(train_counts.values())
        self.weights = [train_counts[i] / total_samples for i in range(len(train_counts))]

    def aggregate(self, client_state_dicts, weights: list[float] | None = None, *args, **kwargs):
        aggregated_state = param_aggregate(client_state_dicts, weights)
        self.model.load_state_dict(aggregated_state)

    def evaluate(self, *args, **kwargs):
        if self.pfl:
            # 是个性化联邦学习或使用了本地测试集，计算平均准确率
            accs = []
            for client_id in self.clients:
                acc_ = self.clients[client_id].evaluate(self.test_set[client_id])
                accs.append(acc_)
            acc = sum(accs) / len(accs)
        else:
            acc = evaluate_model(self.model, self.test_set, self.device)
        return acc

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
