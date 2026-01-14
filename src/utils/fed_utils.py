import copy
from collections import defaultdict
import argparse

import torch
import torch.multiprocessing as mp

from .aggregate import param_aggregate
from .evaluate import evaluate_model
from torch.utils.data import DataLoader


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
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.device = args.cuda[client_id]

        self.ce = torch.nn.CrossEntropyLoss()
        self.mse = torch.nn.MSELoss()
        self.KL = torch.nn.KLDivLoss(reduction="batchmean")

    def build_train_loader(self) -> DataLoader:
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

    def train(self, *args, **kwargs):
        raise NotImplementedError

    def set_client(self, *args, **kwargs):
        raise NotImplementedError


class BaseServer:
    def __init__(
            self,
            model: torch.nn.Module,
            test_set: torch.utils.data.Dataset | dict[int, torch.utils.data.Dataset] | None,
            train_counts: dict[int, int],
            args: argparse.Namespace
    ):
        self.model = model.cpu()
        self.test_set = test_set
        self.args = args
        self.rounds = args.rounds

        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        # Use pre-calculated counts for sample weights
        total_samples = sum(train_counts.values())
        self.weights = [train_counts[i] / total_samples for i in range(len(train_counts))]

        self.no_mp = self.args.no_mp
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

    def aggregate(self, client_state_dicts, weights: list[float] | None = None, *args, **kwargs):
        aggregated_state = param_aggregate(client_state_dicts, weights)
        self.model.load_state_dict(aggregated_state)

    def evaluate(self, pfl=False, *args, **kwargs):
        if self.test_set is None:
            return 0.0

        if isinstance(self.test_set, dict):
            # 是个性化联邦学习或使用了本地测试集，计算平均准确率
            accs = []
            for client_id, dataset in self.test_set.items():
                loader = DataLoader(dataset, batch_size=128, shuffle=False)
                acc = evaluate_model(self.model, loader, self.device)
                accs.append(acc)
            return sum(accs) / len(accs)
        else:
            loader = DataLoader(self.test_set, batch_size=128, shuffle=False)
            acc = evaluate_model(self.model, loader, self.device)
            return acc

    def fit(self, *args, **kwargs):
        raise NotImplementedError

    def __del__(self, *args, **kwargs):
        if hasattr(self, "gpu_pools"):
            for pool in self.gpu_pools.values():
                pool.close()
                pool.join()
