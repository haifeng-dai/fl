import copy
from collections import defaultdict
import argparse

import torch
import torch.multiprocessing as mp
from dataclasses import dataclass

from .aggregate import param_aggregate
from .evaluate import evaluate_model


class BaseClient:
    def __init__(
            self,
            client_id: int,
            model: torch.nn.Module,
            train_loader: torch.utils.data.DataLoader,
            lr: float,
            epochs: int,
            device: torch.device,
    ):
        self.client_id = client_id
        self.model = copy.deepcopy(model).to(device)
        self.train_loader = train_loader
        self.lr = lr
        self.epochs = epochs
        self.device = device

        self.ce = torch.nn.CrossEntropyLoss()
        self.mse = torch.nn.MSELoss()
        self.KL = torch.nn.KLDivLoss(reduction="batchmean")

    def train(self, *args, **kwargs):
        raise NotImplementedError

    def set_client(self, *args, **kwargs):
        raise NotImplementedError


class BaseServer:
    def __init__(
            self,
            model: torch.nn.Module,
            test_loader: torch.utils.data.DataLoader,
            clients_info: ClientInfo,
            rounds: int,
    ):
        self.model = model.cpu()
        self.test_loader = test_loader
        self.clients_info = clients_info
        self.rounds = rounds

        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        device_counts = defaultdict(int)
        for _, device in self.clients_info.cuda.items():
            device_counts[device] += 1

        self.gpu_pools = {}
        for device, count in device_counts.items():
            print(f"-> 为设备 {device} 分配并行池 (Worker: {count})")
            self.gpu_pools[device] = mp.Pool(processes=count)

    def aggregate(self, client_state_dicts, weights: list[float] | None = None, *args, **kwargs):
        aggregated_state = param_aggregate(client_state_dicts, weights)
        self.model.load_state_dict(aggregated_state)

    def evaluate(self, pfl=False, *args, **kwargs):
        if isinstance(self.test_loader, dict):
            #是个性化联邦学习或使用了本地测试集，计算平均准确率
            accs = []
            for client_id, loader in self.test_loader.items():
                acc = evaluate_model(self.model, loader, self.device)
                accs.append(acc)
            return sum(accs) / len(accs)
        else:
            acc = evaluate_model(self.model, self.test_loader, self.device)
            return acc

    def fit(self, *args, **kwargs):
        raise NotImplementedError

    def __del__(self, *args, **kwargs):
        if hasattr(self, "gpu_pools"):
            for pool in self.gpu_pools.values():
                pool.close()
                pool.join()


@dataclass
class ClientInfo:
    args: argparse.Namespace

    def __post_init__(self):
        self.lr: float = self.args.lr
        self.epochs: int = self.args.epochs
        gpu_ids = [int(i) for i in self.args.gpus.split(",")]
        self.cuda = {
            i: torch.device(
                f"cuda:{gpu_ids[i % len(gpu_ids)]}"
                if torch.cuda.is_available()
                else "cpu"
            ) for i in range(self.args.num_clients)
        }
