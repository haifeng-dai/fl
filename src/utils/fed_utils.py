import argparse
import os

import torch
import torch.multiprocessing as mp
import wandb

from ..models import CNN, HARCNN, HARMLP, ResNet18, ResNet50
from .aggregate import param_aggregate
from .evaluate import evaluate_model, evaluate_prototype
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
        self.acc_proto: list[float] = []
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
        self.clients_state = [self.model.state_dict() for _ in range(self.num_clients)]

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
            print(
                f"-> Multiprocessing not enabled, using sequential training (Device: {self.client_gpu[0]})"
            )

        self.device = gpu_ids[-1]

    def aggregate(
        self, client_state_dicts, weights: list[float] | None = None, *args, **kwargs
    ):
        if weights is None:
            weights = self.weights
        aggregated_state = param_aggregate(client_state_dicts, weights)
        self.model.load_state_dict(aggregated_state)

    def evaluate(self, model_states=None, protos=None):
        """
        智能评估接口：自动在全局评估与个性化评估之间切换，
        并支持可选的原型匹配功能。

        参数说明:
            model_states: (可选) 用于进行评估的模型状态字典列表。
                          如果为 None，则默认使用 self.clients_state。
            protos: (可选) 当前用于评估的全局原型参数 (Dict 或 Tensor 格式)。
        """
        # 1. 状态保存保护：若处于 pFL (个性化联邦学习) 模式下，则备份全局模型
        if self.pfl:
            global_backup = {
                k: v.cpu().clone() for k, v in self.model.state_dict().items()
            }

        # 2. 基于普通模型的指标评估
        if not self.pfl:
            # 模式 A: 传统/全局联邦学习
            acc = evaluate_model(self.model, self.test_set, self.device)
            self.acc.append(acc)
        else:
            # 模式 B: 个性化联邦学习 (pFL)
            accs = []
            self.model.to(self.device)

            # 使用提供传入的状态列表或回退使用实例自身的 clients_state
            target_states = (
                model_states if model_states is not None else self.clients_state
            )

            assert target_states, (
                "Personalized algorithms (pfl=True) must provide model_states or maintain self.clients_state."
            )

            for i in range(self.num_clients):
                self.model.load_state_dict(target_states[i])
                accs.append(evaluate_model(self.model, self.test_set[i], self.device))
            self.acc.append(sum(accs) / len(accs) if accs else 0.0)

        # 3. 基于原型的指标评估（可选调用）
        if protos is not None:
            # 将所传原型参数标准化为 Tensor 形态 [C, d]
            if isinstance(protos, dict):
                proto_tensor = torch.zeros(
                    self.num_class, self.args.feature_dim, device=self.device
                )
                for k, v in protos.items():
                    proto_tensor[k] = v.to(self.device)
            else:
                proto_tensor = protos.to(self.device)

            if not self.pfl:
                p_acc = evaluate_prototype(
                    self.model, proto_tensor, self.test_set, self.device
                )
            else:
                p_accs = []
                # 为保持一致性，使用相同的 target_states 来重新评估对应的原型参数
                target_states = (
                    model_states if model_states is not None else self.clients_state
                )
                for i in range(self.num_clients):
                    self.model.load_state_dict(target_states[i])
                    p_accs.append(
                        evaluate_prototype(
                            self.model, proto_tensor, self.test_set[i], self.device
                        )
                    )
                p_acc = sum(p_accs) / len(p_accs) if p_accs else 0.0
            self.acc_proto.append(p_acc)

        # 4. 模型状态恢复复原
        if self.pfl:
            self.model.load_state_dict(global_backup)

        self.model.cpu()

    def run_clients(self, client_worker, parameters):
        """运行并行或顺序客户端训练。"""
        if not self.mp:
            res = {p[0]: client_worker(p) for p in parameters}
        else:
            async_results = {
                p[0]: self.gpu_pools[p[1]].apply_async(client_worker, (p,))
                for p in parameters
            }
            res = {i: r.get() for i, r in async_results.items()}
        return res

    def fit(self, *args, **kwargs):
        raise NotImplementedError

    def close(self):
        """显式关闭并行池，释放 GPU 资源"""
        if hasattr(self, "gpu_pools"):
            for device, pool in self.gpu_pools.items():
                print(f"-> Closing parallel pool on device {device}...")
                pool.terminate()
                pool.join()
            # 防止重复关闭
            self.gpu_pools = {}

    def log_dict(self, round_idx, metrics: dict = None):
        """通用 WandB 日志记录接口"""
        if wandb.run is not None:
            # 基础指标汇总
            log_data = {
                "test/acc": self.acc[-1] if self.acc else 0.0,
                "train/loss": self.loss[-1] if self.loss else 0.0,
            }
            if self.acc_proto:
                log_data["test/proto_acc"] = self.acc_proto[-1]

            # 合并额外指标
            if metrics:
                log_data.update(metrics)

            wandb.log(log_data, step=round_idx)

    def deal_save(self, params):
        path = os.path.join(
            self.args.save_path, f"{self.args.file_name}_{self.args.times}.pt"
        )
        if self.args.test:
            print(f"\nnot save to {path}\n")
        else:
            print(f"\nsaved to {path}\n")
            torch.save(params, path)
        self.close()


def get_model(model_name, dataset, feature_dim):
    if dataset == "cifar100":
        num_classes = 100
    elif dataset == "flowers102":
        num_classes = 102
    else:
        num_classes = 10
    if model_name == "resnet18":
        global_model = ResNet18(
            num_classes=num_classes, dataset_name=dataset, feature_dim=feature_dim
        )
    elif model_name == "resnet50":
        global_model = ResNet50(
            num_classes=num_classes, dataset_name=dataset, feature_dim=feature_dim
        )
    elif model_name == "harcnn":
        global_model = HARCNN(in_channels=9, num_classes=6, feature_dim=feature_dim)
    elif model_name == "harmlp":
        global_model = HARMLP(input_dim=561, num_classes=6, feature_dim=feature_dim)
    elif model_name == "cnn":
        # 根据数据集选择输入通道数
        input_channels = 1 if dataset == "mnist" else 3
        global_model = CNN(
            input_channels=input_channels,
            num_classes=num_classes,
            feature_dim=feature_dim,
        )
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    return global_model
