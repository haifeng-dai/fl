import os

import ray
import torch

from ..models import CNN, HARCNN, HARMLP, ResNet18, ResNet50
from .aggregate import param_aggregate
from .evaluate import evaluate_model, evaluate_prototype
from .load_data import load_data


@ray.remote
def worker(worker_func, params):
    """
    Ray 远程工作者的通用包装函数。
    """
    p_list = list(params)
    # 在 Ray 托管的环境中，CUDA_VISIBLE_DEVICES 会被自动设置
    if torch.cuda.is_available():
        p_list[1] = torch.device("cuda:0")
    else:
        p_list[1] = torch.device("cpu")

    return worker_func(tuple(p_list))


class BaseServer:
    def __init__(self, pfl: bool, args):
        self.model = get_model(args.model, args.dataset, args.feature_dim).cpu()
        self.args = args
        self.rounds: int = args.rounds

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

        # 1. 解析 GPU 资源
        gpu_ids = [int(i) for i in args.gpus.split(",")]
        self.device = gpu_ids[-1]  # 用于 Driver 进程评估

        # 2. 强制设备映射：在 Ray Worker 环境中逻辑显卡始终映射为 cuda:0
        dev_str = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.client_gpu = {i: torch.device(dev_str) for i in range(self.num_clients)}

    def aggregate(
        self, client_state_dicts, weights: list[float] | None = None, *args, **kwargs
    ):
        if weights is None:
            weights = self.weights
        aggregated_state = param_aggregate(client_state_dicts, weights)
        self.model.load_state_dict(aggregated_state)

    def evaluate(self, model_states=None, protos=None):
        """
        智能评估接口：自动在全局评估与个性化评估之间切换。
        """
        if self.pfl:
            global_backup = {
                k: v.cpu().clone() for k, v in self.model.state_dict().items()
            }

        if not self.pfl:
            acc = evaluate_model(self.model, self.test_set, self.device)
            self.acc.append(acc)
        else:
            accs = []
            self.model.to(self.device)
            target_states = (
                model_states if model_states is not None else self.clients_state
            )
            for i in range(self.num_clients):
                self.model.load_state_dict(target_states[i])
                accs.append(evaluate_model(self.model, self.test_set[i], self.device))
            self.acc.append(sum(accs) / len(accs) if accs else 0.0)

        if protos is not None:
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

        if self.pfl:
            self.model.load_state_dict(global_backup)

        self.model.cpu()

    def run_clients(self, client_worker, parameters):
        """强制通过 Ray 运行客户端训练。"""
        # 根据 max_workers_per_gpu 计算 Ray 需要的显存比例 (1/n)
        ray_gpu_fraction = 1.0 / max(1, self.args.max_workers_per_gpu)

        remote_worker = worker.options(num_gpus=ray_gpu_fraction)
        futures = [remote_worker.remote(client_worker, p) for p in parameters]
        results_list = ray.get(futures)
        return {parameters[i][0]: results_list[i] for i in range(len(parameters))}

    def deal_save(self, f):
        """将实验结果字典持久化到磁盘"""
        os.makedirs(self.args.save_path, exist_ok=True)
        save_name = f"{self.args.name_pre}_{self.args.times}.pt"
        save_full_path = os.path.join(self.args.save_path, save_name)
        torch.save(f, save_full_path)
        print(f"\n-> Results saved to: {save_full_path}")


def get_model(model_name, dataset_name, feature_dim=512):
    """
    模型工厂函数。
    """
    if dataset_name == "mnist":
        n_class = 10
    elif dataset_name == "cifar10":
        n_class = 10
    elif dataset_name == "cifar100":
        n_class = 100
    elif dataset_name == "flowers102":
        n_class = 102
    elif dataset_name == "tiny_imagenet":
        n_class = 200
    elif dataset_name == "har" or dataset_name == "har_feat":
        n_class = 6
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    input_channels = (
        3
        if (
            "cifar" in dataset_name
            or dataset_name in ["tiny_imagenet", "flowers102", "cars", "gtsrb"]
        )
        else 1
    )
    if model_name == "cnn":
        return CNN(input_channels, n_class, feature_dim, dataset_name)
    elif model_name == "resnet18":
        return ResNet18(n_class, feature_dim, dataset_name)
    elif model_name == "resnet50":
        return ResNet50(n_class, feature_dim, dataset_name)
    elif model_name == "harcnn":
        return HARCNN(n_class, feature_dim)
    elif model_name == "harmlp":
        return HARMLP(n_class, feature_dim)
    else:
        raise ValueError(f"Unknown model: {model_name}")
