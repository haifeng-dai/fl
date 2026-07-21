import os

import ray
import torch

from ...models import CNN, HARCNN, HARMLP, ResNet18, ResNet50
from .aggregate import param_aggregate
from .evaluate import evaluate_model, evaluate_prototype
from .load_data import load_data


@ray.remote
def train(worker_func, params):
    """
    Ray 远程工作者的通用包装函数。
    """
    p_list = list(params)
    # 在 Ray 托管的环境中，CUDA_VISIBLE_DEVICES 会被自动设置
    if torch.cuda.is_available():
        p_list[1] = torch.device("cuda:0")
    else:
        p_list[1] = torch.device("cpu")

    p_list[3] = ray.get(p_list[3])

    return worker_func(tuple(p_list))


@ray.remote
def evaluate(
    model_name, dataset_name, feature_dim, state_dict, test_set, device, prototype=None
):
    """Ray Worker: 并行评估单个客户端的模型准确率与原型准确率。"""
    model = get_model(model_name, dataset_name, feature_dim).to(device)
    model.load_state_dict(state_dict)
    acc = evaluate_model(model, test_set, device)
    p_acc = 0.0
    if prototype is not None:
        p_acc = evaluate_prototype(model, prototype.to(device), test_set, device)
    return {"acc": acc, "p_acc": p_acc}


class BaseServer:
    def __init__(self, pfl: bool, args):
        self.model = get_model(args.model, args.dataset, args.feature_dim).cpu()
        self.args = args
        self.rounds: int = args.rounds

        # 自适应 Round 调整逻辑
        if self.rounds == 0:
            self.rounds = 500 if pfl else 1000
            args.rounds = self.rounds
            print(
                f"-> Adaptive Rounds: detected {'PFL' if pfl else 'GFL'} algorithm, setting rounds={self.rounds}"
            )

        self.num_clients: int = self.args.num_clients
        self.pfl = pfl
        self.acc: list[float] = []
        self.acc_proto: list[float] = []
        self.acc_source: list[float] = []
        self.acc_target: list[float] = []
        self.loss: list[float] = []

        # 领域数据分支
        self.domain_dataset = getattr(args, "domain_dataset", None)
        self.effective_dataset = (
            self.domain_dataset if self.domain_dataset is not None else args.dataset
        )
        self.source_test = None
        self.target_test = None

        if self.domain_dataset is not None:
            (
                self.train_sets,
                self.test_set,
                train_counts,
                self.num_class,
                self.source_test,
                self.target_test,
                self.domain_labels,
            ) = load_data(
                dataset_name=self.domain_dataset,
                domain_partition=args.domain_partition,
                num_clients=args.num_clients,
                alpha=args.alpha,
                pfl=self.pfl,
                domain_aware=getattr(args, "domain_aware", True),
            )
        else:
            (
                self.train_sets,
                self.test_set,
                train_counts,
                self.num_class,
                _,
                _,
                _,
            ) = load_data(
                dataset_name=args.dataset,
                partition=args.partition,
                num_clients=args.num_clients,
                alpha=args.alpha,
                n_classes=args.n_class,
                pfl=self.pfl,
            )
        self.train_set_refs = [ray.put(ds) for ds in self.train_sets.values()]

        total_samples = sum(train_counts.values())
        self.weights = [
            train_counts[i] / total_samples for i in range(len(train_counts))
        ]
        self.clients_state = [self.model.state_dict() for _ in range(self.num_clients)]

        # 1. 解析 GPU 资源
        if isinstance(args.gpus, int):
            gpu_ids = [args.gpus]
        elif isinstance(args.gpus, str) and args.gpus.strip():
            gpu_ids = [int(i) for i in args.gpus.split(",") if i.strip()]
        else:
            gpu_ids = [0]
        # 由于 CUDA_VISIBLE_DEVICES 会将指定 GPU 编号映射为连续的 0 到 N-1，
        # 故 Server 使用的 GPU 设备索引应为本地可见的最后一个，即 len(gpu_ids) - 1
        dev_idx = len(gpu_ids) - 1
        self.device = torch.device(
            f"cuda:{dev_idx}" if torch.cuda.is_available() and dev_idx >= 0 else "cpu"
        )

        # 2. 强制设备映射：在 Ray Worker 环境中逻辑显卡始终映射为 cuda:0
        dev_str = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.client_gpu = {i: torch.device(dev_str) for i in range(self.num_clients)}

        # 3. 缓存测试集到 Ray Object Store，供并行评估使用
        if pfl:
            self.test_set_refs = [
                ray.put(self.test_set[i]) for i in range(self.num_clients)
            ]
        else:
            global_test_ref = ray.put(self.test_set)
            self.test_set_refs = [global_test_ref for _ in range(self.num_clients)]

        # 4. 领域模式缓存源域/目标域测试集
        if self.domain_dataset is not None:
            self.source_test_ref = (
                ray.put(self.source_test) if self.source_test is not None else None
            )
            self.target_test_ref = (
                ray.put(self.target_test) if self.target_test is not None else None
            )

    def aggregate(
        self, client_state_dicts, weights: list[float] | None = None, *args, **kwargs
    ):
        if weights is None:
            weights = self.weights
        aggregated_state = param_aggregate(client_state_dicts, weights)
        self.model.load_state_dict(aggregated_state)

    def evaluate(self, model_states=None, protos=None):
        if self.domain_dataset is not None:
            # 领域模式：分别评估源域和目标域准确率
            if self.source_test is not None:
                acc_src = evaluate_model(self.model, self.source_test, self.device)
                self.acc_source.append(acc_src)
                self.acc.append(acc_src)
            if self.target_test is not None:
                acc_tgt = evaluate_model(self.model, self.target_test, self.device)
                self.acc_target.append(acc_tgt)
            if protos is not None:
                if self.source_test is not None:
                    p_acc = evaluate_prototype(
                        self.model,
                        protos.to(self.device),
                        self.source_test,
                        self.device,
                    )
                    self.acc_proto.append(p_acc)
            return

        if not self.pfl:
            acc = evaluate_model(self.model, self.test_set, self.device)
            self.acc.append(acc)
            if protos is not None:
                p_acc = evaluate_prototype(
                    self.model, protos.to(self.device), self.test_set, self.device
                )
                self.acc_proto.append(p_acc)
            return

        # pfl=True: Ray 并行评估
        target_states = model_states if model_states is not None else self.clients_state
        ray_gpu_fraction = 1.0 / max(1, self.args.max_workers_per_gpu)
        client_proto = protos.cpu() if protos is not None else None
        futures = []
        for i in range(self.num_clients):
            futures.append(
                evaluate.options(
                    num_gpus=ray_gpu_fraction,
                    scheduling_strategy="SPREAD",
                ).remote(
                    self.args.model,
                    self.args.dataset,
                    self.args.feature_dim,
                    target_states[i],
                    self.test_set_refs[i],
                    self.client_gpu[i],
                    client_proto,
                )
            )
        results = ray.get(futures)
        self.acc.append(sum(r["acc"] for r in results) / self.num_clients)
        if protos is not None:
            self.acc_proto.append(sum(r["p_acc"] for r in results) / self.num_clients)

    def build_base_params(self, selected_clients):
        return [
            [
                i,
                self.client_gpu[i],
                self.model.state_dict(),
                self.train_sets[i],
                self.args.model,
                self.effective_dataset,
                self.args.lr,
                self.args.batch_size,
                self.args.epochs,
                self.args.feature_dim,
            ]
            for i in selected_clients
        ]

    def run_clients(self, worker_func, parameters):
        """强制通过 Ray 运行客户端训练。"""
        # 根据 max_workers_per_gpu 计算 Ray 需要的显存比例 (1/n)
        ray_gpu_fraction = 1.0 / max(1, self.args.max_workers_per_gpu)

        optimized_parameters = []
        for p in parameters:
            p_list = list(p)
            cid = p_list[0]
            p_list[3] = self.train_set_refs[cid]
            optimized_parameters.append(tuple(p_list))

        remote_worker = train.options(
            num_gpus=ray_gpu_fraction, scheduling_strategy="SPREAD"
        )
        futures = [remote_worker.remote(worker_func, p) for p in optimized_parameters]
        results_list = ray.get(futures)

        results_map = {
            parameters[i][0]: results_list[i] for i in range(len(parameters))
        }
        return results_map

    def deal_save(self, metrics, params):
        max_a = max(self.acc) if self.acc else 0.0
        summary_str = f"\n[Summary] Max Acc: {max_a:.2f}%"
        if self.acc_target:
            max_ta = max(self.acc_target)
            summary_str += f" | Max Target Acc: {max_ta:.2f}%"
        if self.acc_proto:
            max_pa = max(self.acc_proto)
            summary_str += f" | Max Proto Acc: {max_pa:.2f}%"
        print(summary_str)

        name = f"{self.args.file_name}_{self.args.cur_time}.pt"
        metrics_path = os.path.join(self.args.save_path, name)
        params_path = os.path.join(
            self.args.save_path, name.replace(".pt", "_params.pt")
        )
        if getattr(self.args, "test", False):
            print(f"\n-> [Test Mode] Would save metrics to: {metrics_path}")
            print(f"\n-> [Test Mode] Would save params to: {params_path}")
            return
        os.makedirs(self.args.save_path, exist_ok=True)
        torch.save(metrics, metrics_path)
        torch.save(params, params_path)
        print(f"-> Metrics saved to: {metrics_path}")
        print(f"-> Params saved to: {params_path}")


def get_model(model_name, dataset_name, feature_dim=512):
    """
    模型工厂函数。
    """
    if dataset_name == "mnist":
        n_class = 10
    elif dataset_name == "cifar10":
        n_class = 10
    elif dataset_name == "cinic10":
        n_class = 10
    elif dataset_name == "cifar100":
        n_class = 100
    elif dataset_name == "flowers102":
        n_class = 102
    elif dataset_name == "tiny_imagenet":
        n_class = 200
    elif dataset_name == "svhn":
        n_class = 10
    elif dataset_name == "femnist":
        n_class = 62
    elif dataset_name == "emnist":
        n_class = 47
    elif dataset_name == "har" or dataset_name == "har_feat":
        n_class = 6
    elif dataset_name == "pacs":
        n_class = 7
    elif dataset_name == "officehome":
        n_class = 65
    elif dataset_name == "vlcs":
        n_class = 5
    elif dataset_name == "domainnet":
        n_class = 345
    elif dataset_name == "cifar10_dg":
        n_class = 10
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    input_channels = (
        3
        if (
            "cifar" in dataset_name
            or dataset_name
            in [
                "tiny_imagenet",
                "flowers102",
                "cars",
                "gtsrb",
                "cinic10",
                "svhn",
                "pacs",
                "officehome",
                "vlcs",
                "domainnet",
            ]
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
