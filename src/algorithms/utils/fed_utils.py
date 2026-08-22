import os

import ray
import torch
from torch.utils.data import Subset

from ...models import CNN, HARCNN, HARMLP, ResNet18, ResNet50
from .aggregate import param_aggregate
from .evaluate import evaluate_model, evaluate_prototype
from .load_data import load_data


@ray.remote
def train(worker_func, params):
    """
    Ray 远程工作者的通用包装函数。

    worker 返回后统一清空显存缓存（caching allocator 预留），
    防止大张量操作（mixup / EMA / 原型等）导致 reserved 持续膨胀。
    """
    p_list = list(params)
    # 在 Ray 托管的环境中，CUDA_VISIBLE_DEVICES 会被自动设置
    if torch.cuda.is_available():
        p_list[1] = torch.device("cuda:0")
    else:
        p_list[1] = torch.device("cpu")

    p_list[3] = ray.get(p_list[3])

    try:
        return worker_func(tuple(p_list))
    finally:
        torch.cuda.empty_cache()


@ray.remote
def evaluate(
    model_name,
    dataset_name,
    feature_dim,
    state_dict,
    test_set,
    device,
    n_class,
    prototype=None,
):
    """Ray Worker: 并行评估单个客户端的模型准确率与原型准确率。"""
    model = get_model(model_name, dataset_name, n_class, feature_dim).to(device)
    model.load_state_dict(state_dict)
    acc = evaluate_model(model, test_set, device)
    p_acc = 0.0
    if prototype is not None:
        p_acc = evaluate_prototype(model, prototype.to(device), test_set, device)
    return {"acc": acc, "p_acc": p_acc}


class BaseServer:
    def __init__(self, pfl: bool, args):
        self.rounds: int = args.rounds
        self.num_clients: int = args.num_clients
        self.join_ratio: float = args.join_ratio
        self.epochs: int = args.epochs
        self.batch_size: int = args.batch_size
        self.lr: float = args.lr
        self.model_name: str = args.model
        self.dataset: str = args.dataset
        self.feature_dim: int = args.feature_dim
        self.max_workers_per_gpu: int = args.max_workers_per_gpu
        self.save_path: str = args.save_path
        self.log_path: str = args.log_path
        self.file_name: str = args.file_name
        self.cur_time: str = args.cur_time
        self.test: int = args.test
        self.pfl = pfl

        # 自适应 Round 调整逻辑
        if self.rounds == 0:
            self.rounds = 1000 if pfl else 200
            print(
                f"-> Adaptive Rounds: detected {'PFL' if pfl else 'GFL'} algorithm, setting rounds={self.rounds}"
            )

        self.acc: list[float] = []
        self.acc_proto: list[float] = []
        self.loss: list[float] = []

        self.fdg = args.fdg
        if self.fdg:
            self.selected_domains = args.selected_domains
            self.target_domain = args.target_domain

        self.ssl = args.ssl
        if self.is_sfd:
            self.label_domain = args.label_domain
            self.unlabel_domain = args.unlabel_domain
            self.acc_source: list[float] = []
            self.acc_target: list[float] = []
            self.acc_source_p: list[float] = []
            self.acc_target_p: list[float] = []

        if self.is_ssl:
            self.label_ratio = args.label_ratio
            self.lam = args.lam
            self.confidence = args.confidence

        (
            self.train_sets,
            self.test_set,
            train_counts,
            self.num_class,
        ) = load_data(args, self.pfl)
        self.train_set_refs = [ray.put(ds) for ds in self.train_sets.values()]

        total_samples = sum(train_counts.values())
        self.weights = [
            train_counts[i] / total_samples for i in range(len(train_counts))
        ]

        self.model = get_model(
            self.model_name, self.dataset, self.num_class, self.feature_dim
        ).cpu()
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

    @property
    def is_sfd(self) -> bool:
        return self.ssl == "sfd"

    @property
    def is_ssl(self) -> bool:
        return self.ssl in ("sample", "client", "sfd")

    def aggregate(
        self, client_state_dicts, weights: list[float] | None = None, *args, **kwargs
    ):
        if weights is None:
            weights = self.weights
        aggregated_state = param_aggregate(client_state_dicts, weights)
        self.model.load_state_dict(aggregated_state)

    def evaluate(self, model_states=None, protos=None):
        if self.is_sfd:
            # SFD：按 unlabel_domain 切分测试集（label 域 -> acc_source，unlabel 域 -> acc_target）
            assert isinstance(self.test_set.domains, list)
            unlabel_domain = self.unlabel_domain
            mask_tgt = torch.tensor(
                [d == unlabel_domain for d in self.test_set.domains]
            )
            tgt_idx = torch.where(mask_tgt)[0].tolist()
            src_idx = torch.where(~mask_tgt)[0].tolist()
            src_eval = Subset(self.test_set, src_idx)
            acc_src = evaluate_model(self.model, src_eval, self.device)
            self.acc_source.append(acc_src)
            tgt_eval = Subset(self.test_set, tgt_idx)
            acc_tgt = evaluate_model(self.model, tgt_eval, self.device)
            self.acc_target.append(acc_tgt)
            acc_all = evaluate_model(self.model, self.test_set, self.device)
            self.acc.append(acc_all)
            if protos is not None:
                p_acc = evaluate_prototype(
                    self.model, protos.to(self.device), src_eval, self.device
                )
                self.acc_source_p.append(p_acc)
                p_acc = evaluate_prototype(
                    self.model, protos.to(self.device), tgt_eval, self.device
                )
                self.acc_target_p.append(p_acc)
                p_acc = evaluate_prototype(
                    self.model, protos.to(self.device), self.test_set, self.device
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
        ray_gpu_fraction = 1.0 / max(1, self.max_workers_per_gpu)
        client_proto = protos.cpu() if protos is not None else None
        futures = []
        for i in range(self.num_clients):
            futures.append(
                evaluate.options(
                    num_gpus=ray_gpu_fraction,
                    scheduling_strategy="SPREAD",
                ).remote(
                    self.model_name,
                    self.dataset,
                    self.feature_dim,
                    target_states[i],
                    self.test_set_refs[i],
                    self.client_gpu[i],
                    self.num_class,
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
                self.model_name,
                self.dataset,
                self.lr,
                self.batch_size,
                self.epochs,
                self.feature_dim,
                self.num_class,
            ]
            for i in selected_clients
        ]

    def run_clients(self, worker_func, parameters):
        """通过 Ray 运行客户端训练。"""
        # 根据 max_workers_per_gpu 计算 Ray 需要的显存比例 (1/n)
        ray_gpu_fraction = 1.0 / max(1, self.max_workers_per_gpu)

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
        # acc 恒存在；proto 精度可选；source/target 仅 SFD 场景
        summary_str = f"\n[Summary] Max Acc: {max(self.acc):.2f}%"
        if self.acc_proto:
            summary_str += f" | Max Proto Acc: {max(self.acc_proto):.2f}%"
        if self.is_sfd:
            summary_str += (
                f" | Max Source Acc: {max(self.acc_source):.2f}%"
                f" | Max Target Acc: {max(self.acc_target):.2f}%"
            )
            if self.acc_source_p:
                summary_str += f" | Max Source Proto Acc: {max(self.acc_source_p):.2f}%"
            if self.acc_target_p:
                summary_str += f" | Max Target Proto Acc: {max(self.acc_target_p):.2f}%"
        print(summary_str)

        name = f"{self.file_name}_{self.cur_time}.pt"
        metrics_path = os.path.join(self.save_path, name)
        params_path = os.path.join(self.save_path, name.replace(".pt", "_params.pt"))
        if self.test:
            print(f"\n-> [Test Mode] Would save metrics to: {metrics_path}")
            print(f"\n-> [Test Mode] Would save params to: {params_path}")
            return
        os.makedirs(self.save_path, exist_ok=True)
        torch.save(metrics, metrics_path)
        torch.save(params, params_path)
        print(f"-> Metrics saved to: {metrics_path}")
        print(f"-> Params saved to: {params_path}")


def get_model(model_name, dataset_name, n_class, feature_dim):
    """
    模型工厂函数。
    """
    sets = [
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
    input_channels = 3 if ("cifar" in dataset_name or dataset_name in sets) else 1
    if model_name == "cnn":
        return CNN(input_channels, n_class, feature_dim, dataset_name)
    elif model_name == "resnet18":
        return ResNet18(n_class, feature_dim, dataset_name)
    elif model_name == "resnet50":
        return ResNet50(n_class, feature_dim, dataset_name)
    elif model_name == "harcnn":
        # HARCNN(in_channels, num_classes, feature_dim)：HAR 传感器数据为 9 通道
        return HARCNN(in_channels=9, num_classes=n_class, feature_dim=feature_dim)
    elif model_name == "harmlp":
        # HARMLP(input_dim, num_classes, feature_dim)：har_feat 特征维度为 561
        return HARMLP(input_dim=561, num_classes=n_class, feature_dim=feature_dim)
    else:
        raise ValueError(f"Unknown model: {model_name}")
