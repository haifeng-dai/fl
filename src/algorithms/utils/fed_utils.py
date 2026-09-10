import dataclasses
import os
from typing import Any

import ray
import torch
from torch.utils.data import Dataset, Subset

from src.models import CNN, HARCNN, HARMLP, ResNet18, ResNet50

from .aggregate import param_aggregate
from .evaluate import evaluate_model, evaluate_prototype
from .input import normalize_dataset
from .load_data import load_data


@dataclasses.dataclass
class BaseParams:
    client_id: int
    client_gpu: str
    model_state: dict[str, Any]
    train_set: Dataset
    model_name: str
    dataset: str
    lr: float
    momentum: float
    weight_decay: float
    batch_size: int
    epochs: int
    feature_dim: int
    num_class: int


@ray.remote
def train(worker_func, p: BaseParams):
    """
    Ray 远程工作者的通用包装函数。

    worker 返回后统一清空显存缓存（caching allocator 预留），
    防止大张量操作（mixup / EMA / 原型等）导致 reserved 持续膨胀。
    """
    if not torch.cuda.is_available():
        raise RuntimeError("Ray Worker 未获得 CUDA GPU；本项目不支持 CPU 训练模式")

    p.client_gpu = "cuda:0"
    p.train_set = ray.get(p.train_set)

    try:
        return worker_func(p)
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
    def __init__(self, args, pfl=False, is_ssl=False):
        self.rounds: int = args.rounds
        self.start_round: int = 0
        self.num_clients: int = args.num_clients
        self.join_ratio: float = args.join_ratio
        self.epochs: int = args.epochs
        self.batch_size: int = args.batch_size
        self.lr: float = args.lr
        self.momentum: float = args.momentum
        self.weight_decay: float = args.weight_decay
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
        self.is_ssl = is_ssl
        self.checkpoint_enabled: bool = args.checkpoint_enabled
        self.checkpoint_interval: int = args.checkpoint_interval
        self.resume_from = args.resume_from
        self.checkpoint_dir = os.path.join(self.save_path, "checkpoints")

        self.acc: list[float] = []
        self.acc_proto: list[float] = []
        self.loss: list[float] = []

        self.fdg = args.fdg
        if self.fdg:
            self.selected_domains = args.selected_domains
            self.target_domain = args.target_domain

        self.ssl = args.ssl
        if self.ssl != "none" and not self.is_ssl:
            raise ValueError(f"算法 {args.algo} 不支持半监督配置 ssl={self.ssl}")
        if self.is_sfd:
            self.label_domain = args.label_domain
            self.unlabel_domain = args.unlabel_domain
            self.acc_source: list[float] = []
            self.acc_target: list[float] = []
            self.acc_source_p: list[float] = []
            self.acc_target_p: list[float] = []

        if self.ssl != "none":
            self.label_ratio = args.label_ratio

        (
            self.train_sets,
            self.test_set,
            train_counts,
            self.num_class,
        ) = load_data(args, self.pfl)

        if not self.is_ssl:
            for ds in self.train_sets.values():
                normalize_dataset(ds, self.dataset)

        if self.pfl:
            for ds in self.test_set.values():
                normalize_dataset(ds, self.dataset)
        else:
            normalize_dataset(self.test_set, self.dataset)

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
        if not torch.cuda.is_available():
            raise RuntimeError("BaseServer 未检测到 CUDA GPU")
        if isinstance(args.gpus, int):
            gpu_ids = [args.gpus]
        else:
            gpu_ids = [int(gpu_id) for gpu_id in args.gpus.split(",")]
        self.ray_gpu_fraction = 1.0 / max(1, self.max_workers_per_gpu)
        # 由于 CUDA_VISIBLE_DEVICES 会将指定 GPU 编号映射为连续的 0 到 N-1，
        # 故 Server 使用的 GPU 设备索引应为本地可见的最后一个，即 len(gpu_ids) - 1
        dev_idx = len(gpu_ids) - 1
        self.device = torch.device(f"cuda:{dev_idx}")

        # 2. 强制设备映射：在 Ray Worker 环境中逻辑显卡始终映射为 cuda:0
        self.client_gpu = {i: "cuda:0" for i in range(self.num_clients)}

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

    def save_checkpoint(self, completed_round, metrics, params):
        """保存单轮 checkpoint，并在成功后清理同实验旧轮次文件。"""
        if not self.checkpoint_enabled:
            return None
        if completed_round % self.checkpoint_interval != 0:
            return None
        if self.test:
            print("-> [Test Mode] Skip checkpoint save")
            return None

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        prefix = f"{self.file_name}_{self.cur_time}_round_"
        filename = f"{prefix}{completed_round:06d}.pt"
        path = os.path.join(self.checkpoint_dir, filename)
        temp_path = f"{path}.tmp.{os.getpid()}"
        try:
            checkpoint = {
                "round": completed_round,
                "metrics": metrics,
                "params": params,
            }
            torch.save(checkpoint, temp_path)
            os.replace(temp_path, path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

        for name in os.listdir(self.checkpoint_dir):
            if not name.startswith(prefix) or not name.endswith(".pt"):
                continue
            round_text = name[len(prefix) : -len(".pt")]
            if round_text.isdigit() and name != filename:
                os.remove(os.path.join(self.checkpoint_dir, name))
        print(f"-> Checkpoint saved: {path}")
        return path

    def load_checkpoint(self, path):
        """加载单轮 checkpoint 的通用状态。"""
        data = torch.load(path, map_location="cpu", weights_only=False)
        self.start_round = data["round"]
        metrics = data["metrics"]
        for name, values in metrics.items():
            setattr(self, name, values)
        params = data["params"]
        if "global" in params:
            self.model.load_state_dict(params["global"])
        if "client" in params:
            self.clients_state = params["client"]
        return params

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
                    self.model,
                    protos.to(self.device),
                    src_eval,
                    self.device,
                )
                self.acc_source_p.append(p_acc)
                p_acc = evaluate_prototype(
                    self.model,
                    protos.to(self.device),
                    tgt_eval,
                    self.device,
                )
                self.acc_target_p.append(p_acc)
                p_acc = evaluate_prototype(
                    self.model,
                    protos.to(self.device),
                    self.test_set,
                    self.device,
                )
                self.acc_proto.append(p_acc)
            return

        if not self.pfl:
            acc = evaluate_model(self.model, self.test_set, self.device)
            self.acc.append(acc)
            if protos is not None:
                p_acc = evaluate_prototype(
                    self.model,
                    protos.to(self.device),
                    self.test_set,
                    self.device,
                )
                self.acc_proto.append(p_acc)
            return

        # pfl=True: Ray 并行评估
        target_states = model_states if model_states is not None else self.clients_state
        client_proto = protos.cpu() if protos is not None else None
        futures = []
        for i in range(self.num_clients):
            futures.append(
                evaluate.options(
                    num_gpus=self.ray_gpu_fraction,
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
            BaseParams(
                client_id=i,
                client_gpu=self.client_gpu[i],
                model_state=self.model.state_dict(),
                train_set=self.train_sets[i],
                model_name=self.model_name,
                dataset=self.dataset,
                lr=self.lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
                batch_size=self.batch_size,
                epochs=self.epochs,
                feature_dim=self.feature_dim,
                num_class=self.num_class,
            )
            for i in selected_clients
        ]

    def run_clients(self, worker_func, parameters: list[BaseParams]):
        """通过 Ray 运行客户端训练。"""
        for p in parameters:
            p.train_set = self.train_set_refs[p.client_id]

        remote_worker = train.options(
            num_gpus=self.ray_gpu_fraction,
            scheduling_strategy="SPREAD",
        )
        futures = [remote_worker.remote(worker_func, p) for p in parameters]
        results_list = ray.get(futures)

        return {p.client_id: results_list[i] for i, p in enumerate(parameters)}

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


def get_model(
    p=None,
    dataset_name=None,
    n_class=None,
    feature_dim=None,
    *,
    model_name=None,
):
    """
    模型工厂函数。

    支持两种调用方式：
    1. get_model(p): 直接传入包含模型配置的参数对象（如 BaseParams、Server 实例等）
    2. get_model(model_name, dataset_name, n_class, feature_dim): 兼容传统显式传参
    """
    if p is not None and not isinstance(p, str):
        model_name = p.model_name
        dataset_name = getattr(p, "dataset", getattr(p, "dataset_name", None))
        n_class = getattr(p, "num_class", getattr(p, "n_class", None))
        feature_dim = p.feature_dim
    else:
        model_name = model_name or p

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
