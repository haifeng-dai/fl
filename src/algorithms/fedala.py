import os
import random
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from .utils import (
    BaseParams,
    BaseServer,
    check_losses,
    clone_cpu_state,
    fmt_num,
    get_model,
)


def get_path(args):
    args.file_name = f"{args.common_name}_{fmt_num(args.eta)}_{fmt_num(args.rand_percent)}_{fmt_num(args.layer_idx)}_{fmt_num(args.ala_threshold)}_{fmt_num(args.num_pre_loss)}"
    return os.path.join(args.log_path, f"{args.file_name}_{args.cur_time}.log")


@dataclass
class Params(BaseParams):
    local_model_state: dict[str, torch.Tensor]
    saved_weights: list[torch.Tensor] | None
    eta: float
    rand_percent: int
    layer_idx: int
    ala_threshold: float
    num_pre_loss: int


class ALA:
    """FedALA 的自适应本地聚合 (Adaptive Local Aggregation) 模块"""

    def __init__(
        self,
        client_id: int,
        train_data,
        batch_size: int,
        model_name: str,
        dataset_name: str,
        rand_percent: int,
        layer_idx: int = 0,
        eta: float = 1.0,
        device: str = "cpu",
        threshold: float = 0.1,
        num_pre_loss: int = 10,
        feature_dim: int = 512,
        n_class: int = 10,
    ):
        self.client_id = client_id
        self.train_data = train_data
        self.batch_size = batch_size
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.rand_percent = rand_percent
        self.layer_idx = layer_idx
        self.eta = eta
        self.threshold = threshold
        self.num_pre_loss = num_pre_loss
        self.feature_dim = feature_dim
        self.n_class = n_class
        self.device = device

        self.weights = None  # 可学习的本地聚合权重
        self.start_phase = True

    def adaptive_local_aggregation(
        self, global_model: torch.nn.Module, local_model: torch.nn.Module
    ):
        """
        应用自适应本地聚合来初始化本地模型。

        参数:
            global_model: 接收到的全局模型
            local_model: 当前的本地模型
        """
        # 随机采样部分本地训练数据进行权重优化抽样过程
        rand_ratio = self.rand_percent / 100
        rand_num = int(rand_ratio * len(self.train_data))
        rand_idx = random.randint(0, len(self.train_data) - rand_num)

        # 为聚合权重优化准备采样数据
        indices = list(range(rand_idx, rand_idx + rand_num))
        subset = Subset(self.train_data, indices)
        rand_loader = DataLoader(subset, self.batch_size, drop_last=False)

        # 获取参数引用
        params_g = list(global_model.parameters())
        params = list(local_model.parameters())

        # 如果本地模型和全局模型完全一致（例如初始轮次），则跳过 ALA
        if torch.sum(params_g[0] - params[0]) == 0:
            return

        # 保留底层的所有更新内容
        if self.layer_idx > 0:
            for param, param_g in zip(
                params[: -self.layer_idx], params_g[: -self.layer_idx]
            ):
                param.data = param_g.data.clone()

        # 初始化用于精炼聚合权重的辅助模型
        model_t = get_model(
            self.model_name, self.dataset_name, self.n_class, self.feature_dim
        )
        model_t.to(self.device)
        model_t.load_state_dict(local_model.state_dict())
        params_t = list(model_t.parameters())

        # 选择 ALA 的候选层；如果 layer_idx 为 0，则默认选择所有层
        params_p = params[-self.layer_idx :]
        params_gp = params_g[-self.layer_idx :]
        params_tp = params_t[-self.layer_idx :]

        # 冻结低层参数以减少计算开销
        if self.layer_idx > 0:
            for param in params_t[: -self.layer_idx]:
                param.requires_grad = False

        # 使用占位优化器；权重将通过 ALA 特定的推导过程手动更新
        optimizer = torch.optim.SGD(params_tp, lr=0)

        # 初始时将权重初始化为全 1
        if self.weights is None:
            self.weights = [
                torch.ones_like(param.data).to(self.device) for param in params_p
            ]

        # 初始化辅助模型中的高层参数
        for param_t, param, param_g, weight in zip(
            params_tp, params_p, params_gp, self.weights
        ):
            param_t.data = param + (param_g - param) * weight

        # 权重学习循环
        loss_t = []
        losses = []
        while True:
            for x, y, *_ in rand_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                output = model_t(x)
                loss = F.cross_entropy(output, y)
                check_losses(loss, locals())
                loss.backward()

                # 利用 ALA 梯度推导新的聚合权重
                for param_t, param, param_g, weight in zip(
                    params_tp, params_p, params_gp, self.weights
                ):
                    assert param_t.grad is not None
                    weight.data = torch.clamp(
                        weight - self.eta * (param_t.grad * (param_g - param)), 0, 1
                    )

                # 使用更新后的权重执行自适应本地聚合步骤
                for param_t, param, param_g, weight in zip(
                    params_tp, params_p, params_gp, self.weights
                ):
                    param_t.data = param + (param_g - param) * weight

                loss_t.append(loss.item())
            losses.append(np.mean(loss_t))

            # 在随后的迭代中仅训练一个 epoch
            if not self.start_phase:
                break

            # 在第一轮迭代中训练至收敛
            if (
                len(losses) > self.num_pre_loss
                and np.std(losses[-self.num_pre_loss :]) < self.threshold
            ):
                break

        # 将学习到的聚合参数应用到本地模型中
        for param, param_t in zip(params_p, params_tp):
            param.data = param_t.data.clone()


def train(p: Params):
    device = torch.device(p.client_gpu)

    # 初始化模型
    global_model = get_model(p.model_name, p.dataset, p.num_class, p.feature_dim).to(
        device
    )
    global_model.load_state_dict(p.model_state)
    local_model = get_model(p.model_name, p.dataset, p.num_class, p.feature_dim).to(
        device
    )
    local_model.load_state_dict(p.local_model_state)

    # 初始化 ALA 模块
    ala = ALA(
        client_id=p.client_id,
        train_data=p.train_set,
        batch_size=p.batch_size,
        model_name=p.model_name,
        dataset_name=p.dataset,
        rand_percent=p.rand_percent,
        layer_idx=p.layer_idx,
        eta=p.eta,
        device=device,
        threshold=p.ala_threshold,
        num_pre_loss=p.num_pre_loss,
        feature_dim=p.feature_dim,
        n_class=p.num_class,
    )

    # 如果存在（非首次参与），则加载先前学习到的聚合权重
    if p.saved_weights is not None:
        ala.weights = [w.to(device) for w in p.saved_weights]

    # 确定 ALA 阶段：从未参与过的客户端（无权重）执行收敛学习，
    # 而对于具有聚合历史记录的客户端，则进行后续的微观调优。
    ala.start_phase = p.saved_weights is None

    # 执行 ALA（内部逻辑会自动处理模型一致的情况并跳过）
    ala.adaptive_local_aggregation(global_model, local_model)

    # 执行标准的本地训练流程
    optimizer = torch.optim.SGD(
        local_model.parameters(),
        lr=p.lr,
        momentum=p.momentum,
        weight_decay=p.weight_decay,
    )
    loader = DataLoader(p.train_set, batch_size=p.batch_size, shuffle=True)

    total_loss = 0.0
    num_batches = 0
    for _ in range(p.epochs):
        for x, y, *_ in loader:
            x, y = x.to(device), y.to(device)
            output = local_model(x)
            loss = F.cross_entropy(output, y)
            optimizer.zero_grad()
            check_losses(loss, locals())
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
    avg_loss = total_loss / num_batches

    # 返回结果（移动至 CPU 并克隆以彻底释放句柄）
    model_state = clone_cpu_state(local_model.state_dict())
    weights_cpu = None
    if ala.weights is not None:
        weights_cpu = [w.cpu().detach().clone() for w in ala.weights]
    return {
        "loss": avg_loss,
        "state": model_state,
        "weights": weights_cpu,
    }


class Server(BaseServer):
    def __init__(self, args):
        # FedALA 是一种个性化联邦学习 (pFL) 方法
        super().__init__(args, pfl=True)
        self.eta = args.eta
        self.rand_percent = args.rand_percent
        self.layer_idx = args.layer_idx
        self.ala_threshold = args.ala_threshold
        self.num_pre_loss = args.num_pre_loss

        self.clients_weights = [None] * self.num_clients

    def fit(self):
        """运行 FedALA 训练流程"""
        num_join = max(1, int(self.num_clients * self.join_ratio))

        for r in range(self.start_round, self.rounds):
            t0 = time.time()
            print(f"\n--- FedALA Round {r + 1}/{self.rounds} ---")

            # 选择本轮参与的客户端
            selected = torch.randperm(self.num_clients)[:num_join].tolist()
            print(f"Selected clients: {selected}")

            # 为并行执行准备参数配置
            global_model_state_cpu = {
                k: v.cpu() for k, v in self.model.state_dict().items()
            }

            base_params = self.build_base_params(selected)
            for base in base_params:
                base.model_state = global_model_state_cpu

            p = [
                Params(
                    **asdict(base),
                    local_model_state=self.clients_state[base.client_id],
                    saved_weights=self.clients_weights[base.client_id],
                    eta=self.eta,
                    rand_percent=self.rand_percent,
                    layer_idx=self.layer_idx,
                    ala_threshold=self.ala_threshold,
                    num_pre_loss=self.num_pre_loss,
                )
                for base in base_params
            ]
            # 运行并行客户端训练任务
            results = self.run_clients(train, p)

            # 处理结果：计算总损失，获取所选客户端的状态和权重信息清单。
            total_loss = 0.0
            selected_states = []
            current_weights = []
            for cid, res in results.items():
                total_loss += res["loss"]
                self.clients_state[cid] = res["state"]
                self.clients_weights[cid] = res["weights"]
                selected_states.append(res["state"])
                current_weights.append(self.weights[cid])
            self.loss.append(total_loss / num_join)
            sum_weights = sum(current_weights)
            norm_weights = [w / sum_weights for w in current_weights]

            self.aggregate(selected_states, weights=norm_weights)
            self.evaluate()

            print(
                f"Personalized Accuracy: {self.acc[-1]:.2f}%, Avg Loss: {self.loss[-1]:.4f}"
            )
            print(f"Round finished in {time.time() - t0:.2f} seconds")
            metrics = {"acc": self.acc, "loss": self.loss}
            params = {
                "global": self.model.state_dict(),
                "client": self.clients_state,
                "aux": {"clients_weights": self.clients_weights},
            }
            self.save_checkpoint(r + 1, metrics, params)

    def load_checkpoint(self, path):
        params = super().load_checkpoint(path)
        self.clients_weights = params["aux"]["clients_weights"]
        return params

    def save(self):
        metrics = {"acc": self.acc, "loss": self.loss}
        params = {
            "client": self.clients_state,
            "aux": {"clients_weights": self.clients_weights},
        }
        self.deal_save(metrics, params)
