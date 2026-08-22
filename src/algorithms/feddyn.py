import os
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from .utils import (
    BaseParams,
    BaseServer,
    fmt_num,
    get_model,
)


def get_path(args):
    args.file_name = f"{args.common_name}_{fmt_num(args.alpha_coef)}"
    return os.path.join(args.log_path, f"{args.file_name}_{args.cur_time}.log")


@dataclass
class Params(BaseParams):
    grad_prev: torch.Tensor | None
    global_model_vector: torch.Tensor | None
    alpha_coef: float


def train(p: Params):
    device = torch.device(p.client_gpu)

    # 1. 初始化模型并加载全局状态
    model = get_model(p.model_name, p.dataset, p.num_class, p.feature_dim).to(device)
    model.load_state_dict(p.model_state)
    optimizer = torch.optim.SGD(model.parameters(), lr=p.lr)
    loader = torch.utils.data.DataLoader(p.train_set, batch_size=p.batch_size, shuffle=True)

    # 将参数向量移动到计算设备
    grad_prev = p.grad_prev.to(device) if p.grad_prev is not None else None
    global_model_vector = (
        p.global_model_vector.to(device) if p.global_model_vector is not None else None
    )

    total_loss = 0.0
    num_batches = 0

    model.train()
    for _ in range(p.epochs):
        for x, y, *_ in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            task_loss = F.cross_entropy(logits, y)

            # FedDyn 动态正则化项
            # L = L_task - <grad_prev, w> + (alpha/2) * ||w - w_global||^2
            curr_params = parameters_to_vector(model.parameters())

            # 线性惩罚项: - <grad_prev, w>
            lin_penalty = 0.0
            if grad_prev is not None:
                lin_penalty = -torch.dot(grad_prev, curr_params)

            # 二次惩罚项: (alpha/2) * ||w - w_global||^2
            quad_penalty = 0.0
            if global_model_vector is not None:
                diff = curr_params - global_model_vector
                quad_penalty = (p.alpha_coef / 2.0) * torch.sum(diff**2)

            loss = task_loss + lin_penalty + quad_penalty

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += task_loss.item()
            num_batches += 1

    return {
        "loss": total_loss / num_batches,
        "state": {k: v.cpu().detach().clone() for k, v in model.state_dict().items()},
    }


class Server(BaseServer):
    def __init__(self, args):
        super().__init__(False, args)
        self.alpha_coef = args.alpha_coef

        # FedDyn 服务器状态
        # h: 全局梯度历史记录（向量模式）
        # 使用 parameters_to_vector 获取其结构并初始化为零向量
        self.h = parameters_to_vector(self.model.parameters()).detach().clone().zero_()

        # 本地梯度历史记录 (nabla L_k)
        # 存储于 CPU 内存中以节省 GPU 显存
        self.local_grads = {
            i: torch.zeros_like(self.h) for i in range(self.num_clients)
        }

    def fit(self):
        num_join = max(1, int(self.num_clients * self.join_ratio))

        # 用于下一轮次训练的全局模型向量
        global_model_vector = (
            parameters_to_vector(self.model.parameters()).detach().clone()
        )

        for r in range(self.rounds):
            t0 = time.time()
            print(f"\n--- FedDyn Round {r + 1}/{self.rounds} ---")

            selected = torch.randperm(self.num_clients)[:num_join].tolist()
            print(f"Selected clients: {selected}")

            p = [
                Params(
                    **asdict(base),
                    grad_prev=self.local_grads[base.client_id],
                    global_model_vector=global_model_vector,
                    alpha_coef=self.alpha_coef,
                )
                for base in self.build_base_params(selected)
            ]
            results = self.run_clients(train, p)

            total_loss = 0.0
            sum_model_params = torch.zeros_like(global_model_vector)
            for cid, res in results.items():
                total_loss += res["loss"]

                # 将客户端模型状态字典转化为一维向量
                self.model.load_state_dict(res["state"])
                client_flat = parameters_to_vector(self.model.parameters()).detach()

                sum_model_params += client_flat

                # 更新本地梯度历史记录：
                # nabla L_k(w^{t+1}) 约等于 nabla L_k(w^t) - alpha * (w^{t+1} - w^t)
                model_diff = client_flat - global_model_vector
                self.local_grads[cid] -= self.alpha_coef * model_diff
            self.loss.append(total_loss / num_join)

            # 1. 计算所有客户端模型的平均值
            avg_model_params = sum_model_params / num_join

            # 2. 更新全局历史梯度 h
            # 理论公式: h_{t+1} = h_t - \alpha * \frac{|P_t|}{N} * (w_{avg} - w_t)
            # 在非全量客户端参与时，必须乘以参与比例 (num_join / num_clients) 防止更新过激导致散度爆炸
            scale_factor = num_join / self.num_clients
            self.h -= (
                self.alpha_coef
                * scale_factor
                * (avg_model_params - global_model_vector)
            )

            # 3. 更新全局模型参数
            # w_{t+1} = w_{avg} - (1/alpha) * h_{t+1}
            new_global_vector = avg_model_params - (1.0 / self.alpha_coef) * self.h

            # 重新加载回模型实体中
            vector_to_parameters(new_global_vector, self.model.parameters())
            global_model_vector = new_global_vector

            self.evaluate()
            print(
                f"Global Accuracy: {self.acc[-1]:.2f}%, Avg Loss: {self.loss[-1]:.4f}"
            )
            print(f"Round finished in {time.time() - t0:.2f} seconds")

    def save(self):
        metrics = {"acc": self.acc, "loss": self.loss}
        params = {"global": self.model.state_dict(), "aux": {"h": self.h}}
        self.deal_save(metrics, params)
