import os
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from .utils import (
    BaseParams,
    BaseServer,
    clone_cpu_state,
    fmt_num,
    get_model,
)


@dataclass
class Params(BaseParams):
    mu: float


def get_path(args):
    args.file_name = f"{args.common_name}_{fmt_num(args.mu)}"
    return os.path.join(args.log_path, f"{args.file_name}_{args.cur_time}.log")


def train(p: Params):
    """
    带有近端项 (Proximal term) 的 FedProx 本地训练流程。
    """
    device = torch.device(p.client_gpu)

    # 1. 初始化模型并加载全局状态
    model = get_model(p.model_name, p.dataset, p.num_class, p.feature_dim).to(device)
    model.load_state_dict(p.model_state)

    # 2. 缓存全局模型参数，用于计算近端正则化项
    global_model_params = {k: v.to(device) for k, v in p.model_state.items()}

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=p.lr,
        momentum=p.momentum,
        weight_decay=p.weight_decay,
    )
    loader = torch.utils.data.DataLoader(
        p.train_set, batch_size=p.batch_size, shuffle=True
    )

    total_loss = 0.0
    num_batches = 0

    for _ in range(p.epochs):
        for x, y, *_ in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            optimizer.zero_grad()
            loss.backward()

            # FedProx 近端项优化算法：
            # 相比于将 (mu/2)*||w-w_t||^2 加入损失函数并进行反向传播，
            # 这里直接将其关于参数的导数 mu*(w-w_t) 累加到 param.grad 中。
            # 这可以避免为正则化项构建庞大的计算图，极大节省内存和算力。
            if p.mu > 0:
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        assert param.grad is not None
                        if name in global_model_params and param.requires_grad:
                            # grad += mu * (param - global_param)
                            param.grad.add_(
                                param - global_model_params[name], alpha=p.mu
                            )

            optimizer.step()

            # 仅记录任务原本的分类损失用于分析，以避免和加入了正则化的计算混淆
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    model_state = clone_cpu_state(model.state_dict())
    return {"loss": avg_loss, "state": model_state}


class Server(BaseServer):
    def __init__(self, args):
        super().__init__(args)
        self.mu = args.mu

    def fit(self):
        num_join = max(1, int(self.num_clients * self.join_ratio))

        print(f"FedProx with mu={self.mu}")
        for r in range(self.start_round, self.rounds):
            t0 = time.time()
            print(f"\n--- FedProx Round {r + 1}/{self.rounds} ---")

            selected = torch.randperm(self.num_clients)[:num_join].tolist()
            print(f"Selected clients: {selected}")

            p = [
                Params(**asdict(base), mu=self.mu)
                for base in self.build_base_params(selected)
            ]
            results = self.run_clients(train, p)

            # 汇集并处理各客户端结果
            total_loss = 0.0
            selected_states = []
            current_weights = []
            for cid, res in results.items():
                total_loss += res["loss"]
                selected_states.append(res["state"])
                current_weights.append(self.weights[cid])
            self.loss.append(total_loss / num_join)
            sum_weights = sum(current_weights)
            norm_weights = [w / sum_weights for w in current_weights]

            self.aggregate(selected_states, weights=norm_weights)
            self.evaluate()
            print(
                f"Global Accuracy: {self.acc[-1]:.2f}%, Avg Loss: {self.loss[-1]:.4f}"
            )
            print(f"Round finished in {time.time() - t0:.2f} seconds")
            metrics = {"acc": self.acc, "loss": self.loss}
            params = {"global": self.model.state_dict()}
            self.save_checkpoint(r + 1, metrics, params)

    def save(self):
        metrics = {"acc": self.acc, "loss": self.loss}
        params = {"global": self.model.state_dict()}
        self.deal_save(metrics, params)
