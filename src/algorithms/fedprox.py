import os
import time

import numpy as np
import torch

from .utils import (
    BaseServer,
    ce_loss,
    get_model,
    _fmt_num,
)


def get_path(args):
    args.file_name = f"{args.common_name}_{_fmt_num(args.mu)}"
    return os.path.join(args.log_path, f"{args.file_name}_{args.cur_time}.log")


def train_worker(params):
    """
    带有近端项 (Proximal term) 的 FedProx 本地训练流程。
    """
    (
        _,
        device,
        model_state,
        train_set,
        model_name,
        dataset_name,
        lr,
        batch_size,
        epochs,
        mu,
        feature_dim,
    ) = params

    # 1. 初始化模型并加载全局状态
    model = get_model(model_name, dataset_name, feature_dim).to(device)
    model.load_state_dict(model_state)

    # 2. 缓存全局模型参数，用于计算近端正则化项
    global_model_params = {k: v.to(device) for k, v in model_state.items()}

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    total_loss = 0.0
    num_batches = 0

    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = ce_loss(logits, y)
            optimizer.zero_grad()
            loss.backward()

            # FedProx 近端项优化算法：
            # 相比于将 (mu/2)*||w-w_t||^2 加入损失函数并进行反向传播，
            # 这里直接将其关于参数的导数 mu*(w-w_t) 累加到 param.grad 中。
            # 这可以避免为正则化项构建庞大的计算图，极大节省内存和算力。
            if mu > 0:
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        assert param.grad is not None
                        if name in global_model_params and param.requires_grad:
                            # grad += mu * (param - global_param)
                            param.grad.add_(param - global_model_params[name], alpha=mu)

            optimizer.step()

            # 仅记录任务原本的分类损失用于分析，以避免和加入了正则化的计算混淆
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    model_state = {k: v.cpu().detach().clone() for k, v in model.state_dict().items()}
    return {"loss": avg_loss, "state": model_state}


class Server(BaseServer):
    def __init__(self, args):
        super().__init__(False, args)

    def fit(self):
        num_join_clients = int(self.num_clients * self.args.join_ratio)
        num_join_clients = max(1, num_join_clients)

        print(f"FedProx with mu={self.args.mu}")
        for r in range(self.rounds):
            t0 = time.time()
            print(f"\n--- FedProx Round {r + 1}/{self.rounds} ---")

            selected_clients = np.random.choice(
                self.num_clients, num_join_clients, replace=False
            )
            print(f"Selected clients: {selected_clients}")

            p = [
                [
                    i,
                    self.client_gpu[i],
                    self.model.state_dict(),
                    self.train_sets[i],
                    self.args.model,
                    self.args.dataset,
                    self.args.lr,
                    self.args.batch_size,
                    self.args.epochs,
                    self.args.mu,
                    self.args.feature_dim,
                ]
                for i in selected_clients
            ]
            results = self.run_clients(train_worker, p)

            # 汇集并处理各客户端结果
            total_loss = 0.0
            selected_states = []
            current_weights = []
            for cid, res in results.items():
                total_loss += res["loss"]
                selected_states.append(res["state"])
                current_weights.append(self.weights[cid])
            self.loss.append(total_loss / num_join_clients)
            sum_weights = sum(current_weights)
            norm_weights = [w / sum_weights for w in current_weights]

            self.aggregate(selected_states, weights=norm_weights)
            self.evaluate()
            print(
                f"Global Accuracy: {self.acc[-1]:.2f}%, Avg Loss: {self.loss[-1]:.4f}"
            )
            print(f"Round finished in {time.time() - t0:.2f} seconds")

    def save(self):
        f = {
            "acc": self.acc,
            "loss": self.loss,
            "state_dict": {
                "global": self.model.state_dict(),
            },
        }
        self.deal_save(f)
