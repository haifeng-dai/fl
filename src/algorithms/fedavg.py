import os
import time

import torch

from .utils import (
    BaseServer,
    ce_loss,
    get_model,
)


def get_path(args):
    args.file_name = f"{args.common_name}"
    return os.path.join(args.log_path, f"{args.file_name}_{args.cur_time}.log")


def train(params):
    """
    标准的 FedAvg 本地训练流程。
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
        feature_dim,
        num_class,
    ) = params

    # 1. 初始化模型并加载最新的全局模型参数
    model = get_model(model_name, dataset_name, num_class, feature_dim).to(device)
    model.load_state_dict(model_state)

    # 2. 设置优化器与数据加载器
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    # 3. 本地模型多轮次 (Epochs) 训练
    total_loss = 0.0
    num_batches = 0
    for _ in range(epochs):
        for x, y, *_ in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = ce_loss(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
    avg_loss = total_loss / num_batches

    # 4. 整理返回结果（将模型状态移至 CPU 以节省 GPU 显存容量消耗）
    model_state = {k: v.cpu().detach().clone() for k, v in model.state_dict().items()}
    return {"loss": avg_loss, "state": model_state}


class Server(BaseServer):
    def __init__(self, args):
        # FedAvg 是传统的全局联邦学习方法，因此 pfl=False
        super().__init__(False, args)

    def fit(self):
        """运行 FedAvg 训练流程"""
        num_join = max(1, int(self.num_clients * self.join_ratio))

        for r in range(self.rounds):
            t0 = time.time()
            print(f"\n--- FedAvg Round {r + 1}/{self.rounds} ---")

            selected = sorted(torch.randperm(self.num_clients)[:num_join].tolist())
            print(f"Selected clients: {selected}")

            p = self.build_base_params(selected)
            results = self.run_clients(train, p)

            # 汇集各客户端的回传结果，计算总损失与聚合权重分布
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

            # 根据客户端的数据量权重，对上传的模型参数进行加权平均汇聚
            self.aggregate(selected_states, weights=norm_weights)
            self.evaluate()
            print(
                f"Global Accuracy: {self.acc[-1]:.2f}%, Avg Loss: {self.loss[-1]:.4f}"
            )
            print(f"Round finished in {time.time() - t0:.2f} seconds")

    def save(self):
        """保存全局模型的实验结果与最终参数"""
        metrics = {"acc": self.acc, "loss": self.loss}
        params = {"global": self.model.state_dict()}
        self.deal_save(metrics, params)
