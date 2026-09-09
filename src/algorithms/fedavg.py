import os
import time

import torch
import torch.nn.functional as F

from .utils import (
    BaseParams,
    BaseServer,
    check_losses,
    clone_cpu_state,
    get_model,
)


def get_path(args):
    args.file_name = f"{args.common_name}"
    return os.path.join(args.log_path, f"{args.file_name}_{args.cur_time}.log")


def train(p: BaseParams):
    """
    标准的 FedAvg 本地训练流程。
    """
    device = torch.device(p.client_gpu)

    # 1. 初始化模型并加载最新的全局模型参数
    model = get_model(p.model_name, p.dataset, p.num_class, p.feature_dim).to(device)
    model.load_state_dict(p.model_state)

    # 2. 设置优化器与数据加载器
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=p.lr,
        momentum=p.momentum,
        weight_decay=p.weight_decay,
    )
    loader = torch.utils.data.DataLoader(
        p.train_set, batch_size=p.batch_size, shuffle=True
    )

    # 3. 本地模型多轮次 (Epochs) 训练
    total_loss = 0.0
    num_batches = 0
    for _ in range(p.epochs):
        for x, y, *_ in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            optimizer.zero_grad()
            check_losses(loss, locals())
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
    avg_loss = total_loss / max(1, num_batches)

    # 4. 整理返回结果（将模型状态移至 CPU 以节省 GPU 显存容量消耗）
    model_state = clone_cpu_state(model.state_dict())
    return {"loss": avg_loss, "state": model_state}


class Server(BaseServer):
    def __init__(self, args):
        # FedAvg 是传统的全局联邦学习方法，因此 pfl=False
        super().__init__(args)

    def fit(self):
        """运行 FedAvg 训练流程"""
        num_join = max(1, int(self.num_clients * self.join_ratio))

        for r in range(self.start_round, self.rounds):
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
            metrics = {"acc": self.acc, "loss": self.loss}
            params = {"global": self.model.state_dict()}
            self.save_checkpoint(r + 1, metrics, params)

    def save(self):
        """保存全局模型的实验结果与最终参数"""
        metrics = {"acc": self.acc, "loss": self.loss}
        params = {"global": self.model.state_dict()}
        self.deal_save(metrics, params)
