import argparse
import time
import os
import torch
import torch.nn.functional as F
import numpy as np

from .utils import BaseServer, ce_loss, get_model, run_parallel_clients


def get_path(args):
    args.file_name = f"{args.name_pre}"
    return os.path.join(args.log_path, f"{args.file_name}_{args.times}.log")


def client_worker(params):
    (
        client_id,
        device,
        model_state,
        global_protos,
        alpha,
        train_set,
        model_name,
        dataset_name,
        lr,
        batch_size,
        epochs,
        num_classes,
        feature_dim,
    ) = params

    # 1. 初始化模型并加载全局状态
    model = get_model(model_name, dataset_name, feature_dim).to(device)
    model.load_state_dict(model_state)
    global_protos = global_protos.data.clone().to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    # 2. 本地模型多轮次训练
    total_loss = 0.0
    num_batches = 0

    model.train()
    for _ in range(epochs):
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output, features = model(data)

            # 标准交叉熵分类损失
            loss_ce = ce_loss(output, target)

            # 基于原型的对比损失 (Prototypical Contrastive Loss)
            # 将特征和原型进行标准化
            features_norm = F.normalize(features, dim=1)
            protos_norm = F.normalize(global_protos, dim=1)

            # 计算相似度并求取交叉熵损失 (不带温度缩放系数)
            logits_con = torch.matmul(features_norm, protos_norm.T)
            loss_con = ce_loss(logits_con, target)

            loss = (1 - alpha) * loss_ce + alpha * loss_con
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

    # 3. 计算最新的本地原型（按类别平均特征向量）
    model.eval()
    local_protos = {}
    sum_features = torch.zeros((num_classes, feature_dim), device=device)
    sum_counts = torch.zeros(num_classes, device=device)

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            _, features = model(data)

            sum_features.index_add_(0, target, features)
            sum_counts.index_add_(
                0, target, torch.ones_like(target, dtype=torch.float32)
            )

    active_classes = torch.where(sum_counts > 0)[0]
    for c in active_classes:
        c_item = int(c.item())
        local_protos[c_item] = (sum_features[c_item] / sum_counts[c_item]).cpu()

    avg_loss = total_loss / num_batches
    model_state = {k: v.cpu() for k, v in model.state_dict().items()}

    return [avg_loss, model_state, local_protos]


class Server(BaseServer):
    def __init__(self, args: argparse.Namespace):
        super().__init__(False, args)
        self.global_protos = torch.zeros((self.num_class, self.args.feature_dim))

    def fit(self):
        num_join_clients = int(self.num_clients * self.args.join_ratio)
        num_join_clients = max(1, num_join_clients)

        for r in range(self.rounds):
            t0 = time.time()
            print(f"\n--- FedProc Round {r + 1}/{self.rounds} ---")

            selected_clients = np.random.choice(
                self.num_clients, num_join_clients, replace=False
            )
            print(f"Selected clients: {selected_clients}")

            # 计算动态权重系数 alpha = 1 - r/rounds
            alpha = 1.0 - (r / self.rounds)

            p = [
                [
                    i,
                    self.client_gpu[i],
                    self.model.state_dict(),
                    self.global_protos,
                    alpha,
                    self.train_sets[i],
                    self.args.model,
                    self.args.dataset,
                    self.args.lr,
                    self.args.batch_size,
                    self.args.epochs,
                    self.num_class,
                    self.args.feature_dim,
                ]
                for i in selected_clients
            ]

            results = run_parallel_clients(
                client_worker=client_worker,
                parameters=p,
                gpu_pools=self.gpu_pools,
                mp=self.mp,
            )

            total_loss = 0.0
            selected_states = []
            all_local_protos = []

            for idx, i in enumerate(selected_clients):
                loss, state, local_protos = results[idx]
                total_loss += loss
                selected_states.append(state)
                all_local_protos.append(local_protos)

            self.loss.append(total_loss / num_join_clients)

            # 聚合模型参数
            self.aggregate(selected_states)

            # 聚合原型向量
            self.global_protos = self.aggregate_protos(all_local_protos)

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
                "proto": self.global_protos,
            },
        }
        self.deal_save(f)

    def aggregate_protos(self, all_local_protos):
        new_protos = torch.zeros_like(self.global_protos)
        counts = torch.zeros(self.num_class)

        for local_protos in all_local_protos:
            for label, proto in local_protos.items():
                new_protos[label] += proto
                counts[label] += 1

        mask = counts > 0
        new_protos[mask] /= counts[mask].unsqueeze(1)
        # 此处亦可使用动量更新，但在基础实现中简单平均是标准做法
        new_protos[~mask] = self.global_protos[~mask]

        return new_protos
