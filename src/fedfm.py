import argparse
import os
import time

import numpy as np
import torch

from .utils import BaseServer, ce_loss, get_model, mse_loss


def add_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("FedFM Specific Arguments")
    group.add_argument(
        "--mu", type=float, default=1.0, help="Weight for contrastive guiding loss"
    )
    return parser


def get_path(args):
    args.file_name = f"{args.name_pre}_{args.mu}"
    return os.path.join(args.log_path, f"{args.file_name}_{args.times}.log")


def client_worker(params):
    """
    FedFM 客户端双阶段工作函数。
    通过 mode 参数区分当前执行的阶段：
      - mode='train'  : 阶段一，执行本地模型训练
      - mode='extract': 阶段二，使用聚合后的全局模型提取本地锚点
    """
    (
        client_id,
        device,
        model_state,
        train_set,
        model_name,
        dataset_name,
        lr,
        batch_size,
        epochs,
        mu,
        num_classes,
        feature_dim,
        mode,
        global_anchors,  # 阶段一时为全局锚点 Tensor，阶段二时为 None
    ) = params

    model = get_model(model_name, dataset_name, feature_dim).to(device)
    model.load_state_dict(model_state)
    loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    # ==================== 阶段一：本地模型训练 ====================
    if mode == "train":
        global_anchors = global_anchors.to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        total_loss = 0.0
        num_batches = 0

        model.train()
        for _ in range(epochs):
            for data, target in loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output, features, _ = model(data)
                loss_ce = ce_loss(output, target)

                # 特征与对应类别锚点之间的 MSE 损失
                target_anchors = global_anchors[target]

                # 过滤掉全零锚点（第一轮尚未建立有效锚点时的保护措施）
                valid_mask = target_anchors.abs().sum(dim=1) > 0
                if valid_mask.sum() > 0:
                    loss_cg = mse_loss(features[valid_mask], target_anchors[valid_mask])
                    loss = loss_ce + mu * loss_cg
                else:
                    loss = loss_ce

                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        model_state = {k: v.cpu() for k, v in model.state_dict().items()}
        return [avg_loss, model_state]

    # ==================== 阶段二：锚点提取（冻结模型） ====================
    elif mode == "extract":
        model.eval()
        sum_features = torch.zeros((num_classes, feature_dim), device=device)
        sum_counts = torch.zeros(num_classes, device=device)

        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(device), target.to(device)
                _, features, _ = model(data)
                sum_features.index_add_(0, target, features)
                sum_counts.index_add_(
                    0, target, torch.ones_like(target, dtype=torch.float32)
                )

        # 按类别求平均，返回本地锚点及对应的样本计数
        local_anchors = {}
        local_counts = {}
        active_classes = torch.where(sum_counts > 0)[0]
        for c in active_classes:
            c_item = int(c.item())
            local_anchors[c_item] = (sum_features[c_item] / sum_counts[c_item]).cpu()
            local_counts[c_item] = int(sum_counts[c_item].item())

        return [local_anchors, local_counts]


class Server(BaseServer):
    def __init__(self, args: argparse.Namespace):
        super().__init__(False, args)
        self.global_anchors = torch.zeros((self.num_class, self.args.feature_dim))

    def fit(self):
        num_join_clients = int(self.num_clients * self.args.join_ratio)
        num_join_clients = max(1, num_join_clients)

        for r in range(self.rounds):
            t0 = time.time()
            print(f"\n--- FedFM Round {r + 1}/{self.rounds} ---")

            selected_clients = np.random.choice(
                self.num_clients, num_join_clients, replace=False
            )
            print(f"Selected clients: {selected_clients}")

            # ========== 阶段一：下发全局模型 + 全局锚点，执行本地训练 ==========
            p_train = [
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
                    self.num_class,
                    self.args.feature_dim,
                    "train",
                    self.global_anchors,
                ]
                for i in selected_clients
            ]
            results_train = self.run_clients(client_worker, p_train)

            # 收集训练结果并聚合全局模型
            total_loss = 0.0
            selected_states = []
            for i in selected_clients:
                client_loss, client_state = results_train[i]
                total_loss += client_loss
                selected_states.append(client_state)
            self.loss.append(total_loss / num_join_clients)
            self.aggregate(selected_states)

            # ========== 阶段二：下发聚合后的全局模型，提取对齐锚点 ==========
            global_state = self.model.state_dict()
            p_extract = [
                [
                    i,
                    self.client_gpu[i],
                    global_state,
                    self.train_sets[i],
                    self.args.model,
                    self.args.dataset,
                    self.args.lr,
                    self.args.batch_size,
                    self.args.epochs,
                    self.args.mu,
                    self.num_class,
                    self.args.feature_dim,
                    "extract",
                    None,
                ]
                for i in selected_clients
            ]
            results_extract = self.run_clients(client_worker, p_extract)

            # 收集本地锚点并按样本数量加权聚合为全局锚点
            all_local_anchors = []
            all_local_counts = []
            for i in selected_clients:
                client_anchors, client_counts = results_extract[i]
                all_local_anchors.append(client_anchors)
                all_local_counts.append(client_counts)
            self.aggregate_anchors(all_local_anchors, all_local_counts)

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
                "proto": self.global_anchors,
            },
        }
        self.deal_save(f)

    def aggregate_anchors(self, all_local_anchors, all_local_counts):
        """
        将来自不同客户端的本地锚点按样本数量加权聚合为全局锚点。
        """
        # 按样本数加权求和
        weighted_sum = torch.zeros_like(self.global_anchors)
        total_counts = torch.zeros(self.num_class, dtype=torch.float32)

        for anchors, counts in zip(all_local_anchors, all_local_counts):
            for label, proto in anchors.items():
                n = counts[label]
                weighted_sum[label] += proto * n
                total_counts[label] += n

        # 对有贡献的类别取加权平均
        mask = total_counts > 0
        self.global_anchors[mask] = weighted_sum[mask] / total_counts[mask].unsqueeze(1)
