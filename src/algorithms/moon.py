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
    args.file_name = f"{args.common_name}_{_fmt_num(args.mu)}_{_fmt_num(args.tau)}"
    return os.path.join(args.log_path, f"{args.file_name}_{args.cur_time}.log")


def client_worker(params):
    """
    带有模型交叉学习对抗损失 (Model-Contrastive Loss) 的 MOON 本地训练流程。
    """
    (
        _,
        device,
        global_state,
        train_set,
        prev_state,
        model_name,
        dataset_name,
        lr,
        batch_size,
        epochs,
        mu,
        tau,
        feature_dim,
    ) = params

    # 1. 初始化包含全局权重的当前本地模型
    model = get_model(model_name, dataset_name, feature_dim).to(device)
    model.load_state_dict(global_state)

    # 2. 初始化全局模型（冻结）用于计算对抗损失
    global_model = get_model(model_name, dataset_name, feature_dim).to(device)
    global_model.load_state_dict(global_state)
    global_model.eval()

    # 3. 初始化上一轮本地模型（冻结）用于计算对抗损失
    prev_model = get_model(model_name, dataset_name, feature_dim).to(device)
    prev_model.load_state_dict(prev_state)
    prev_model.eval()

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    ce_moon = torch.nn.CosineSimilarity(dim=-1)

    total_loss = 0.0
    num_batches = 0
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            # 前向传播：获取输出和表达特征 (z)
            z = model.extractor(x)
            output = model.classifier(z)
            with torch.no_grad():
                z_glob = global_model.extractor(x)
                z_prev = prev_model.extractor(x)

            # 标准交叉熵分类损失
            loss_ce = ce_loss(output, y)

            # MOON 对抗损失计算
            # 拉近与全局模型的相似度（正样本），推远与上一轮本地模型的相似度（负样本）
            pos_sim = ce_moon(z, z_glob)
            neg_sim = ce_moon(z, z_prev)
            logits = torch.cat([pos_sim.reshape(-1, 1), neg_sim.reshape(-1, 1)], dim=1)
            logits /= tau
            labels = torch.zeros(z.size(0)).to(device).long()
            loss_con = ce_loss(logits, labels)

            # 整体损失
            loss = loss_ce + mu * loss_con
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    model_state = {k: v.cpu().detach().clone() for k, v in model.state_dict().items()}
    return {"loss": avg_loss, "state": model_state}


class Server(BaseServer):
    def __init__(self, args):
        super().__init__(False, args)
        # 使用最初始的全局模型来初始化所有客户端作为其“上一轮状态”

    def fit(self):
        num_join_clients = int(self.num_clients * self.args.join_ratio)
        num_join_clients = max(1, num_join_clients)

        for r in range(self.rounds):
            t0 = time.time()
            print(f"\n--- MOON Round {r + 1}/{self.rounds} ---")

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
                    self.clients_state[i],
                    self.args.model,
                    self.args.dataset,
                    self.args.lr,
                    self.args.batch_size,
                    self.args.epochs,
                    self.args.mu,
                    self.args.tau,
                    self.args.feature_dim,
                ]
                for i in selected_clients
            ]
            results = self.run_clients(client_worker, p)

            # 汇集各客户端回传结果，增量计算加权平均损失
            total_loss = 0.0
            selected_states = []
            current_weights = []
            for cid, res in results.items():
                total_loss += res["loss"]
                selected_states.append(res["state"])
                self.clients_state[cid] = res["state"]
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
            "state_dict": {"global": self.model.state_dict()},
        }
        self.deal_save(f)
