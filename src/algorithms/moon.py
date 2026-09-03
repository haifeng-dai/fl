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


def get_path(args):
    args.file_name = f"{args.common_name}_{fmt_num(args.mu)}_{fmt_num(args.tau)}"
    return os.path.join(args.log_path, f"{args.file_name}_{args.cur_time}.log")


@dataclass
class Params(BaseParams):
    prev_state: dict[str, torch.Tensor]
    mu: float
    tau: float


def train(p: Params):
    """
    带有模型交叉学习对抗损失 (Model-Contrastive Loss) 的 MOON 本地训练流程。
    """
    device = torch.device(p.client_gpu)

    # 1. 初始化包含全局权重的当前本地模型
    model = get_model(p.model_name, p.dataset, p.num_class, p.feature_dim).to(device)
    model.load_state_dict(p.model_state)

    # 2. 初始化全局模型（冻结）用于计算对抗损失
    global_model = get_model(p.model_name, p.dataset, p.num_class, p.feature_dim).to(
        device
    )
    global_model.load_state_dict(p.model_state)
    global_model.eval()

    # 3. 初始化上一轮本地模型（冻结）用于计算对抗损失
    prev_model = get_model(p.model_name, p.dataset, p.num_class, p.feature_dim).to(
        device
    )
    prev_model.load_state_dict(p.prev_state)
    prev_model.eval()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=p.lr,
        momentum=p.momentum,
        weight_decay=p.weight_decay,
    )
    loader = torch.utils.data.DataLoader(
        p.train_set, batch_size=p.batch_size, shuffle=True
    )
    ce_moon = torch.nn.CosineSimilarity(dim=-1)

    total_loss = 0.0
    num_batches = 0
    for _ in range(p.epochs):
        for x, y, *_ in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            # 前向传播：获取输出和表达特征 (z)
            z = model.extractor(x)
            output = model.classifier(z)
            with torch.no_grad():
                z_glob = global_model.extractor(x)
                z_prev = prev_model.extractor(x)

            # 标准交叉熵分类损失
            loss_ce = F.cross_entropy(output, y)

            # MOON 对抗损失计算
            # 拉近与全局模型的相似度（正样本），推远与上一轮本地模型的相似度（负样本）
            pos_sim = ce_moon(z, z_glob)
            neg_sim = ce_moon(z, z_prev)
            logits = torch.cat([pos_sim.reshape(-1, 1), neg_sim.reshape(-1, 1)], dim=1)
            logits /= p.tau
            labels = torch.zeros(z.size(0)).to(device).long()
            loss_con = F.cross_entropy(logits, labels)

            # 整体损失
            loss = loss_ce + p.mu * loss_con
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    model_state = clone_cpu_state(model.state_dict())
    return {"loss": avg_loss, "state": model_state}


class Server(BaseServer):
    def __init__(self, args):
        super().__init__(args)
        self.mu = args.mu
        self.tau = args.tau

    def fit(self):
        num_join = max(1, int(self.num_clients * self.join_ratio))

        for r in range(self.rounds):
            t0 = time.time()
            print(f"\n--- MOON Round {r + 1}/{self.rounds} ---")

            selected = torch.randperm(self.num_clients)[:num_join].tolist()
            print(f"Selected clients: {selected}")

            p = [
                Params(
                    **asdict(base),
                    prev_state=self.clients_state[base.client_id],
                    mu=self.mu,
                    tau=self.tau,
                )
                for base in self.build_base_params(selected)
            ]
            results = self.run_clients(train, p)

            # 汇集各客户端回传结果，增量计算加权平均损失
            total_loss = 0.0
            selected_states = []
            current_weights = []
            for cid, res in results.items():
                total_loss += res["loss"]
                selected_states.append(res["state"])
                self.clients_state[cid] = res["state"]
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

    def save(self):
        metrics = {"acc": self.acc, "loss": self.loss}
        params = {"global": self.model.state_dict()}
        self.deal_save(metrics, params)
