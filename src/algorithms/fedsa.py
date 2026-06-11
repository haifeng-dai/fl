import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from .utils import (
    BaseServer,
    ce_loss,
    dist_contrastive_loss,
    extract_prototypes,
    get_model,
    mse_loss,
    param_aggregate,
    proto_aggregate,
    _fmt_num,
)


def get_path(args):
    args.file_name = f"{args.common_name}_{_fmt_num(args.alpha_sa)}_{_fmt_num(args.lambda_r)}_{_fmt_num(args.lambda_mcl)}_{_fmt_num(args.lambda_cc)}"
    return os.path.join(args.log_path, f"{args.file_name}_{args.cur_time}.log")


def margin(anchor: torch.Tensor) -> float:
    """实现公式 (6)：计算原型间的平均边界 (Average margin)。"""
    # 过滤掉未激活类别（全零行）
    norms = torch.norm(anchor, dim=1)
    valid_indices = torch.where(norms > 1e-8)[0]
    N = len(valid_indices)
    if N <= 1:
        return 0.0

    valid_anchors = anchor[valid_indices]
    dists = torch.cdist(valid_anchors, valid_anchors, p=2)
    d = dists.sum()

    denom = (N - 1) ** 2
    return d.item() / denom


def client_worker(params):
    """
    基于语义锚点 (Semantic Anchors) 与多重正则化的 FedSA 本地训练流程。
    对齐论文公式 (5), (7), (8), (9)。
    """
    (
        _,
        device,
        model_state,
        train_set,
        prev_local_anchors,
        global_anchors,
        model_name,
        dataset_name,
        lr,
        batch_size,
        epochs,
        lambda_r,
        lambda_mcl,
        lambda_cc,
        num_classes,
        feature_dim,
    ) = params

    # 1. 初始化模型
    model = get_model(model_name, dataset_name, feature_dim).to(device)
    model.load_state_dict(model_state)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    model.train()
    total_loss = 0.0
    num_batches = 0

    global_anchors = global_anchors.to(device)
    prev_local_anchors = prev_local_anchors.to(device)

    # 为 MCL 损失计算边界 'd_i^*' - 公式 (7) 上下文
    d_star = max(margin(global_anchors), margin(prev_local_anchors))

    # 2. 训练循环
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            features = model.extractor(x)
            logits = model.classifier(features)

            # 公式 (8): 使用语义锚点作为输入进行分类器校准 (Classifier Calibration)
            output_cc = model.classifier(global_anchors)

            # 监督分类损失
            loss_ce = ce_loss(logits, y)

            # 公式 (5): 基于锚点的正则化（平滑后的 MSE_loss）
            # 在批处理训练期间，我们使用当前特征作为本地原型的代理
            loss_r = mse_loss(features, global_anchors[y])

            # 公式 (7): 边界增强对比损失 (Margin-enhanced Contrastive Loss)
            loss_mcl = dist_contrastive_loss(features, global_anchors, y, margin=d_star)

            # 公式 (8): 分类器校准损失
            loss_cc = ce_loss(output_cc, torch.arange(num_classes, device=device))

            # 公式 (9): 总体损失
            loss = (
                loss_ce
                + lambda_r * loss_r
                + lambda_mcl * loss_mcl
                + lambda_cc * loss_cc
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

    # 3. 计算最新的本地原型及样本计数
    local_anchors, local_counts = extract_prototypes(
        model, loader, num_classes, feature_dim, device, return_counts=True
    )

    model_state = {k: v.cpu().detach().clone() for k, v in model.state_dict().items()}
    return {
        "loss": total_loss / num_batches,
        "state": model_state,
        "protos": local_anchors,
        "counts": local_counts,
    }


class Server(BaseServer):
    def __init__(self, args):
        # FedSA 是个性化联邦学习算法 (pfl=True)
        super().__init__(True, args)

        # 根据论文按随机分布初始化语义锚点
        self.anchors = torch.randn(self.num_class, self.args.feature_dim)
        self.anchors = F.normalize(self.anchors, p=2, dim=1)

        self.clients_anchors = [
            self.anchors.data.clone() for _ in range(self.num_clients)
        ]

    def fit(self):
        num_join_clients = int(self.num_clients * self.args.join_ratio)
        num_join_clients = max(1, num_join_clients)

        print(f"FedSA Training with alpha_sa={self.args.alpha_sa} (EMA factor)")

        for r in range(self.rounds):
            t0 = time.time()
            print(f"\n--- FedSA Round {r + 1}/{self.rounds} ---")

            selected_clients = np.random.choice(
                self.num_clients, num_join_clients, replace=False
            )
            print(f"Selected clients: {selected_clients}")

            p = [
                [
                    i,
                    self.client_gpu[i],
                    self.clients_state[i],
                    self.train_sets[i],
                    self.clients_anchors[i],
                    self.anchors,
                    self.args.model,
                    self.args.dataset,
                    self.args.lr,
                    self.args.batch_size,
                    self.args.epochs,
                    self.args.lambda_r,
                    self.args.lambda_mcl,
                    self.args.lambda_cc,
                    self.num_class,
                    self.args.feature_dim,
                ]
                for i in selected_clients
            ]
            results = self.run_clients(client_worker, p)

            total_loss = 0.0
            selected_states = []
            local_anchors_list = []
            local_counts_list = []
            current_weights = []
            for cid, res in results.items():
                total_loss += res["loss"]
                self.clients_state[cid] = res["state"]
                selected_states.append(res["state"])
                local_anchors_list.append(res["protos"])
                local_counts_list.append(res["counts"])
                current_weights.append(self.weights[cid])
                # 更新服务端缓存的客户端锚点，用于下一轮的边界 (margin) 计算
                self.clients_anchors[cid] = res["protos"].detach().clone()

            self.loss.append(total_loss / num_join_clients)
            norm_weights = [w / sum(current_weights) for w in current_weights]

            # 1. 聚合全局模型（用于提供基础的特征表达能力）
            self.model.load_state_dict(param_aggregate(selected_states, norm_weights))

            # 2. 聚合各个本地原型以生成全局 P_bar 并更新语义锚点 A_bar (公式 10)
            self.update_global_anchors(local_anchors_list, local_counts_list)

            self.evaluate()
            print(
                f"Global Accuracy (Avg Personal): {self.acc[-1]:.2f}%, Avg Loss: {self.loss[-1]:.4f}"
            )
            print(f"Round finished in {time.time() - t0:.2f} seconds")

    def update_global_anchors(self, local_anchors_list, local_counts_list):
        """对本地原型进行加权聚合，并对语义锚点执行 EMA（指数移动平均）更新。"""
        # 使用统一的张量聚合函数，优先使用按类样本计数
        new_p_bar = proto_aggregate(
            local_anchors_list,
            local_counts_list=local_counts_list,
        )

        # 公式 (10): A_t+1 = alpha * A_t + (1 - alpha) * P_bar_t
        mask = torch.norm(new_p_bar, dim=1) > 1e-8
        alpha = self.args.alpha_sa

        self.anchors[mask] = alpha * self.anchors[mask] + (1 - alpha) * new_p_bar[mask]

    def save(self):
        f = {
            "acc": self.acc,
            "loss": self.loss,
            "state_dict": {
                "global": self.model.state_dict(),
                "proto": self.anchors.data,
            },
        }
        self.deal_save(f)
