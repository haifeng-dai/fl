import os
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from .utils import (
    BaseParams,
    BaseServer,
    clone_cpu_state,
    dist_contrastive_loss,
    extract_prototypes,
    fmt_num,
    get_model,
    param_aggregate,
    proto_aggregate,
)


def get_path(args):
    args.file_name = f"{args.common_name}_{fmt_num(args.alpha_sa)}_{fmt_num(args.lambda_r)}_{fmt_num(args.lambda_mcl)}_{fmt_num(args.lambda_cc)}"
    return os.path.join(args.log_path, f"{args.file_name}_{args.cur_time}.log")


@dataclass
class Params(BaseParams):
    prev_local_anchors: torch.Tensor
    global_anchors: torch.Tensor
    lambda_r: float
    lambda_mcl: float
    lambda_cc: float


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


def train(p: Params):
    """
    基于语义锚点 (Semantic Anchors) 与多重正则化的 FedSA 本地训练流程。
    对齐论文公式 (5), (7), (8), (9)。
    """
    device = torch.device(p.client_gpu)

    # 1. 初始化模型
    model = get_model(p.model_name, p.dataset, p.num_class, p.feature_dim).to(device)
    model.load_state_dict(p.model_state)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=p.lr,
        momentum=p.momentum,
        weight_decay=p.weight_decay,
    )
    loader = torch.utils.data.DataLoader(
        p.train_set, batch_size=p.batch_size, shuffle=True
    )

    model.train()
    total_loss = 0.0
    num_batches = 0

    global_anchors = p.global_anchors.to(device)
    prev_local_anchors = p.prev_local_anchors.to(device)

    # 为 MCL 损失计算边界 'd_i^*' - 公式 (7) 上下文
    d_star = max(margin(global_anchors), margin(prev_local_anchors))

    # 2. 训练循环
    for _ in range(p.epochs):
        for x, y, *_ in loader:
            x, y = x.to(device), y.to(device)
            features = model.extractor(x)
            logits = model.classifier(features)

            # 公式 (8): 使用语义锚点作为输入进行分类器校准 (Classifier Calibration)
            output_cc = model.classifier(global_anchors)

            # 监督分类损失
            loss_ce = F.cross_entropy(logits, y)

            # 公式 (5): 基于锚点的正则化（平滑后的 MSE_loss）
            # 在批处理训练期间，我们使用当前特征作为本地原型的代理
            loss_r = F.mse_loss(features, global_anchors[y])

            # 公式 (7): 边界增强对比损失 (Margin-enhanced Contrastive Loss)
            loss_mcl = dist_contrastive_loss(features, global_anchors, y, margin=d_star)

            # 公式 (8): 分类器校准损失
            loss_cc = F.cross_entropy(
                output_cc, torch.arange(p.num_class, device=device)
            )

            # 公式 (9): 总体损失
            loss = (
                loss_ce
                + p.lambda_r * loss_r
                + p.lambda_mcl * loss_mcl
                + p.lambda_cc * loss_cc
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

    # 3. 计算最新的本地原型及样本计数
    local_anchors, local_counts = extract_prototypes(
        model, loader, p.num_class, p.feature_dim, device, return_counts=True
    )

    model_state = clone_cpu_state(model.state_dict())
    return {
        "loss": total_loss / num_batches,
        "state": model_state,
        "protos": local_anchors,
        "counts": local_counts,
    }


class Server(BaseServer):
    def __init__(self, args):
        # FedSA 是个性化联邦学习算法 (pfl=True)
        super().__init__(args, pfl=True)
        self.alpha_sa = args.alpha_sa
        self.lambda_r = args.lambda_r
        self.lambda_mcl = args.lambda_mcl
        self.lambda_cc = args.lambda_cc

        # 根据论文按随机分布初始化语义锚点
        self.anchors = torch.randn(self.num_class, self.feature_dim)
        self.anchors = F.normalize(self.anchors, p=2, dim=1)

        self.clients_anchors = [
            self.anchors.data.clone() for _ in range(self.num_clients)
        ]

    def fit(self):
        num_join = max(1, int(self.num_clients * self.join_ratio))

        print(f"FedSA Training with alpha_sa={self.alpha_sa} (EMA factor)")

        for r in range(self.start_round, self.rounds):
            t0 = time.time()
            print(f"\n--- FedSA Round {r + 1}/{self.rounds} ---")

            selected = torch.randperm(self.num_clients)[:num_join].tolist()
            print(f"Selected clients: {selected}")

            base_params = self.build_base_params(selected)
            for base in base_params:
                base.model_state = self.clients_state[base.client_id]

            p = [
                Params(
                    **asdict(base),
                    prev_local_anchors=self.clients_anchors[base.client_id],
                    global_anchors=self.anchors,
                    lambda_r=self.lambda_r,
                    lambda_mcl=self.lambda_mcl,
                    lambda_cc=self.lambda_cc,
                )
                for base in base_params
            ]
            results = self.run_clients(train, p)

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

            self.loss.append(total_loss / num_join)
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
            metrics = {"acc": self.acc, "loss": self.loss}
            params = {
                "global": self.model.state_dict(),
                "client": self.clients_state,
                "aux": {
                    "anchors": self.anchors,
                    "clients_anchors": self.clients_anchors,
                },
            }
            self.save_checkpoint(r + 1, metrics, params)

    def load_checkpoint(self, path):
        params = super().load_checkpoint(path)
        aux = params["aux"]
        self.anchors = aux["anchors"]
        self.clients_anchors = aux["clients_anchors"]
        return params

    def update_global_anchors(self, local_anchors_list, local_counts_list):
        """对本地原型进行加权聚合，并对语义锚点执行 EMA（指数移动平均）更新。"""
        # 使用统一的张量聚合函数，优先使用按类样本计数
        new_p_bar = proto_aggregate(
            local_anchors_list,
            local_counts_list=local_counts_list,
        )

        # 公式 (10): A_t+1 = alpha * A_t + (1 - alpha) * P_bar_t
        mask = torch.norm(new_p_bar, dim=1) > 1e-8
        alpha = self.alpha_sa

        self.anchors[mask] = alpha * self.anchors[mask] + (1 - alpha) * new_p_bar[mask]

    def save(self):
        metrics = {"acc": self.acc, "loss": self.loss}
        params = {"global": self.model.state_dict(), "proto": self.anchors.data}
        self.deal_save(metrics, params)
