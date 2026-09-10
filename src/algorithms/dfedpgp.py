import os
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from .utils import (
    BaseParams,
    BaseServer,
    check_losses,
    fmt_num,
    get_model,
)
from .utils.aggregate import flattened_matrix_aggregate
from .utils.topology import generate_adjacency_matrix


def get_path(args):
    """生成日志文件路径，包含拓扑参数"""
    adj_suffix = f"{args.adj_type}"
    if args.adj_type == "random":
        adj_suffix += f"_{fmt_num(args.edge_p)}"
    elif args.adj_type == "small_world":
        adj_suffix += f"_{fmt_num(args.k_small_world)}_{fmt_num(args.edge_p)}"
    elif args.adj_type == "scale_free":
        adj_suffix += f"_{fmt_num(args.m_scale_free)}"

    # 将算法的关键超参加入文件名，便于区分实验
    args.file_name = (
        f"{args.common_name}_{adj_suffix}_{fmt_num(args.local_v_epochs)}"
        f"_{fmt_num(args.lr_v)}_{fmt_num(args.momentum_v)}"
        f"_{fmt_num(args.weight_decay_v)}"
    )
    return os.path.join(args.log_path, f"{args.file_name}_{args.cur_time}.log")


@dataclass
class Params(BaseParams):
    mu: float
    head_state: dict[str, torch.Tensor]
    lr_v: float
    local_v_epochs: int
    momentum_v: float
    weight_decay_v: float


def train(p: Params):
    """
    DFedPGP 客户端工作函数：实现解耦更新和梯度推送
    """
    device = torch.device(p.client_gpu)

    # 1. 初始化模型并加载参数
    model = get_model(p).to(device)

    # 合并 body 和 head 参数以加载完整模型
    full_state = {}
    full_state.update(p.model_state)
    full_state.update(p.head_state)
    model.load_state_dict(full_state)

    # 2. 准备解偏后的特征提取器参考值 (z_0 = u/mu)
    with torch.no_grad():
        z_0 = {k: v.to(device) / p.mu for k, v in p.model_state.items()}

    # 3. 初始化两个独立的优化器
    optimizer_v = torch.optim.SGD(
        model.classifier.parameters(),
        lr=p.lr_v,
        momentum=p.momentum_v,
        weight_decay=p.weight_decay_v,
    )
    optimizer_u = torch.optim.SGD(
        model.extractor.parameters(),
        lr=p.lr,
        momentum=p.momentum,
        weight_decay=p.weight_decay,
    )

    # 4. 准备数据加载器
    loader = torch.utils.data.DataLoader(
        p.train_set, batch_size=p.batch_size, shuffle=True
    )

    # ========== Phase 1: 训练分类头 V (固定 Body 为初始解偏值 z_0) ==========
    for param in model.extractor.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

    # 将解偏后的 z_0 注入 Extractor
    model.extractor.load_state_dict(
        {k.replace("extractor.", ""): v for k, v in z_0.items()}
    )

    model.train()
    for _ in range(p.local_v_epochs):
        for x, y, *_ in loader:
            x, y = x.to(device), y.to(device)
            optimizer_v.zero_grad()
            out = model(x)
            loss = F.cross_entropy(out, y)
            check_losses(loss, locals())
            loss.backward()
            optimizer_v.step()

    # ========== Phase 2: 训练特征提取器 U (固定已训练好的分类头) ==========
    for param in model.extractor.parameters():
        param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = False

    # 恢复为原始带偏的 body_biased (U)
    model.extractor.load_state_dict(
        {k.replace("extractor.", ""): v.to(device) for k, v in p.model_state.items()}
    )

    total_loss = 0.0
    num_batches = 0

    for _ in range(p.epochs):
        for x, y, *_ in loader:
            x, y = x.to(device), y.to(device)

            # a. 执行 U -> Z 转换 (除以 mu)，使 Forward 作用在解偏状态上
            with torch.no_grad():
                for param in model.extractor.parameters():
                    param.data.div_(p.mu)

            # b. 标准前向与反向传播
            optimizer_u.zero_grad()
            out = model(x)
            loss = F.cross_entropy(out, y)
            check_losses(loss, locals())
            loss.backward()

            # c. 梯度修正与状态回滚：将梯度适配到 u，并将参数乘回 mu 复位
            with torch.no_grad():
                for param in model.extractor.parameters():
                    if param.grad is not None:
                        param.grad.data.div_(p.mu)
                    param.data.mul_(p.mu)

            # d. 执行局部更新
            optimizer_u.step()

            total_loss += loss.item()
            num_batches += 1

    # 5. 提取并返回更新后的 body 和 head
    new_full_state = model.state_dict()

    shared_keys = [k for k in new_full_state if k.startswith("extractor.")]
    head_keys = [k for k in new_full_state if k.startswith("classifier.")]

    return {
        "loss": total_loss / num_batches,  # avg_loss
        "body": {
            k: new_full_state[k].cpu().detach().clone() for k in shared_keys
        },  # body_shared
        "head": {
            k: new_full_state[k].cpu().detach().clone() for k in head_keys
        },  # head_state
    }


class Server(BaseServer):
    """
    DFedPGP Server：处理基于 Push-Sum 的去中心化个性化聚合

    关键特性：
    - 使用 Gossip 算法进行去中心化聚合
    - 维护每个客户端的共享体参数 (body) 和私有头参数 (head)
    - 使用标量 mu 进行偏置校正
    - 支持任意网络拓扑
    """

    def __init__(self, args):
        super().__init__(args, pfl=True)

        # 验证模型架构：必须有 extractor 和 classifier
        init_state = self.model.state_dict()
        self.shared_keys = [k for k in init_state if k.startswith("extractor.")]
        self.head_keys = [k for k in init_state if k.startswith("classifier.")]

        # 1. 拓扑初始化与混合矩阵预计算
        self.adj_matrix = generate_adjacency_matrix(args)
        out_degrees = self.adj_matrix.sum(dim=1)
        # 预计算 M = (A/d)^T，用于向量化混合：U_next = M @ U_curr
        self.M = (self.adj_matrix / out_degrees.view(-1, 1)).t().to(self.device)

        # 2. 客户端状态池初始化
        body_proto = {k: init_state[k].clone().cpu() for k in self.shared_keys}
        head_proto = {k: init_state[k].clone().cpu() for k in self.head_keys}

        self.client_body = {
            cid: {k: v.clone() for k, v in body_proto.items()}
            for cid in range(self.num_clients)
        }
        self.client_head = {
            cid: {k: v.clone() for k, v in head_proto.items()}
            for cid in range(self.num_clients)
        }
        self.client_mu = {cid: 1.0 for cid in range(self.num_clients)}

        # 3. 初始化 clients_state 为完整模型（用于评估）
        self.update_clients_state()

        # 4. 设置学习率
        self.lr_u = args.lr
        self.local_u_epochs = args.epochs
        self.lr_v = args.lr_v
        self.local_v_epochs = args.local_v_epochs
        self.momentum_v = args.momentum_v
        self.weight_decay_v = args.weight_decay_v

    def fit(self):
        """主训练循环"""
        num_join = max(1, int(self.num_clients * self.join_ratio))

        for r in range(self.start_round, self.rounds):
            t0 = time.time()
            print(f"\n--- DFedPGP Round {r + 1}/{self.rounds} ---")

            # 1. 随机选择参与的客户端
            selected = torch.randperm(self.num_clients)[:num_join].tolist()
            print(f"Selected clients: {selected}")

            base_params = self.build_base_params(selected)
            for base in base_params:
                base.model_state = self.client_body[base.client_id]
                base.lr = self.lr_u
                base.epochs = self.local_u_epochs

            p = [
                Params(
                    **asdict(base),
                    mu=self.client_mu[base.client_id],
                    head_state=self.client_head[base.client_id],
                    lr_v=self.lr_v,
                    local_v_epochs=self.local_v_epochs,
                    momentum_v=self.momentum_v,
                    weight_decay_v=self.weight_decay_v,
                )
                for base in base_params
            ]

            # 3. 启动客户端并行训练
            results = self.run_clients(train, p)

            # 4. 收集客户端的更新
            total_loss = 0.0
            for cid, res in results.items():
                total_loss += res["loss"]
                self.client_body[cid] = res["body"]
                self.client_head[cid] = res["head"]

            self.loss.append(total_loss / len(selected))

            # 5. 执行去中心化聚合（Gossip/Push-Sum）
            self.aggregate()

            # 6. 更新客户端状态并评估
            self.update_clients_state()
            self.evaluate()

            print(f"Avg Loss: {self.loss[-1]:.4f}, Acc: {self.acc[-1]:.2f}%")
            print(f"Round finished in {time.time() - t0:.2f} seconds")
            metrics = {"acc": self.acc, "loss": self.loss}
            params = {
                "global": self.model.state_dict(),
                "aux": {
                    "client_body": self.client_body,
                    "client_head": self.client_head,
                    "client_mu": self.client_mu,
                },
            }
            self.save_checkpoint(r + 1, metrics, params)

    def load_checkpoint(self, path):
        params = super().load_checkpoint(path)
        aux = params["aux"]
        self.client_body = aux["client_body"]
        self.client_head = aux["client_head"]
        self.client_mu = aux["client_mu"]
        self.update_clients_state()
        return params

    def aggregate(self):
        """
        执行基于 Push-Sum 的去中心化聚合。
        主体参数用 GPU 矩阵乘法加速，mu 标量保持 CPU 精度。
        """
        body_list = [self.client_body[j] for j in range(self.num_clients)]
        new_body_list = flattened_matrix_aggregate(body_list, self.M, self.device)
        self.client_body = {i: new_body_list[i] for i in range(self.num_clients)}

        # Mu 聚合保持 CPU float64（与原有精度一致）
        new_client_mu = {}
        for i in range(self.num_clients):
            weights = self.M[i].tolist()
            mu_val = 0.0
            for j in range(self.num_clients):
                mu_val += weights[j] * self.client_mu[j]
            new_client_mu[i] = mu_val
        self.client_mu = new_client_mu

    def update_clients_state(self):
        """
        将 (body, mu, head) 转换为完整的去偏模型供评估使用

        对于每个客户端：
        - 共享部分：z = u / mu（执行去偏）
        - 私有部分：v（保持不变）
        """
        for cid in range(self.num_clients):
            full_state = {}
            # 合并私有分类头
            full_state.update({k: v.clone() for k, v in self.client_head[cid].items()})
            # 共享部分执行推拉偏置校正 (z = u / mu)
            for k, v in self.client_body[cid].items():
                full_state[k] = v.clone() / self.client_mu[cid]
            self.clients_state[cid] = full_state

    def save(self):
        metrics = {"acc": self.acc, "loss": self.loss}
        params = {
            "client": self.clients_state,
            "aux": {
                "body": self.client_body,
                "head": self.client_head,
                "mu": self.client_mu,
                "topology": (
                    self.adj_matrix.cpu()
                    if isinstance(self.adj_matrix, torch.Tensor)
                    else self.adj_matrix
                ),
            },
        }
        self.deal_save(metrics, params)
