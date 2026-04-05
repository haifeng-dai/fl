import argparse
import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .utils import (
    BaseServer,
    ce_loss,
    extract_prototypes,
    get_model,
    mse_loss,
)


def add_args(parser: argparse.ArgumentParser):
    """添加 FedTGP 相关的特定参数"""
    group = parser.add_argument_group("FedTGP Specific Arguments")
    group.add_argument(
        "--lamda_",
        type=float,
        default=10.0,
        help="Weight for prototype matching loss (default: 10.0)",
    )
    group.add_argument(
        "--head_epochs",
        type=int,
        default=4,
        help="Number of local head-only epochs (default: 4)",
    )
    group.add_argument(
        "--body_epochs",
        type=int,
        default=1,
        help="Number of local body-only epochs (default: 1)",
    )
    group.add_argument(
        "--lr_head",
        type=float,
        default=0.01,
        help="Learning rate for local head-only training (default: 0.01)",
    )
    group.add_argument(
        "--lr_body",
        type=float,
        default=0.001,
        help="Learning rate for local body-only training (default: 0.001)",
    )
    group.add_argument(
        "--server_epochs",
        type=int,
        default=10,
        help="Number of server-side TGP training epochs (default: 10)",
    )
    group.add_argument(
        "--server_lr",
        type=float,
        default=0.01,
        help="Learning rate for server-side TGP training (default: 0.01)",
    )
    group.add_argument(
        "--margin_threshold",
        type=float,
        default=5.0,
        help="Margin threshold for TGP training (default: 5.0)",
    )
    return parser


def get_path(args):
    args.file_name = f"{args.name_pre}_{args.lamda_}_{args.head_epochs}_{args.body_epochs}_{args.lr_head}_{args.lr_body}_{args.server_epochs}_{args.server_lr}_{args.margin_threshold}"
    return os.path.join(args.log_path, f"{args.file_name}_{args.times}.log")


class TGP(nn.Module):
    """可训练的全局原型模块 (Trainable Global Prototypes, TGP)"""

    def __init__(self, num_classes, hidden_dim, feature_dim, device):
        super().__init__()
        self.device = device
        self.num_classes = num_classes

        self.embeddings = nn.Embedding(num_classes, feature_dim)
        self.middle = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.ReLU())
        self.fc = nn.Linear(hidden_dim, feature_dim)

    def forward(self, class_ids):
        """
        参数:
            class_ids: 类别索引的张量 (Tensor) 或列表 (List)
        """
        if isinstance(class_ids, list):
            class_ids = torch.tensor(class_ids, device=self.device)
        elif not isinstance(class_ids, torch.Tensor):
            class_ids = torch.tensor(class_ids, device=self.device)

        class_ids = class_ids.to(self.device)

        emb = self.embeddings(class_ids)
        mid = self.middle(emb)
        out = self.fc(mid)
        return out


def proto_cluster(protos_list):
    """从多个客户端中聚合并计算平均原型"""
    proto_clusters = defaultdict(list)
    for protos in protos_list:
        for k, v in protos.items():
            proto_clusters[k].append(v)

    avg_protos = {}
    for k, v in proto_clusters.items():
        protos = torch.stack(v)
        avg_protos[k] = torch.mean(protos, dim=0).detach()

    return avg_protos


def client_worker(params):
    """
    FedTGP-Dec 客户端训练流程：解耦的交替优化。
    Phase 1: 冻结分类头，仅微调特征提取器并对齐全局原型。
    Phase 2: 冻结特征提取器，仅优化分类头。
    """
    (
        _,
        device,
        model_state,
        train_set,
        model_name,
        dataset_name,
        batch_size,
        head_epochs,
        body_epochs,
        lr_head,
        lr_body,
        lamda_,
        global_protos,
        num_class,
        feature_dim,
    ) = params

    # 初始化模型并加载本地持久化状态
    model = get_model(model_name, dataset_name, feature_dim).to(device)
    model.load_state_dict(model_state)
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    # 预处理全局原型：从单一 Tensor 快速搬运到 GPU
    global_protos_tensor = (
        global_protos.to(device) if global_protos is not None else None
    )

    # === Phase 1: Local Body Alignment ===
    # 冻结 classifier，激活 extractor
    for param in model.classifier.parameters():
        param.requires_grad = False
    for param in model.extractor.parameters():
        param.requires_grad = True

    avg_loss_proto = 0.0
    if global_protos_tensor is not None:
        optimizer_body = torch.optim.SGD(model.extractor.parameters(), lr=lr_body)

        total_loss_proto = 0.0
        num_batches_body = 0
        model.train()
        for _ in range(body_epochs):
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                features = model.extractor(x)

                # 恢复语义锚定：计算分类损失以维持特征的判别力
                out = model.classifier(features)
                l_ce = ce_loss(out, y)

                target_protos = global_protos_tensor[y]
                l_proto = mse_loss(features, target_protos)

                # 双重约束：本地决策稳定性 + 全局流形靠拢
                loss = l_ce + lamda_ * l_proto

                optimizer_body.zero_grad()
                loss.backward()
                optimizer_body.step()

                total_loss_proto += l_proto.item()
                num_batches_body += 1

        avg_loss_proto = total_loss_proto / num_batches_body if num_batches_body > 0 else 0.0

    # === Phase 2: Local Head Optimization ===
    # 冻结 extractor，仅更新 classifier
    for param in model.extractor.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

    optimizer_head = torch.optim.SGD(model.classifier.parameters(), lr=lr_head)
    model.train()

    total_loss_ce = 0.0
    num_batches_head = 0

    # 提前准备原型标签 (用于在 Phase 2 中锚定分类器)
    proto_labels = (
        torch.arange(num_class, device=device)
        if global_protos_tensor is not None
        else None
    )

    for _ in range(head_epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            # 1. 本地数据交叉熵损失
            out = model(x)
            loss_ce_local = ce_loss(out, y)

            # 2. 全局原型锚定损失：将全局原型输入分类器并计算 CE
            loss_ce_proto = 0.0
            if global_protos_tensor is not None:
                p_out = model.classifier(global_protos_tensor)
                loss_ce_proto = ce_loss(p_out, proto_labels)

            # 合并损失：在拟合本地数据的同时，保持对全局原型的判别力
            loss = loss_ce_local + loss_ce_proto

            optimizer_head.zero_grad()
            loss.backward()
            optimizer_head.step()

            total_loss_ce += loss.item()
            num_batches_head += 1

    avg_loss_ce = (
        total_loss_ce / num_batches_head if num_batches_head > 0 else 0.0
    )

    # === Phase 3: Recalculate Precise Prototypes ===
    model.eval()
    local_protos_avg = extract_prototypes(
        model, loader, num_class, feature_dim, device
    )

    model_state = {k: v.cpu() for k, v in model.state_dict().items()}
    return [avg_loss_ce, avg_loss_proto, model_state, local_protos_avg]


class Server(BaseServer):
    def __init__(self, args: argparse.Namespace):
        super().__init__(True, args)
        if hasattr(self.model, "feature_dim"):
            self.feature_dim = self.model.feature_dim
        else:
            self.feature_dim = args.feature_dim

        self.tgp = TGP(
            num_classes=self.num_class,
            hidden_dim=self.feature_dim,
            feature_dim=self.feature_dim,
            device=self.device,
        ).to(self.device)

        self.global_protos = None
        self.gap = torch.ones(self.num_class, device=self.device) * 1e9

        # 初始化指标记录列表
        self.loss_proto = []
        self.loss_tgp = []
        self.loss_tgp_ce = []
        self.loss_tgp_mse = []
        self.loss_tgp_ortho = []

    def fit(self):
        num_join_clients = int(self.num_clients * self.args.join_ratio)
        num_join_clients = max(1, num_join_clients)

        for r in range(self.rounds):
            t0 = time.time()
            print(f"\n--- FedTGP Round {r + 1}/{self.rounds} ---")

            selected_clients = np.random.choice(
                self.num_clients, num_join_clients, replace=False
            )
            print(f"Selected clients: {selected_clients}")

            global_protos_cpu = None
            if self.global_protos is not None:
                global_protos_cpu = self.global_protos.cpu()

            # --- Fine-tuning Strategies: Lambda Warm-up & LR Decay ---
            # 1. Lambda Warm-up: 0-5 rounds: 0, 5-30 rounds: linear ramp
            # if r < 5:
            #     current_lamda = 0.0
            # elif r < 30:
            #     current_lamda = self.args.lamda_ * ((r - 5) / 25.0)
            # else:
            #     current_lamda = self.args.lamda_
            current_lamda = self.args.lamda_

            # 2. LR Decay (Commented out by default)
            current_lr_head = self.args.lr_head
            current_lr_body = self.args.lr_body
            # if r >= int(self.rounds * 0.5):
            #     current_lr_head *= 0.5
            #     current_lr_body *= 0.5
            # if r >= int(self.rounds * 0.75):
            #     current_lr_head *= 0.5
            #     current_lr_body *= 0.5

            print(f"Current Lamda: {current_lamda:.4f}")
            # print(f"Current LR Head: {current_lr_head:.4f}, Body: {current_lr_body:.4f}")

            p = [
                [
                    i,
                    self.client_gpu[i],
                    self.clients_state[i],
                    self.train_sets[i],
                    self.args.model,
                    self.args.dataset,
                    self.args.batch_size,
                    self.args.head_epochs,
                    self.args.body_epochs,
                    current_lr_head,
                    current_lr_body,
                    current_lamda,
                    global_protos_cpu,
                    self.num_class,
                    self.args.feature_dim,
                ]
                for i in selected_clients
            ]
            results = self.run_clients(client_worker, p)

            total_loss_ce = 0.0
            total_loss_proto = 0.0
            selected_states = []
            selected_protos = []
            for i in selected_clients:
                l_ce, l_p, client_state, client_proto = results[i]
                total_loss_ce += l_ce
                total_loss_proto += l_p
                # 仅更新本地状态映射，不进行任何全局聚合
                self.clients_state[i] = client_state
                selected_protos.append(client_proto)

            self.loss.append(total_loss_ce / num_join_clients)
            self.loss_proto.append(total_loss_proto / num_join_clients)

            uploaded_protos = []
            for p_dict in selected_protos:
                for label, proto in p_dict.items():
                    uploaded_protos.append((proto.to(self.device), label))

            self.calculate_gap(selected_protos)
            self.update_tgp(uploaded_protos)
            self.evaluate(protos=self.global_protos)

            print(
                f"Model Acc: {self.acc[-1]:.2f}%, Proto Acc: {self.acc_proto[-1]:.2f}%, "
                f"Loss CE: {self.loss[-1]:.4f}, Loss Proto: {self.loss_proto[-1]:.4f}, "
                f"TGP Loss: {self.loss_tgp[-1]:.4f} (Ortho: {self.loss_tgp_ortho[-1]:.4f})"
            )
            self.log_dict(
                r,
                {
                    "train/loss_proto": self.loss_proto[-1],
                    "server/tgp_loss_total": self.loss_tgp[-1],
                    "server/tgp_loss_ce": self.loss_tgp_ce[-1],
                    "server/tgp_loss_mse": self.loss_tgp_mse[-1],
                    "server/tgp_loss_ortho": self.loss_tgp_ortho[-1],
                },
            )
            print(f"Round finished in {time.time() - t0:.2f} seconds")

    def calculate_gap(self, protos_per_client):
        """向量化计算类别间的最小间距 (GPU 加速)"""
        avg_protos_dict = proto_cluster(protos_per_client)

        # 将各类别原型聚合成单一 Tensor [C, d]
        # 注意：此处需处理本地数据中可能缺失的类别
        all_protos = torch.zeros(self.num_class, self.feature_dim, device=self.device)
        for label, proto in avg_protos_dict.items():
            all_protos[label] = proto.to(self.device)

        # 计算两两之间的欧氏距离
        dist_matrix = torch.cdist(all_protos, all_protos, p=2.0)

        # 将对角线(自距离)设为无穷大，防止被误选为最小间距
        dist_matrix.fill_diagonal_(float("inf"))

        # 如果某些类别在当前 Round 缺失，它们在 dist_matrix 中为全 0 行/列
        # 这会导致其最小距离为 0 (与另一个全 0 行的距离)，这是误导性的。
        # 我们应该排除这些类别，或者给它们一个较大的默认值。
        present_labels = list(avg_protos_dict.keys())
        if len(present_labels) < self.num_class:
            # 创建掩码，仅保留存在的类别
            mask = torch.ones(self.num_class, dtype=torch.bool, device=self.device)
            all_labels = torch.arange(self.num_class, device=self.device)
            mask[
                ~torch.isin(all_labels, torch.tensor(present_labels, device=self.device))
            ] = False

            # 对于不存在的类别，将其整行和整列设为 inf，避免被最小间距选中
            dist_matrix[~mask, :] = float("inf")
            dist_matrix[:, ~mask] = float("inf")

        # 获取每个类别的最小间距 [C]
        self.gap = torch.min(dist_matrix, dim=1)[0]

        min_gap = torch.min(self.gap)
        print(f"Min gap: {min_gap:.4f}, Max gap: {torch.max(self.gap):.4f}")

    def update_tgp(self, uploaded_protos):
        self.tgp.train()
        optimizer = torch.optim.SGD(self.tgp.parameters(), lr=self.args.server_lr)

        # 预先生成类别索引张量，避免循环中重复转换
        all_class_ids = torch.arange(self.num_class, device=self.device)
        margin = min(torch.max(self.gap).item(), self.args.margin_threshold)

        for epoch in range(self.args.server_epochs):
            proto_loader = DataLoader(
                uploaded_protos, batch_size=self.args.batch_size, shuffle=True
            )
            epoch_loss = 0.0
            epoch_loss_ce = 0.0
            epoch_loss_mse = 0.0
            epoch_loss_ortho = 0.0
            for proto_batch, labels_batch in proto_loader:
                proto_batch = proto_batch.to(self.device)
                labels_batch = labels_batch.to(self.device, dtype=torch.long)

                # 一次性生成所有类别的原型 logits，避免多次 TGP 前向计算
                proto_gen = self.tgp(all_class_ids)
                # dist = torch.cdist(proto_batch, proto_gen, p=2.0)
                # one_hot = F.one_hot(labels_batch, self.num_class).to(self.device)
                # dist = dist + one_hot * margin
                # loss_ce = ce_loss(-dist, labels_batch)

                logits_ortho = torch.matmul(proto_gen, proto_gen.T) / 0.1
                labels_ortho = torch.arange(self.num_class, device=self.device)
                loss_ortho = ce_loss(logits_ortho, labels_ortho)

                loss_mse = mse_loss(proto_batch, proto_gen[labels_batch])

                # 标准的基于余弦相似度的对比学习 Loss (InfoNCE style)
                # 对 Batch 特征和生成的全局原型进行 L2 归一化
                p_batch_norm = F.normalize(proto_batch, p=2, dim=1)
                p_gen_norm = F.normalize(proto_gen, p=2, dim=1)
                # 计算相似度矩阵并除以温度系数 (默认 0.1)
                logits = torch.matmul(p_batch_norm, p_gen_norm.T) / 0.1
                loss_ce = ce_loss(logits, labels_batch)

                loss = loss_mse + 0.01 * loss_ce + loss_ortho

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                epoch_loss_ce += loss_ce.item()
                epoch_loss_mse += loss_mse.item()
                epoch_loss_ortho += loss_ortho.item()

            avg_loss = epoch_loss / len(proto_loader)
            avg_loss_ce = epoch_loss_ce / len(proto_loader)
            avg_loss_mse = epoch_loss_mse / len(proto_loader)
            avg_loss_ortho = epoch_loss_ortho / len(proto_loader)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(
                    f"  TGP Epoch {epoch+1}/{self.args.server_epochs}, "
                    f"Loss: {avg_loss:.4f} (CE: {avg_loss_ce:.4f}, MSE: {avg_loss_mse:.4f}, Ortho: {avg_loss_ortho:.4f})"
                )

        # 记录每轮最后一轮 TGP 优化的损失均值
        self.loss_tgp.append(avg_loss)
        self.loss_tgp_ce.append(avg_loss_ce)
        self.loss_tgp_mse.append(avg_loss_mse)
        self.loss_tgp_ortho.append(avg_loss_ortho)

        self.tgp.eval()
        with torch.no_grad():
            # 彻底转向 Tensor 型原型：[C, d]
            self.global_protos = self.tgp(all_class_ids).detach()

    def save(self):
        f = {
            "acc": {
                "model": self.acc,
                "proto": self.acc_proto,
            },
            "loss": {
                "model_ce": self.loss,
                "client_proto": self.loss_proto,
                "server_tgp": self.loss_tgp,
                "server_tgp_ce": self.loss_tgp_ce,
                "server_tgp_mse": self.loss_tgp_mse,
                "server_tgp_ortho": self.loss_tgp_ortho,
            },
            "state_dict": {
                "global_model_init": self.model.state_dict(),
                "client_states": self.clients_state,
                "global_prototypes": self.global_protos,
                "aux": {
                    "tgp_net": self.tgp.state_dict(),
                    "gap": self.gap.cpu(),
                },
            },
        }
        self.deal_save(f)
