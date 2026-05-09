import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .utils import (
    BaseServer,
    ce_loss,
    extract_prototypes,
    get_model,
    mse_loss,
    orthogonality_loss,
    proto_aggregate,
)


def get_path(args):
    args.file_name = f"{args.common_name}_{args.lamda_}_{args.head_epochs}_{args.body_epochs}_{args.lr_head}_{args.lr_body}_{args.server_epochs}_{args.server_lr}_{args.lambda_p}_{args.lambda_acl}"
    return os.path.join(args.log_path, f"{args.file_name}_{args.cur_time}.log")


class PLN(nn.Module):
    """原型学习网络 (Prototype Learning Network, PLN)"""

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


def client_worker(params):
    """
    FedDPC 客户端训练流程：解耦的交替优化。
    Phase 1: 冻结特征提取器，仅优化分类头。
    Phase 2: 冻结分类头，仅微调特征提取器并对齐全局原型。
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
        lambda_p,
    ) = params

    # 初始化模型并加载本地持久化状态
    model = get_model(model_name, dataset_name, feature_dim).to(device)
    model.load_state_dict(model_state)
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    # 预处理全局原型：从单一 Tensor 快速搬运到 GPU
    global_protos_tensor = (
        global_protos.to(device) if global_protos is not None else None
    )

    # === Phase 1: Local Head Optimization ===
    # 冻结 extractor，仅更新 classifier
    for param in model.extractor.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

    optimizer_head = torch.optim.SGD(model.classifier.parameters(), lr=lr_head)
    model.train()

    total_loss_ce = 0.0
    num_batches_head = 0
    # 提前准备原型标签 (用于在 Phase 1 中锚定分类器)
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
            loss = loss_ce_local + lambda_p * loss_ce_proto

            optimizer_head.zero_grad()
            loss.backward()
            optimizer_head.step()

            total_loss_ce += loss.item()
            num_batches_head += 1

    avg_loss_ce = total_loss_ce / num_batches_head

    # === Phase 2: Local Body Alignment ===
    # 冻结 classifier，激活 extractor
    for param in model.classifier.parameters():
        param.requires_grad = False
    for param in model.extractor.parameters():
        param.requires_grad = True

    if global_protos_tensor is not None:
        optimizer_body = torch.optim.SGD(model.extractor.parameters(), lr=lr_body)

        total_loss_proto = 0.0
        num_batches_body = 0
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

        avg_loss_proto = total_loss_proto / num_batches_body
    else:
        avg_loss_proto = 0.0

    # === Phase 3: Recalculate Precise Prototypes ===
    model.eval()
    local_protos_avg = extract_prototypes(model, loader, num_class, feature_dim, device)

    model_state = {k: v.cpu().detach().clone() for k, v in model.state_dict().items()}
    return {
        "loss": avg_loss_ce,
        "loss_proto": avg_loss_proto,
        "state": model_state,
        "protos": local_protos_avg,
    }


class Server(BaseServer):
    def __init__(self, args):
        super().__init__(True, args)
        if hasattr(self.model, "feature_dim"):
            self.feature_dim = self.model.feature_dim
        else:
            self.feature_dim = args.feature_dim

        self.pln = PLN(
            num_classes=self.num_class,
            hidden_dim=self.feature_dim,
            feature_dim=self.feature_dim,
            device=self.device,
        ).to(self.device)

        self.global_protos = None
        self.gap = torch.ones(self.num_class, device=self.device) * 1e9

        # 初始化指标记录列表
        self.loss_proto = []
        self.loss_pln = []
        self.loss_pln_mse = []
        self.loss_pln_ortho = []

    def fit(self):
        num_join_clients = int(self.num_clients * self.args.join_ratio)
        num_join_clients = max(1, num_join_clients)

        for r in range(self.rounds):
            t0 = time.time()
            print(f"\n--- FedDPC Round {r + 1}/{self.rounds} ---")

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
                    self.args.model,
                    self.args.dataset,
                    self.args.batch_size,
                    self.args.head_epochs,
                    self.args.body_epochs,
                    self.args.lr_head,
                    self.args.lr_body,
                    self.args.lamda_,
                    self.global_protos.cpu(),
                    self.num_class,
                    self.args.feature_dim,
                    self.args.lambda_p,
                ]
                for i in selected_clients
            ]
            results = self.run_clients(client_worker, p)

            total_loss_ce = 0.0
            total_loss_proto = 0.0
            selected_protos = []
            for cid, res in results.items():
                total_loss_ce += res["loss"]
                total_loss_proto += res["loss_proto"]
                # 仅更新本地状态映射，不进行任何全局聚合
                self.clients_state[cid] = res["state"]
                selected_protos.append(res["protos"])

            self.loss.append(total_loss_ce / num_join_clients)
            self.loss_proto.append(total_loss_proto / num_join_clients)

            uploaded_protos = []
            for p_tensor in selected_protos:
                # 遍历所有类别，仅上传非零（即在该客户端存在的）原型
                # p_tensor 形状为 [num_class, feature_dim]
                mask = torch.norm(p_tensor, dim=1) > 1e-8
                indices = torch.where(mask)[0]
                for label in indices:
                    uploaded_protos.append(
                        (p_tensor[label].to(self.device), label.item())
                    )

            self.calculate_gap(selected_protos)
            self.update_pln(uploaded_protos)
            self.evaluate(protos=self.global_protos)

            print(
                f"Model Acc: {self.acc[-1]:.2f}%, Proto Acc: {self.acc_proto[-1]:.2f}%, "
                f"Loss CE: {self.loss[-1]:.4f}, Loss Proto: {self.loss_proto[-1]:.4f}, "
                f"PLN Loss: {self.loss_pln[-1]:.4f} (Ortho: {self.loss_pln_ortho[-1]:.4f})"
            )
            print(f"Round finished in {time.time() - t0:.2f} seconds")

    def calculate_gap(self, protos_per_client):
        """向量化计算类别间的最小间距 (GPU 加速)"""
        # 使用统一的张量聚合函数获取平均原型
        all_protos = proto_aggregate(protos_per_client).to(self.device)

        # 计算两两之间的欧氏距离
        dist_matrix = torch.cdist(all_protos, all_protos, p=2.0)

        # 将对角线(自距离)设为无穷大，防止被误选为最小间距
        dist_matrix.fill_diagonal_(float("inf"))

        # 获取每个类别的最小间距 [C]
        self.gap = torch.min(dist_matrix, dim=1)[0]

        min_gap = torch.min(self.gap)
        print(f"Min gap: {min_gap:.4f}, Max gap: {torch.max(self.gap):.4f}")

    def update_pln(self, uploaded_protos):
        self.pln.train()
        optimizer = torch.optim.SGD(self.pln.parameters(), lr=self.args.server_lr)

        # 预先生成类别索引张量，避免循环中重复转换
        all_class_ids = torch.arange(self.num_class, device=self.device)

        epoch_loss = 0.0
        epoch_loss_mse = 0.0
        epoch_loss_ortho = 0.0
        num_batches = 0
        for _ in range(self.args.server_epochs):
            proto_loader = DataLoader(
                uploaded_protos, batch_size=self.args.batch_size, shuffle=True
            )
            for proto_batch, labels_batch in proto_loader:
                proto_batch = proto_batch.to(self.device)
                labels_batch = labels_batch.to(self.device, dtype=torch.long)

                # 一次性生成所有类别的原型 logits，避免多次 PLN 前向计算
                proto_gen = self.pln(all_class_ids)
                loss_ortho = orthogonality_loss(proto_gen)

                loss_mse = mse_loss(proto_batch, proto_gen[labels_batch])

                loss = loss_mse + self.args.lambda_acl * loss_ortho

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                num_batches += 1
                epoch_loss += loss.item()
                epoch_loss_mse += loss_mse.item()
                epoch_loss_ortho += loss_ortho.item()

        avg_loss = epoch_loss / num_batches
        avg_loss_mse = epoch_loss_mse / num_batches
        avg_loss_ortho = epoch_loss_ortho / num_batches

        print(
            f"Loss: {avg_loss:.4f} (MSE: {avg_loss_mse:.4f}, Ortho: {avg_loss_ortho:.4f})"
        )

        # 记录每轮最后一轮 PLN 优化的损失均值
        self.loss_pln.append(avg_loss)
        self.loss_pln_mse.append(avg_loss_mse)
        self.loss_pln_ortho.append(avg_loss_ortho)

        self.pln.eval()
        with torch.no_grad():
            self.global_protos = self.pln(all_class_ids).cpu().clone()

    def save(self):
        f = {
            "acc": {
                "model": self.acc,
                "proto": self.acc_proto,
            },
            "loss": {
                "model_ce": self.loss,
                "client_proto": self.loss_proto,
                "server_pln": self.loss_pln,
                "server_pln_mse": self.loss_pln_mse,
                "server_pln_ortho": self.loss_pln_ortho,
            },
            "state_dict": {
                "global_model_init": self.model.state_dict(),
                "client_states": self.clients_state,
                "global_prototypes": self.global_protos,
                "aux": {
                    "pln_net": self.pln.state_dict(),
                    "gap": self.gap.cpu(),
                },
            },
        }
        self.deal_save(f)
