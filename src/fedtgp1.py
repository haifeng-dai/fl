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
        default=5,
        help="Number of local head-only epochs (default: 5)",
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
        default=1.0,
        help="Margin threshold for TGP training (default: 1.0)",
    )
    return parser


def get_path(args):
    args.file_name = f"{args.name_pre}_{args.lamda_}_{args.head_epochs}_{args.body_epochs}_{args.lr_head}_{args.lr_body}"
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
    ) = params

    # 初始化模型并加载本地持久化状态
    model = get_model(model_name, dataset_name, feature_dim).to(device)
    model.load_state_dict(model_state)
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    # 预处理全局原型以便在 GPU 上高效访问
    global_protos_tensor = None
    if global_protos is not None:
        first_proto = next(iter(global_protos.values()))
        feat_dim = first_proto.shape[0]
        global_protos_tensor = torch.zeros(num_class, feat_dim, device=device)
        for label, proto in global_protos.items():
            global_protos_tensor[label] = proto.to(device)

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
    for _ in range(head_epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = ce_loss(out, y)
            optimizer_head.zero_grad()
            loss.backward()
            optimizer_head.step()
            total_loss_ce += loss.item()
            num_batches_head += 1

    avg_loss_ce = total_loss_ce / num_batches_head if num_batches_head > 0 else 0.0

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

        avg_loss_proto = total_loss_proto / num_batches_body if num_batches_body > 0 else 0.0
    else:
        avg_loss_proto = 0.0

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
                global_protos_cpu = {k: v.cpu() for k, v in self.global_protos.items()}

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
                f"TGP Loss: {self.loss_tgp[-1]:.4f}"
            )
            print(f"Round finished in {time.time() - t0:.2f} seconds")

    def calculate_gap(self, protos_per_client):
        self.gap = torch.ones(self.num_class, device=self.device) * 1e9
        avg_protos = proto_cluster(protos_per_client)

        for k1 in avg_protos.keys():
            for k2 in avg_protos.keys():
                if k1 > k2:
                    dis = torch.norm(
                        avg_protos[k1].to(self.device) - avg_protos[k2].to(self.device),
                        p=2,
                    )
                    self.gap[k1] = torch.min(self.gap[k1], dis)
                    self.gap[k2] = torch.min(self.gap[k2], dis)

        min_gap = torch.min(self.gap)
        for i in range(len(self.gap)):
            if self.gap[i] > 1e8:
                self.gap[i] = min_gap
        print(f"Min gap: {min_gap:.4f}, Max gap: {torch.max(self.gap):.4f}")

    def update_tgp(self, uploaded_protos):
        self.tgp.train()
        optimizer = torch.optim.SGD(self.tgp.parameters(), lr=self.args.server_lr)

        for epoch in range(self.args.server_epochs):
            proto_loader = DataLoader(
                uploaded_protos, batch_size=self.args.batch_size, shuffle=True
            )
            epoch_loss = 0.0
            for proto_batch, labels_batch in proto_loader:
                proto_batch = proto_batch.to(self.device)
                labels_batch = labels_batch.to(self.device, dtype=torch.long)
                proto_gen = self.tgp(list(range(self.num_class)))
                dist = torch.cdist(proto_batch, proto_gen, p=2.0)
                one_hot = F.one_hot(labels_batch, self.num_class).to(self.device)
                margin = min(torch.max(self.gap).item(), self.args.margin_threshold)
                dist = dist + one_hot * margin

                loss = ce_loss(-dist, labels_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(proto_loader)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  TGP Epoch {epoch+1}/{self.args.server_epochs}, Loss: {avg_loss:.4f}")
        
        # 记录每轮最后一轮 TGP 优化的损失均值
        self.loss_tgp.append(avg_loss)

        self.tgp.eval()
        self.global_protos = {}
        with torch.no_grad():
            for class_id in range(self.num_class):
                self.global_protos[class_id] = self.tgp(
                    torch.tensor(class_id, device=self.device)
                ).data.clone()

    def save(self):
        f = {
            "acc": {
                "model": self.acc,
                "proto": self.acc_proto,
            },
            "loss": {
                "model_ce": self.loss,
                "client_proto": self.loss_proto,
                "server_tgp": self.loss_tgp,  # 新增：记录服务端原型优化损失
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
