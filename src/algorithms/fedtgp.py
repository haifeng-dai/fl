import os
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .utils import (
    BaseParams,
    BaseServer,
    dist_contrastive_loss,
    extract_prototypes,
    fmt_num,
    get_model,
    proto_aggregate,
)


def get_path(args):
    args.file_name = f"{args.common_name}_{fmt_num(args.lamda_)}_{fmt_num(args.server_epochs)}_{fmt_num(args.server_lr)}_{fmt_num(args.margin_threshold)}"
    return os.path.join(args.log_path, f"{args.file_name}_{args.cur_time}.log")


@dataclass
class Params(BaseParams):
    lamda_: float
    global_protos: torch.Tensor | None


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


def train(p: Params):
    """
    FedTGP 客户端训练流程（基于原型匹配训练）。
    """
    device = torch.device(p.client_gpu)

    # 初始化模型并加载全局状态
    model = get_model(p.model_name, p.dataset, p.num_class, p.feature_dim).to(device)
    model.load_state_dict(p.model_state)

    # 设置配置与优化器
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=p.lr,
        momentum=p.momentum,
        weight_decay=p.weight_decay,
    )
    loader = DataLoader(p.train_set, batch_size=p.batch_size, shuffle=True)

    model.train()
    total_loss_ce = 0.0
    total_loss_proto = 0.0
    num_batches = 0

    # 预处理全局原型以便在 GPU 上高效访问和计算
    global_protos_tensor = (
        p.global_protos.to(device) if p.global_protos is not None else None
    )

    # 本地模型多轮次训练
    for _ in range(p.epochs):
        for x, y, *_ in loader:
            x, y = x.to(device), y.to(device)
            features = model.extractor(x)
            output = model.classifier(features)
            l_ce = F.cross_entropy(output, y)
            l_proto = torch.tensor(0.0, device=device)

            if global_protos_tensor is not None:
                target_protos = global_protos_tensor[y]
                l_proto = F.mse_loss(features, target_protos)

            loss = l_ce + p.lamda_ * l_proto

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss_ce += l_ce.item()
            total_loss_proto += l_proto.item()
            num_batches += 1

    avg_loss_ce = total_loss_ce / num_batches
    avg_loss_proto = total_loss_proto / num_batches

    # 收集最新的本地原型及样本计数
    local_protos_avg, local_counts = extract_prototypes(
        model, loader, p.num_class, p.feature_dim, device, return_counts=True
    )

    model_state = {k: v.cpu().detach().clone() for k, v in model.state_dict().items()}
    return {
        "loss": avg_loss_ce,
        "loss_proto": avg_loss_proto,
        "state": model_state,
        "protos": local_protos_avg,
        "counts": local_counts,
    }


class Server(BaseServer):
    def __init__(self, args):
        super().__init__(args, pfl=True)
        self.lamda_ = args.lamda_
        self.server_epochs = args.server_epochs
        self.server_lr = args.server_lr
        self.margin_threshold = args.margin_threshold

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

    def fit(self):
        num_join = max(1, int(self.num_clients * self.join_ratio))

        for r in range(self.rounds):
            t0 = time.time()
            print(f"\n--- FedTGP Round {r + 1}/{self.rounds} ---")

            selected = torch.randperm(self.num_clients)[:num_join].tolist()
            print(f"Selected clients: {selected}")

            base_params = self.build_base_params(selected)
            for base in base_params:
                base.model_state = self.clients_state[base.client_id]

            p = [
                Params(
                    **asdict(base),
                    lamda_=self.lamda_,
                    global_protos=self.global_protos.cpu()
                    if self.global_protos is not None
                    else None,
                )
                for base in base_params
            ]
            results = self.run_clients(train, p)

            total_loss_ce = 0.0
            total_loss_proto = 0.0
            selected_states = []
            selected_protos = []
            selected_counts = []
            for cid, res in results.items():
                total_loss_ce += res["loss"]
                total_loss_proto += res["loss_proto"]
                self.clients_state[cid] = res["state"]
                selected_states.append(res["state"])
                selected_protos.append(res["protos"])
                selected_counts.append(res["counts"])

            self.loss.append(total_loss_ce / num_join)
            self.loss_proto.append(total_loss_proto / num_join)

            uploaded_protos = []
            for p_tensor in selected_protos:
                # 遍历所有类别，仅上传非零（即在该客户端存在的）原型
                mask = torch.norm(p_tensor, dim=1) > 1e-8
                indices = torch.where(mask)[0]
                for label in indices:
                    uploaded_protos.append(
                        (p_tensor[label].to(self.device), label.item())
                    )

            self.calculate_gap(selected_protos, selected_counts)
            self.update_tgp(uploaded_protos)
            self.evaluate(protos=self.global_protos)

            print(
                f"Model Acc: {self.acc[-1]:.2f}%, Proto Acc: {self.acc_proto[-1]:.2f}%, "
                f"Loss CE: {self.loss[-1]:.4f}, Loss Proto: {self.loss_proto[-1]:.4f}"
            )
            print(f"Round finished in {time.time() - t0:.2f} seconds")

    def calculate_gap(self, protos_per_client, counts_per_client):
        """向量化计算类别间的最小间距 (GPU 加速)"""
        # 使用统一的张量聚合函数获取按样本计数的平均原型，并用旧原型补全缺失类别
        all_protos = proto_aggregate(
            protos_per_client,
            local_counts_list=counts_per_client,
            old_global_protos=self.global_protos,
        ).to(self.device)
        dist_matrix = torch.cdist(all_protos, all_protos, p=2.0)
        dist_matrix.fill_diagonal_(float("inf"))
        self.gap = torch.min(dist_matrix, dim=1)[0]

        # 处理全零行带来的无效距离
        mask = torch.norm(all_protos, dim=1) > 1e-8
        min_gap = torch.min(self.gap[mask]) if mask.any() else torch.tensor(0.0)
        self.gap[~mask] = min_gap

        print(f"Min gap: {min_gap:.4f}, Max gap: {torch.max(self.gap):.4f}")

    def update_tgp(self, uploaded_protos):
        self.tgp.train()
        optimizer = torch.optim.SGD(self.tgp.parameters(), lr=self.server_lr)

        for _ in range(self.server_epochs):
            proto_loader = DataLoader(
                uploaded_protos, batch_size=self.batch_size, shuffle=True
            )
            for proto_batch, labels_batch in proto_loader:
                proto_batch = proto_batch.to(self.device)
                labels_batch = labels_batch.to(self.device, dtype=torch.long)
                proto_gen = self.tgp(list(range(self.num_class)))
                margin = min(torch.max(self.gap).item(), self.margin_threshold)
                loss = dist_contrastive_loss(
                    proto_batch, proto_gen, labels_batch, margin=margin
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.tgp.eval()
        with torch.no_grad():
            all_class_ids = torch.arange(self.num_class, device=self.device)
            self.global_protos = self.tgp(all_class_ids).detach().cpu()

    def save(self):
        metrics = {
            "acc": self.acc,
            "acc_p": self.acc_proto,
            "loss": self.loss,
            "loss_p": self.loss_proto,
        }
        params = {
            "global": self.model.state_dict(),
            "client": self.clients_state,
            "proto": self.global_protos,
            "aux": {"tgp": self.tgp.state_dict(), "gap": self.gap.cpu()},
        }
        self.deal_save(metrics, params)
