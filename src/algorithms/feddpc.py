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
    extract_prototypes,
    fmt_num,
    get_model,
    orthogonality_loss,
)


def get_path(args):
    args.file_name = f"{args.common_name}_{fmt_num(args.lamda_)}_{fmt_num(args.head_epochs)}_{fmt_num(args.body_epochs)}_{fmt_num(args.lr_head)}_{fmt_num(args.lr_body)}_{fmt_num(args.server_epochs)}_{fmt_num(args.server_lr)}_{fmt_num(args.lambda_p)}_{fmt_num(args.lambda_acl)}"
    return os.path.join(args.log_path, f"{args.file_name}_{args.cur_time}.log")


@dataclass
class Params(BaseParams):
    head_epochs: int
    body_epochs: int
    lr_head: float
    lr_body: float
    lamda_: float
    global_protos: torch.Tensor | None
    lambda_p: float


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


def train(p: Params):
    """
    FedDPC 客户端训练流程：解耦的交替优化。
    Phase 1: 冻结特征提取器，仅优化分类头。
    Phase 2: 冻结分类头，仅微调特征提取器并对齐全局原型。
    """
    device = torch.device(p.client_gpu)

    # 初始化模型并加载本地持久化状态
    model = get_model(p.model_name, p.dataset, p.num_class, p.feature_dim).to(device)
    model.load_state_dict(p.model_state)
    loader = DataLoader(p.train_set, batch_size=p.batch_size, shuffle=True)

    # 预处理全局原型：从单一 Tensor 快速搬运到 GPU
    global_protos_tensor = (
        p.global_protos.to(device) if p.global_protos is not None else None
    )

    # === Phase 1: Local Head Optimization ===
    # 冻结 extractor，仅更新 classifier
    for param in model.extractor.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

    optimizer_head = torch.optim.SGD(model.classifier.parameters(), lr=p.lr_head)
    model.train()

    total_loss_ce = 0.0
    num_batches_head = 0
    # 提前准备原型标签 (用于在 Phase 1 中锚定分类器)
    proto_labels = (
        torch.arange(p.num_class, device=device)
        if global_protos_tensor is not None
        else None
    )

    for _ in range(p.head_epochs):
        for x, y, *_ in loader:
            x, y = x.to(device), y.to(device)

            # 1. 本地数据交叉熵损失
            out = model(x)
            loss_ce_local = F.cross_entropy(out, y)

            # 2. 全局原型锚定损失：将全局原型输入分类器并计算 CE
            loss_ce_proto = 0.0
            if global_protos_tensor is not None:
                p_out = model.classifier(global_protos_tensor)
                loss_ce_proto = F.cross_entropy(p_out, proto_labels)

            # 合并损失：在拟合本地数据的同时，保持对全局原型的判别力
            loss = loss_ce_local + p.lambda_p * loss_ce_proto

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
        optimizer_body = torch.optim.SGD(model.extractor.parameters(), lr=p.lr_body)

        total_loss_proto = 0.0
        num_batches_body = 0
        model.train()

        for _ in range(p.body_epochs):
            for x, y, *_ in loader:
                x, y = x.to(device), y.to(device)

                # 仅计算特征与对应类别全局原型的 MSE 损失
                features = model.extractor(x)
                target_protos = global_protos_tensor[y]
                loss_proto = F.mse_loss(features, target_protos)

                optimizer_body.zero_grad()
                loss_proto.backward()
                optimizer_body.step()

                total_loss_proto += loss_proto.item()
                num_batches_body += 1

        avg_loss_proto = (
            total_loss_proto / num_batches_body if num_batches_body > 0 else 0.0
        )
    else:
        avg_loss_proto = 0.0

    # 提取本地原型用于在服务器端指导全局 PLN 学习
    local_protos = extract_prototypes(
        model, loader, p.num_class, p.feature_dim, device, return_counts=False
    )

    model_state = {k: v.cpu().detach().clone() for k, v in model.state_dict().items()}
    return {
        "loss": avg_loss_ce,
        "loss_proto": avg_loss_proto,
        "state": model_state,
        "protos": local_protos.cpu().detach().clone(),
    }


class Server(BaseServer):
    def __init__(self, args):
        super().__init__(args, pfl=True)
        self.head_epochs = args.head_epochs
        self.body_epochs = args.body_epochs
        self.lr_head = args.lr_head
        self.lr_body = args.lr_body
        self.lamda_ = args.lamda_
        self.lambda_p = args.lambda_p
        self.lambda_acl = args.lambda_acl
        self.server_epochs = args.server_epochs
        self.server_lr = args.server_lr

        self.pln = PLN(
            num_classes=self.num_class,
            hidden_dim=self.feature_dim,
            feature_dim=self.feature_dim,
            device=self.device,
        ).to(self.device)
        self.global_protos = None

        # 初始化指标记录列表
        self.loss_proto = []
        self.loss_pln = []
        self.loss_pln_mse = []
        self.loss_pln_ortho = []

    def fit(self):
        num_join = max(1, int(self.num_clients * self.join_ratio))

        for r in range(self.rounds):
            t0 = time.time()
            print(f"\n--- FedDPC Round {r + 1}/{self.rounds} ---")

            selected = torch.randperm(self.num_clients)[:num_join].tolist()
            base_params = self.build_base_params(selected)
            for base in base_params:
                base.model_state = self.clients_state[base.client_id]

            p = [
                Params(
                    **asdict(base),
                    head_epochs=self.head_epochs,
                    body_epochs=self.body_epochs,
                    lr_head=self.lr_head,
                    lr_body=self.lr_body,
                    lamda_=self.lamda_,
                    global_protos=self.global_protos.cpu() if self.global_protos is not None else None,
                    lambda_p=self.lambda_p,
                )
                for base in base_params
            ]
            results = self.run_clients(train, p)

            total_loss_ce = 0.0
            total_loss_proto = 0.0
            selected_protos = []
            for cid, res in results.items():
                total_loss_ce += res["loss"]
                total_loss_proto += res["loss_proto"]
                # 仅更新本地状态映射，不进行任何全局聚合
                self.clients_state[cid] = res["state"]
                selected_protos.append(res["protos"])

            self.loss.append(total_loss_ce / num_join)
            self.loss_proto.append(total_loss_proto / num_join)

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

            self.update_pln(uploaded_protos)
            self.evaluate(protos=self.global_protos)

            print(
                f"Model Acc: {self.acc[-1]:.2f}%, Proto Acc: {self.acc_proto[-1]:.2f}%, "
                f"Loss CE: {self.loss[-1]:.4f}, Loss Proto: {self.loss_proto[-1]:.4f}, "
                f"PLN Loss: {self.loss_pln[-1]:.4f} (Ortho: {self.loss_pln_ortho[-1]:.4f})"
            )
            print(f"Round finished in {time.time() - t0:.2f} seconds")

    def update_pln(self, uploaded_protos):
        self.pln.train()
        optimizer = torch.optim.SGD(self.pln.parameters(), lr=self.server_lr)

        # 预先生成类别索引张量，避免循环中重复转换
        all_class_ids = torch.arange(self.num_class, device=self.device)
        proto_loader = DataLoader(
            uploaded_protos, batch_size=self.batch_size, shuffle=True
        )

        epoch_loss = 0.0
        epoch_loss_mse = 0.0
        epoch_loss_ortho = 0.0
        num_batches = 0
        for _ in range(self.server_epochs):
            for proto_batch, labels_batch in proto_loader:
                proto_batch = proto_batch.to(self.device)
                labels_batch = labels_batch.to(self.device, dtype=torch.long)

                # 一次性生成所有类别的原型 logits，避免多次 PLN 前向计算
                proto_gen = self.pln(all_class_ids)
                loss_ortho = orthogonality_loss(proto_gen)

                loss_mse = F.mse_loss(proto_batch, proto_gen[labels_batch])

                loss = loss_mse + self.lambda_acl * loss_ortho

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
        metrics = {
            "acc": self.acc,
            "acc_p": self.acc_proto,
            "loss": self.loss,
            "loss_p": self.loss_proto,
            "loss_pln": self.loss_pln,
            "loss_pln_mse": self.loss_pln_mse,
            "loss_pln_ortho": self.loss_pln_ortho,
        }
        params = {
            "global_model_init": self.model.state_dict(),
            "client_states": self.clients_state,
            "global_prototypes": self.global_protos,
            "aux": {"pln_net": self.pln.state_dict()},
        }
        self.deal_save(metrics, params)
