import time
from dataclasses import asdict

import torch
from torch.utils.data import DataLoader

from .fedtgp import TGP, Params, get_path, train
from .utils import (
    check_losses,
    BaseServer,
    proto_aggregate,
)
from .utils.loss import dist_contrastive_loss


class Server(BaseServer):
    def __init__(self, args):
        # pfl=False 全局模型体系
        super().__init__(args)
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
        self.loss_proto = []

    def fit(self):
        num_join = max(1, int(self.num_clients * self.join_ratio))

        for r in range(self.rounds):
            t0 = time.time()
            print(f"\n--- FedTGP_G Round {r + 1}/{self.rounds} ---")

            selected = torch.randperm(self.num_clients)[:num_join].tolist()
            print(f"Selected clients: {selected}")

            # 默认将全局模型 self.model.state_dict() 分发给各客户端
            base_params = self.build_base_params(selected)

            p = [
                Params(
                    **asdict(base),
                    lamda_=self.lamda_,
                    global_protos=(
                        self.global_protos.cpu()
                        if self.global_protos is not None
                        else None
                    ),
                )
                for base in base_params
            ]
            results = self.run_clients(train, p)

            total_loss_ce = 0.0
            total_loss_proto = 0.0
            selected_states = []
            selected_protos = []
            selected_counts = []
            current_weights = []
            for cid, res in results.items():
                total_loss_ce += res["loss"]
                total_loss_proto += res["loss_proto"]
                selected_states.append(res["state"])
                selected_protos.append(res["protos"])
                selected_counts.append(res["counts"])
                current_weights.append(self.weights[cid])

            self.loss.append(total_loss_ce / num_join)
            self.loss_proto.append(total_loss_proto / num_join)

            # 1. 聚合全局模型参数 (FedAvg)
            sum_weights = sum(current_weights)
            norm_weights = [w / sum_weights for w in current_weights]
            self.aggregate(selected_states, weights=norm_weights)

            # 2. 收集上传原型并优化服务端 TGP 模块
            uploaded_protos = []
            for p_tensor in selected_protos:
                mask = torch.norm(p_tensor, dim=1) > 1e-8
                indices = torch.where(mask)[0]
                for label in indices:
                    uploaded_protos.append(
                        (p_tensor[label].to(self.device), label.item())
                    )

            self.calculate_gap(selected_protos, selected_counts)
            self.update_tgp(uploaded_protos)

            # 3. 评估全局模型准确率及 TGP 生成原型的匹配精度
            self.evaluate(protos=self.global_protos)

            print(
                f"Model Acc: {self.acc[-1]:.2f}%, Proto Acc: {self.acc_proto[-1]:.2f}%, "
                f"Loss CE: {self.loss[-1]:.4f}, Loss Proto: {self.loss_proto[-1]:.4f}"
            )
            print(f"Round finished in {time.time() - t0:.2f} seconds")

    def calculate_gap(self, protos_per_client, counts_per_client):
        """向量化计算类别间的最小间距 (GPU 加速)"""
        all_protos = proto_aggregate(
            protos_per_client,
            local_counts_list=counts_per_client,
            old_global_protos=self.global_protos,
        ).to(self.device)
        dist_matrix = torch.cdist(all_protos, all_protos, p=2.0)
        dist_matrix.fill_diagonal_(float("inf"))
        self.gap = torch.min(dist_matrix, dim=1)[0]

        mask = torch.norm(all_protos, dim=1) > 1e-8
        min_gap = torch.min(self.gap[mask]) if mask.any() else torch.tensor(0.0)
        self.gap[~mask] = min_gap

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
                check_losses(loss, locals())
                loss.backward()
                optimizer.step()

        self.tgp.eval()
        with torch.no_grad():
            all_class_ids = torch.arange(self.num_class, device=self.device)
            self.global_protos = self.tgp(all_class_ids).detach().cpu()

    def save(self):
        metrics = {
            "acc": self.acc,
            "acc_proto": self.acc_proto,
            "loss": self.loss,
            "loss_proto": self.loss_proto,
        }
        params = {
            "global": self.model.state_dict(),
            "proto": self.global_protos,
            "aux": {"tgp": self.tgp.state_dict(), "gap": self.gap.cpu()},
        }
        self.deal_save(metrics, params)
