import argparse
import os
import time
from collections import defaultdict

import numpy as np
import torch

from .utils import (
    BaseServer,
    ce_loss,
    get_model,
    mse_loss,
    extract_prototypes,
)


def add_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("FedProto Specific Arguments")
    group.add_argument(
        "--mu",
        type=float,
        default=0.1,
        help="Weight for prototype consistency loss",
    )
    return parser


def get_path(args):
    args.file_name = f"{args.name_pre}_{args.mu}"
    return os.path.join(args.log_path, f"{args.file_name}_{args.times}.log")


def client_worker(params):
    (
        _,
        device,
        local_model_state,
        train_set,
        model_name,
        dataset_name,
        lr,
        batch_size,
        epochs,
        num_classes,
        feature_dim,
        mu,
        global_protos,
    ) = params

    model = get_model(model_name, dataset_name, feature_dim).to(device)
    model.load_state_dict(local_model_state)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    global_protos_tensor = None
    if global_protos is not None:
        first_proto = next(iter(global_protos.values()))
        feat_dim = first_proto.shape[0]
        global_protos_tensor = torch.zeros(num_classes, feat_dim, device=device)
        for label, proto in global_protos.items():
            global_protos_tensor[label] = proto.to(device)

    total_loss = 0.0
    num_batches = 0
    model.train()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            feature = model.extractor(x)
            logits = model.classifier(feature)
            loss_ce = ce_loss(logits, y)

            if global_protos_tensor is not None:
                target_protos = global_protos_tensor[y]
                loss_proto = mse_loss(feature, target_protos)
                loss = loss_ce + mu * loss_proto
            else:
                loss = loss_ce

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

    local_protos = extract_prototypes(
        model, loader, num_classes, feature_dim, device
    )

    model_state = {k: v.cpu() for k, v in model.state_dict().items()}
    return [avg_loss, model_state, local_protos]


class Server(BaseServer):
    def __init__(self, args: argparse.Namespace):
        super().__init__(True, args)

        self.global_protos = None

    def fit(self):
        num_join_clients = int(self.num_clients * self.args.join_ratio)
        num_join_clients = max(1, num_join_clients)

        for r in range(self.rounds):
            t0 = time.time()
            print(f"\n--- FedProto Round {r + 1}/{self.rounds} ---")

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
                    self.args.lr,
                    self.args.batch_size,
                    self.args.epochs,
                    self.num_class,
                    self.args.feature_dim,
                    self.args.mu,
                    global_protos_cpu,
                ]
                for i in selected_clients
            ]
            results = self.run_clients(client_worker, p)

            total_loss = 0.0
            selected_protos = []
            for i in selected_clients:
                client_loss, client_state, client_proto = results[i]
                total_loss += client_loss
                self.clients_state[i] = client_state
                selected_protos.append(client_proto)
            self.loss.append(total_loss / num_join_clients)

            self.global_protos = self.aggregate_protos(selected_protos)
            self.evaluate(protos=self.global_protos)

            print(
                f"Global Accuracy: {self.acc[-1]:.2f}%, Proto Accuracy: {self.acc_proto[-1]:.2f}%, Avg Loss: {self.loss[-1]:.4f}"
            )
            print(f"Round finished in {time.time() - t0:.2f} seconds")

    def aggregate_protos(self, all_local_protos):
        """按类别汇总聚合各个客户端上传的本地原型，与 Server 对象进行强绑定。"""
        proto_clusters = defaultdict(list)
        for protos in all_local_protos:
            for k, v in protos.items():
                proto_clusters[k].append(v)

        avg_protos = {}
        # 计算本轮参与了更新的类别的对应平均原型
        for k, v in proto_clusters.items():
            protos = torch.stack(v)
            avg_protos[k] = torch.mean(protos, dim=0).detach()

        # 对于本轮没有任何客户端上传的新类别原型，保留老旧历史状态以防遗失 (和 FedProc 逻辑对齐)
        if self.global_protos is not None:
            for k, v in self.global_protos.items():
                if k not in avg_protos:
                    avg_protos[k] = v

        return avg_protos

    def save(self):
        f = {
            "acc": {"model": self.acc, "proto": self.acc_proto},
            "loss": self.loss,
            "state_dict": {
                "client": self.clients_state,
                "proto": self.global_protos,
            },
        }
        self.deal_save(f)
