import os
import time

import numpy as np
import torch

from .utils import (
    BaseServer,
    ce_loss,
    cos_contrastive_loss,
    extract_prototypes,
    get_model,
    proto_aggregate,
)


def get_path(args):
    args.file_name = f"{args.common_name}"
    return os.path.join(args.log_path, f"{args.file_name}_{args.cur_time}.log")


def client_worker(params):
    (
        _,
        device,
        model_state,
        global_protos,
        alpha,
        train_set,
        model_name,
        dataset_name,
        lr,
        batch_size,
        epochs,
        num_classes,
        feature_dim,
    ) = params

    # 1. 初始化模型并加载全局状态
    model = get_model(model_name, dataset_name, feature_dim).to(device)
    model.load_state_dict(model_state)
    global_protos = global_protos.data.clone().to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    # 2. 本地模型多轮次训练
    total_loss = 0.0
    num_batches = 0

    model.train()
    for _ in range(epochs):
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            features = model.extractor(data)
            output = model.classifier(features)
            loss_ce = ce_loss(output, target)

            # 基于原型的对比损失 (Prototypical Contrastive Loss)
            loss_con = cos_contrastive_loss(
                features, global_protos, target, temperature=1.0
            )

            loss = (1 - alpha) * loss_ce + alpha * loss_con
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

    # 3. 提取本地原型及样本计数
    local_protos, local_counts = extract_prototypes(
        model, loader, num_classes, feature_dim, device, return_counts=True
    )

    return {
        "loss": total_loss / num_batches,
        "state": {k: v.cpu().detach().clone() for k, v in model.state_dict().items()},
        "protos": local_protos,
        "counts": local_counts,
    }


class Server(BaseServer):
    def __init__(self, args):
        super().__init__(False, args)
        self.global_protos = torch.zeros((self.num_class, self.args.feature_dim))

    def fit(self):
        num_join_clients = int(self.num_clients * self.args.join_ratio)
        num_join_clients = max(1, num_join_clients)

        for r in range(self.rounds):
            t0 = time.time()
            print(f"\n--- FedProc Round {r + 1}/{self.rounds} ---")

            selected_clients = np.random.choice(
                self.num_clients, num_join_clients, replace=False
            )
            print(f"Selected clients: {selected_clients}")

            p = [
                [
                    i,
                    self.client_gpu[i],
                    self.model.state_dict(),
                    self.global_protos,
                    1.0 - (r / self.rounds),  # alpha = 1 - r/rounds
                    self.train_sets[i],
                    self.args.model,
                    self.args.dataset,
                    self.args.lr,
                    self.args.batch_size,
                    self.args.epochs,
                    self.num_class,
                    self.args.feature_dim,
                ]
                for i in selected_clients
            ]
            results = self.run_clients(client_worker, p)

            total_loss = 0.0
            selected_states = []
            all_local_protos = []
            all_local_counts = []
            for _, res in results.items():
                total_loss += res["loss"]
                selected_states.append(res["state"])
                all_local_protos.append(res["protos"])
                all_local_counts.append(res["counts"])
            self.loss.append(total_loss / num_join_clients)

            # 聚合模型参数
            self.aggregate(selected_states)
            # 聚合原型向量：按样本计数加权
            self.global_protos = proto_aggregate(
                all_local_protos,
                local_counts_list=all_local_counts,
                old_global_protos=self.global_protos,
            ).to(self.device)

            self.evaluate()
            print(
                f"Global Accuracy: {self.acc[-1]:.2f}%, Avg Loss: {self.loss[-1]:.4f}"
            )
            print(f"Round finished in {time.time() - t0:.2f} seconds")

    def save(self):
        f = {
            "acc": self.acc,
            "loss": self.loss,
            "state_dict": {
                "global": self.model.state_dict(),
                "proto": self.global_protos,
            },
        }
        self.deal_save(f)
