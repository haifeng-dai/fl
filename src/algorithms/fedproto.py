import os
import time

import numpy as np
import torch

from .utils import (
    BaseServer,
    ce_loss,
    extract_prototypes,
    get_model,
    mse_loss,
    proto_aggregate,
)


def get_path(args):
    args.file_name = f"{args.common_name}_{args.mu}"
    return os.path.join(args.log_path, f"{args.file_name}_{args.cur_time}.log")


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

    global_protos_tensor = (
        global_protos.to(device) if global_protos is not None else None
    )

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

    return {
        "loss": total_loss / num_batches,
        "state": {k: v.cpu().detach().clone() for k, v in model.state_dict().items()},
        "protos": extract_prototypes(model, loader, num_classes, feature_dim, device),
    }


class Server(BaseServer):
    def __init__(self, args):
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
                    self.global_protos.cpu() if self.global_protos is not None else None,
                ]
                for i in selected_clients
            ]
            results = self.run_clients(client_worker, p)

            total_loss = 0.0
            selected_protos = []
            for cid, res in results.items():
                total_loss += res["loss"]
                self.clients_state[cid] = res["state"]
                selected_protos.append(res["protos"])
            self.loss.append(total_loss / num_join_clients)

            # 计算参与客户端的权重
            current_weights = [self.weights[i] for i in selected_clients]
            sum_weights = sum(current_weights)
            norm_weights = [w / sum_weights for w in current_weights]

            # 使用统一的张量聚合函数
            self.global_protos = proto_aggregate(
                selected_protos,
                weights=norm_weights,
                old_global_protos=self.global_protos,
            ).to(self.device)
            self.evaluate(protos=self.global_protos)

            print(
                f"Global Accuracy: {self.acc[-1]:.2f}%, Proto Accuracy: {self.acc_proto[-1]:.2f}%, Avg Loss: {self.loss[-1]:.4f}"
            )
            print(f"Round finished in {time.time() - t0:.2f} seconds")

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
