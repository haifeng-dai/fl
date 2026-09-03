import os
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from .utils import (
    BaseParams,
    BaseServer,
    clone_cpu_state,
    extract_prototypes,
    fmt_num,
    get_model,
    proto_aggregate,
)


def get_path(args):
    args.file_name = f"{args.common_name}_{fmt_num(args.mu)}"
    return os.path.join(args.log_path, f"{args.file_name}_{args.cur_time}.log")


@dataclass
class Params(BaseParams):
    mu: float
    global_protos: torch.Tensor | None


def train(p: Params):
    device = torch.device(p.client_gpu)

    model = get_model(p.model_name, p.dataset, p.num_class, p.feature_dim).to(device)
    model.load_state_dict(p.model_state)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=p.lr,
        momentum=p.momentum,
        weight_decay=p.weight_decay,
    )
    loader = torch.utils.data.DataLoader(
        p.train_set, batch_size=p.batch_size, shuffle=True
    )

    global_protos_tensor = (
        p.global_protos.to(device) if p.global_protos is not None else None
    )

    total_loss = 0.0
    num_batches = 0
    model.train()
    for _ in range(p.epochs):
        for x, y, *_ in loader:
            x, y = x.to(device), y.to(device)
            feature = model.extractor(x)
            logits = model.classifier(feature)
            loss_ce = F.cross_entropy(logits, y)

            if global_protos_tensor is not None:
                target_protos = global_protos_tensor[y]
                loss_proto = F.mse_loss(feature, target_protos)
                loss = loss_ce + p.mu * loss_proto
            else:
                loss = loss_ce

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

    # 提取本地原型及样本计数
    local_protos, local_counts = extract_prototypes(
        model, loader, p.num_class, p.feature_dim, device, return_counts=True
    )

    return {
        "loss": total_loss / num_batches,
        "state": clone_cpu_state(model.state_dict()),
        "protos": local_protos,
        "counts": local_counts,
    }


class Server(BaseServer):
    def __init__(self, args):
        super().__init__(args, pfl=True)
        self.mu = args.mu

        self.global_protos = None

    def fit(self):
        num_join = max(1, int(self.num_clients * self.join_ratio))

        for r in range(self.rounds):
            t0 = time.time()
            print(f"\n--- FedProto Round {r + 1}/{self.rounds} ---")

            selected = torch.randperm(self.num_clients)[:num_join].tolist()
            print(f"Selected clients: {selected}")

            base_params = self.build_base_params(selected)
            for base in base_params:
                base.model_state = self.clients_state[base.client_id]

            p = [
                Params(
                    **asdict(base),
                    mu=self.mu,
                    global_protos=(
                        self.global_protos.cpu()
                        if self.global_protos is not None
                        else None
                    ),
                )
                for base in base_params
            ]
            results = self.run_clients(train, p)

            total_loss = 0.0
            selected_protos = []
            selected_counts = []
            for cid, res in results.items():
                total_loss += res["loss"]
                self.clients_state[cid] = res["state"]
                selected_protos.append(res["protos"])
                selected_counts.append(res["counts"])
            self.loss.append(total_loss / num_join)

            # 聚合原型向量：按样本计数加权
            self.global_protos = proto_aggregate(
                selected_protos,
                local_counts_list=selected_counts,
                old_global_protos=self.global_protos,
            )
            self.evaluate(protos=self.global_protos)

            print(
                f"Global Accuracy: {self.acc[-1]:.2f}%, Proto Accuracy: {self.acc_proto[-1]:.2f}%, Avg Loss: {self.loss[-1]:.4f}"
            )
            print(f"Round finished in {time.time() - t0:.2f} seconds")

    def save(self):
        metrics = {"acc": self.acc, "acc_p": self.acc_proto, "loss": self.loss}
        params = {"client": self.clients_state, "proto": self.global_protos}
        self.deal_save(metrics, params)
