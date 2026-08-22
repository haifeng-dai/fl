import os
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from .utils import (
    BaseParams,
    BaseServer,
    cos_similarity,
    extract_prototypes,
    get_model,
    proto_aggregate,
)


def get_path(args):
    args.file_name = f"{args.common_name}"
    return os.path.join(args.log_path, f"{args.file_name}_{args.cur_time}.log")


@dataclass
class Params(BaseParams):
    global_protos: torch.Tensor
    alpha: float


def train(p: Params):
    device = torch.device(p.client_gpu)

    # 1. 初始化模型并加载全局状态
    model = get_model(p.model_name, p.dataset, p.num_class, p.feature_dim).to(device)
    model.load_state_dict(p.model_state)
    global_protos = p.global_protos.data.clone().to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=p.lr)
    loader = torch.utils.data.DataLoader(p.train_set, batch_size=p.batch_size, shuffle=True)

    # 2. 本地模型多轮次训练
    total_loss = 0.0
    num_batches = 0

    model.train()
    for _ in range(p.epochs):
        for data, target, *_ in loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            features = model.extractor(data)
            output = model.classifier(features)
            loss_ce = F.cross_entropy(output, target)

            # 基于原型的对比损失 (Prototypical Contrastive Loss)
            loss_con = cos_similarity(features, global_protos, target, tau=1.0)

            loss = (1 - p.alpha) * loss_ce + p.alpha * loss_con
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

    # 3. 提取本地原型及样本计数
    local_protos, local_counts = extract_prototypes(
        model, loader, p.num_class, p.feature_dim, device, return_counts=True
    )

    return {
        "loss": total_loss / num_batches,
        "state": {k: v.cpu().detach().clone() for k, v in model.state_dict().items()},
        "protos": local_protos.cpu().detach().clone(),
        "counts": local_counts.cpu().detach().clone(),
    }


class Server(BaseServer):
    def __init__(self, args):
        super().__init__(False, args)

        self.global_protos = torch.zeros((self.num_class, self.feature_dim))

    def fit(self):
        num_join = max(1, int(self.num_clients * self.join_ratio))

        for r in range(self.rounds):
            t0 = time.time()
            print(f"\n--- FedProc Round {r + 1}/{self.rounds} ---")

            selected = torch.randperm(self.num_clients)[:num_join].tolist()
            print(f"Selected clients: {selected}")

            p = [
                Params(
                    **asdict(base),
                    global_protos=self.global_protos,
                    alpha=1.0 - (r / self.rounds),
                )
                for base in self.build_base_params(selected)
            ]
            results = self.run_clients(train, p)

            total_loss = 0.0
            selected_states = []
            all_local_protos = []
            all_local_counts = []
            for _, res in results.items():
                total_loss += res["loss"]
                selected_states.append(res["state"])
                all_local_protos.append(res["protos"])
                all_local_counts.append(res["counts"])
            self.loss.append(total_loss / num_join)

            # 聚合模型参数
            weights = [self.weights[i] for i in selected]
            sum_w = sum(weights)
            weights = [w / sum_w for w in weights]
            self.aggregate(selected_states, weights=weights)
            # 聚合原型向量：简单平均
            self.global_protos = proto_aggregate(
                all_local_protos,
                local_counts_list=None,
                old_global_protos=self.global_protos,
            ).cpu()

            self.evaluate()
            print(
                f"Global Accuracy: {self.acc[-1]:.2f}%, Avg Loss: {self.loss[-1]:.4f}"
            )
            print(f"Round finished in {time.time() - t0:.2f} seconds")

    def save(self):
        metrics = {"acc": self.acc, "loss": self.loss}
        params = {"global": self.model.state_dict(), "proto": self.global_protos}
        self.deal_save(metrics, params)
