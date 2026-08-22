import os
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from .utils import (
    BaseParams,
    BaseServer,
    fmt_num,
    get_model,
    param_aggregate,
)


def get_path(args):
    args.file_name = f"{args.common_name}_{fmt_num(args.epochs_head)}"
    return os.path.join(args.log_path, f"{args.file_name}_{args.cur_time}.log")


@dataclass
class Params(BaseParams):
    local_head_state: dict[str, torch.Tensor]
    epochs_head: int


def train(p: Params):
    device = torch.device(p.client_gpu)

    model = get_model(p.model_name, p.dataset, p.num_class, p.feature_dim).to(device)
    # 直接加载子模块 (特征提取器与分类器)
    model.extractor.load_state_dict(p.model_state)
    model.classifier.load_state_dict(p.local_head_state)

    optimizer = torch.optim.SGD(model.parameters(), lr=p.lr)
    loader = torch.utils.data.DataLoader(p.train_set, batch_size=p.batch_size, shuffle=True)

    model.train()
    # 阶段 1：仅训练分类头 (Head)
    for param in model.extractor.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True
    for _ in range(p.epochs_head):
        for x, y, *_ in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # 阶段 2：仅训练特征提取器 (Body)
    for param in model.extractor.parameters():
        param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = False
    total_loss = 0.0
    num_batches = 0
    for _ in range(p.epochs):
        for x, y, *_ in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

    return {
        "loss": total_loss / num_batches,  # avg_loss
        "body": {
            k: v.cpu().detach().clone() for k, v in model.extractor.state_dict().items()
        },  # body_state
        "head": {
            k: v.cpu().detach().clone()
            for k, v in model.classifier.state_dict().items()
        },  # head_state
    }


class Server(BaseServer):
    def __init__(self, args):
        super().__init__(True, args)
        self.epochs_head = args.epochs_head

        self.client_head_states = [
            self.model.classifier.state_dict() for _ in range(self.num_clients)
        ]

    def fit(self):
        num_join = max(1, int(self.num_clients * self.join_ratio))

        for r in range(self.rounds):
            t0 = time.time()
            print(f"\n--- FedRep Round {r + 1}/{self.rounds} ---")
            selected = torch.randperm(self.num_clients)[:num_join].tolist()

            # 全局共享特征提取器 (Body)
            global_body_state = self.model.extractor.state_dict()

            base_params = self.build_base_params(selected)
            for base in base_params:
                base.model_state = global_body_state

            p = [
                Params(
                    **asdict(base),
                    local_head_state=self.client_head_states[base.client_id],
                    epochs_head=self.epochs_head,
                )
                for base in base_params
            ]
            results = self.run_clients(train, p)

            total_loss = 0.0
            new_bodies = []
            current_weights = []
            for cid, res in results.items():
                total_loss += res["loss"]
                new_bodies.append(res["body"])
                self.client_head_states[cid] = res["head"]
                current_weights.append(self.weights[cid])

            self.loss.append(total_loss / num_join)
            norm_weights = [w / sum(current_weights) for w in current_weights]

            # 仅聚合特征提取器 (Body)
            self.model.extractor.load_state_dict(
                param_aggregate(new_bodies, norm_weights)
            )

            self.evaluate()

            print(
                f"Personalized Accuracy: {self.acc[-1]:.2f}%, Avg Loss: {self.loss[-1]:.4f}"
            )
            print(f"Round finished in {time.time() - t0:.2f} seconds")

    def evaluate(self):
        full_states = []
        global_body = {
            f"extractor.{k}": v for k, v in self.model.extractor.state_dict().items()
        }
        for i in range(self.num_clients):
            full_state = {k: v.clone() for k, v in global_body.items()}
            head_state = {
                f"classifier.{k}": v.clone()
                for k, v in self.client_head_states[i].items()
            }
            full_state.update(head_state)
            full_states.append(full_state)
        super().evaluate(model_states=full_states)

    def save(self):
        client_states = []
        global_body = {
            f"extractor.{k}": v for k, v in self.model.extractor.state_dict().items()
        }
        for i in range(self.num_clients):
            full_state = {k: v.clone() for k, v in global_body.items()}
            head_state = {
                f"classifier.{k}": v.clone()
                for k, v in self.client_head_states[i].items()
            }
            full_state.update(head_state)
            client_states.append(full_state)

        metrics = {"acc": self.acc, "loss": self.loss}
        params = {"global": self.model.extractor.state_dict(), "client": client_states}
        self.deal_save(metrics, params)
