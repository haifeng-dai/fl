import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .utils import (
    BaseParams,
    BaseServer,
    get_model,
)


def get_path(args):
    args.file_name = f"{args.common_name}"
    return os.path.join(args.log_path, f"{args.file_name}_{args.cur_time}.log")


def train(p: BaseParams):
    """
    纯本地训练机制 - 无任何通信的独立训练流程。
    各个客户端完全基于私有数据持续训练自己的模型。
    """
    device = torch.device(p.client_gpu)

    model = get_model(p.model_name, p.dataset, p.num_class, p.feature_dim).to(device)
    model.load_state_dict(p.model_state)

    optimizer = torch.optim.SGD(model.parameters(), lr=p.lr)
    loader = DataLoader(p.train_set, batch_size=p.batch_size, shuffle=True)

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

    avg_loss = total_loss / num_batches
    model_state = {k: v.cpu().detach().clone() for k, v in model.state_dict().items()}
    return {"loss": avg_loss, "state": model_state}


class Server(BaseServer):
    def __init__(self, args):
        super().__init__(True, args)

    def fit(self):
        num_join = max(1, int(self.num_clients * self.join_ratio))

        for r in range(self.rounds):
            t0 = time.time()
            print(f"\n--- Local Training Round {r + 1}/{self.rounds} ---")

            selected = torch.randperm(self.num_clients)[:num_join].tolist()
            print(f"Selected clients: {selected}")

            p = self.build_base_params(selected)
            for base in p:
                base.model_state = self.clients_state[base.client_id]
            results = self.run_clients(train, p)

            total_loss = 0.0
            for cid, res in results.items():
                total_loss += res["loss"]
                self.clients_state[cid] = res["state"]
            self.loss.append(total_loss / num_join)

            self.evaluate()
            print(
                f"Personalized Accuracy: {self.acc[-1]:.2f}%, Avg Loss: {self.loss[-1]:.4f}"
            )
            print(f"Round finished in {time.time() - t0:.2f} seconds")

    def save(self):
        metrics = {"acc": self.acc, "loss": self.loss}
        params = {"client": self.clients_state}
        self.deal_save(metrics, params)
