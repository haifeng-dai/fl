import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .utils import (
    BaseParams,
    BaseServer,
    get_model,
)


def get_path(args):
    args.file_name = f"{args.common_name}"
    return os.path.join(args.log_path, f"{args.file_name}_{args.cur_time}.log")


def train(p: BaseParams):
    device = torch.device(p.client_gpu)

    model = get_model(p.model_name, p.dataset, p.num_class, p.feature_dim)
    model.load_state_dict(p.model_state)
    model.to(device)

    x_all = p.train_set.x
    y_all = p.train_set.y

    if len(x_all) == 0:
        model_state = {
            k: v.cpu().detach().clone() for k, v in model.state_dict().items()
        }
        return {"loss": 0.0, "state": model_state}

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=p.lr,
        momentum=p.momentum,
        weight_decay=p.weight_decay,
    )
    loader = DataLoader(
        TensorDataset(x_all, y_all),
        batch_size=p.batch_size,
        shuffle=True,
    )

    total_loss = 0.0
    num_batches = 0
    model.train()
    for _ in range(p.epochs):
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model(x_batch)
            loss = F.cross_entropy(logits, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / max(1, num_batches)
    model_state = {k: v.cpu().detach().clone() for k, v in model.state_dict().items()}
    return {"loss": avg_loss, "state": model_state}


class Server(BaseServer):
    def __init__(self, args):
        if args.ssl not in ("sample", "double", "sfd"):
            raise ValueError("fedavg_sl 要求 ssl 为 sample、double 或 sfd。")
        super().__init__(args, is_ssl=True, pfl=False)

    def fit(self):
        num_join = max(1, int(self.num_clients * self.join_ratio))

        for r in range(self.rounds):
            t0 = time.time()
            print(f"\n--- FedAvg-SL Round {r + 1}/{self.rounds} ---")

            selected = sorted(torch.randperm(self.num_clients)[:num_join].tolist())
            print(f"Selected clients: {selected}")

            p = self.build_base_params(selected)
            results = self.run_clients(train, p)

            total_loss = 0.0
            selected_states = []
            current_weights = []
            for cid, res in results.items():
                total_loss += res["loss"]
                selected_states.append(res["state"])
                current_weights.append(self.weights[cid])

            self.loss.append(total_loss / num_join)
            sum_weights = sum(current_weights)
            if sum_weights > 0:
                norm_weights = [w / sum_weights for w in current_weights]
            else:
                norm_weights = [1.0 / len(selected) for _ in selected]

            self.aggregate(selected_states, weights=norm_weights)
            self.evaluate()

            if self.is_sfd:
                print(
                    f"[Accuracy]\n"
                    f"  - Standard:   {self.acc[-1]:.2f}% "
                    f"(Src: {self.acc_source[-1]:.2f}%, "
                    f"Tgt: {self.acc_target[-1]:.2f}%)\n"
                    f"Avg Loss: {self.loss[-1]:.4f}"
                )
            else:
                print(
                    f"Global Accuracy: {self.acc[-1]:.2f}%, Avg Loss: {self.loss[-1]:.4f}"
                )
            print(f"Round finished in {time.time() - t0:.2f} seconds")

    def save(self):
        metrics = {"acc": self.acc, "loss": self.loss}
        if self.is_sfd:
            metrics["acc_source"] = self.acc_source
            metrics["acc_target"] = self.acc_target
        params = {"global": self.model.state_dict()}
        self.deal_save(metrics, params)
