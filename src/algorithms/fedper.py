import os
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from .utils import (
    BaseParams,
    BaseServer,
    get_model,
    param_aggregate,
)


def get_path(args):
    args.file_name = f"{args.common_name}"
    return os.path.join(args.log_path, f"{args.file_name}_{args.cur_time}.log")


@dataclass
class Params(BaseParams):
    local_head_state: dict[str, torch.Tensor]


def train(p: Params):
    device = torch.device(p.client_gpu)

    model = get_model(p.model_name, p.dataset, p.num_class, p.feature_dim).to(device)
    model.extractor.load_state_dict(p.model_state)
    model.classifier.load_state_dict(p.local_head_state)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=p.lr,
        momentum=p.momentum,
        weight_decay=p.weight_decay,
    )
    loader = torch.utils.data.DataLoader(
        p.train_set, batch_size=p.batch_size, shuffle=True
    )

    model.train()
    total_loss = 0.0
    num_batches = 0
    for _ in range(p.epochs):
        for x, y, *_ in loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = F.cross_entropy(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    # 将拆分后的特征提取器和分类头状态返回，以实现高效通信
    new_body = {
        k: v.cpu().detach().clone() for k, v in model.extractor.state_dict().items()
    }
    new_head = {
        k: v.cpu().detach().clone() for k, v in model.classifier.state_dict().items()
    }
    return {"loss": avg_loss, "body": new_body, "head": new_head}


class Server(BaseServer):
    def __init__(self, args):
        super().__init__(args, pfl=True)
        self.client_head_states = [
            self.model.classifier.state_dict() for _ in range(self.num_clients)
        ]

    def fit(self):
        num_join = max(1, int(self.num_clients * self.join_ratio))

        for r in range(self.rounds):
            t0 = time.time()
            print(f"\n--- FedPer Round {r + 1}/{self.rounds} ---")
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
            print(f"Accuracy: {self.acc[-1]:.2f}%, Loss: {self.loss[-1]:.4f}")
            print(f"Round finished in {time.time() - t0:.2f} seconds")

    def evaluate(self):
        """构建包含正确前缀的完整模型状态字典，以供 BaseServer 进行评估。"""
        full_states = []
        global_body = {
            f"extractor.{k}": v for k, v in self.model.extractor.state_dict().items()
        }
        for i in range(self.num_clients):
            # 克隆数据以避免内存副作用 (共享变量意外修改)
            full_state = {k: v.clone() for k, v in global_body.items()}
            head_state = {
                f"classifier.{k}": v.clone()
                for k, v in self.client_head_states[i].items()
            }
            full_state.update(head_state)
            full_states.append(full_state)
        super().evaluate(model_states=full_states)

    def save(self):
        # 为保存状态同时也构建完整的模型状态
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
