import os
import time

import numpy as np
import torch

from .utils import (
    BaseServer,
    ce_loss,
    get_model,
    param_aggregate,
)


def get_path(args):
    args.file_name = f"{args.common_name}"
    return os.path.join(args.log_path, f"{args.file_name}_{args.cur_time}.log")


def client_worker(params):
    (
        _,
        device,
        global_body_state,
        train_set,
        local_head_state,
        model_name,
        dataset_name,
        lr,
        batch_size,
        epochs,
        feature_dim,
    ) = params

    model = get_model(model_name, dataset_name, feature_dim).to(device)
    model.extractor.load_state_dict(global_body_state)
    model.classifier.load_state_dict(local_head_state)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    model.train()
    total_loss = 0.0
    num_batches = 0
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = ce_loss(output, y)
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
        super().__init__(True, args)
        self.client_head_states = [
            self.model.classifier.state_dict() for _ in range(self.num_clients)
        ]

    def fit(self):
        num_join_clients = max(1, int(self.num_clients * self.args.join_ratio))

        for r in range(self.rounds):
            t0 = time.time()
            print(f"\n--- FedPer Round {r + 1}/{self.rounds} ---")
            selected_clients = np.random.choice(
                self.num_clients, num_join_clients, replace=False
            )

            # 全局共享特征提取器 (Body)
            global_body_state = self.model.extractor.state_dict()

            p = [
                [
                    i,
                    self.client_gpu[i],
                    global_body_state,
                    self.train_sets[i],
                    self.client_head_states[i],
                    self.args.model,
                    self.args.dataset,
                    self.args.lr,
                    self.args.batch_size,
                    self.args.epochs,
                    self.args.feature_dim,
                ]
                for i in selected_clients
            ]
            results = self.run_clients(client_worker, p)

            total_loss = 0.0
            new_bodies = []
            current_weights = []
            for cid, res in results.items():
                total_loss += res["loss"]
                new_bodies.append(res["body"])
                self.client_head_states[cid] = res["head"]
                current_weights.append(self.weights[cid])

            self.loss.append(total_loss / num_join_clients)
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

        f = {
            "acc": self.acc,
            "loss": self.loss,
            "state_dict": {
                "global": self.model.extractor.state_dict(),
                "client": client_states,
            },
        }
        self.deal_save(f)
