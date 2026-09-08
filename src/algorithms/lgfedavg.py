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
    global_head_state: dict[str, torch.Tensor]


def train(p: Params):
    """
    LG-FedAvg 本地训练流程:
    - 接收本地特征提取器状态与全局分类器状态。
    - 训练完整模型。
    - 训练完毕后返回更新后的提取器 (用于本地缓存) 和分类器 (用于全局聚合)。
    """
    device = torch.device(p.client_gpu)

    model = get_model(p.model_name, p.dataset, p.num_class, p.feature_dim).to(device)

    # 加载子模块 (注意：字典中的键不应带前缀)
    model.extractor.load_state_dict(p.model_state)
    model.classifier.load_state_dict(p.global_head_state)

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
    body_state = {
        k: model.extractor.state_dict()[k].cpu().detach().clone()
        for k in model.extractor.state_dict()
    }
    head_state = {
        k: model.classifier.state_dict()[k].cpu().detach().clone()
        for k in model.classifier.state_dict()
    }
    return {"loss": avg_loss, "body": body_state, "head": head_state}


class Server(BaseServer):
    def __init__(self, args):
        # pfl=True 表示此算法是个性化算法，评估时使用本地测试集
        super().__init__(args, pfl=True)
        # 覆盖 BaseServer 的初始化逻辑：LG-FedAvg 只需存储各客户端的特征提取器 (Extractor) 状态
        self.clients_state = [
            self.model.extractor.state_dict() for _ in range(self.num_clients)
        ]

    def fit(self):
        num_join = max(1, int(self.num_clients * self.join_ratio))

        for r in range(self.start_round, self.rounds):
            t0 = time.time()
            print(f"\n--- LG-FedAvg Round {r + 1}/{self.rounds} ---")
            selected = torch.randperm(self.num_clients)[:num_join].tolist()

            # 获取当前的全局共享分类头 (Head)
            global_head_state = self.model.classifier.state_dict()

            base_params = self.build_base_params(selected)
            for base in base_params:
                base.model_state = self.clients_state[base.client_id]

            p = [
                Params(
                    **asdict(base),
                    global_head_state=global_head_state,
                )
                for base in base_params
            ]
            results = self.run_clients(train, p)

            total_loss = 0.0
            new_heads = []
            current_weights = []
            for cid, res in results.items():
                total_loss += res["loss"]
                # 将更新后的本地主体结构保存回服务端
                self.clients_state[cid] = res["body"]
                new_heads.append(res["head"])
                current_weights.append(self.weights[cid])

            self.loss.append(total_loss / num_join)
            norm_weights = [w / sum(current_weights) for w in current_weights]

            # 仅聚合分类头模块的过程
            self.model.classifier.load_state_dict(
                param_aggregate(new_heads, norm_weights)
            )

            self.evaluate()
            print(f"Accuracy: {self.acc[-1]:.2f}%, Loss: {self.loss[-1]:.4f}")
            print(f"Round finished in {time.time() - t0:.2f} seconds")
            metrics = {"acc": self.acc, "loss": self.loss}
            params = {
                "global": self.model.state_dict(),
                "aux": {"clients_state": self.clients_state},
            }
            self.save_checkpoint(r + 1, metrics, params)

    def load_checkpoint(self, path):
        params = super().load_checkpoint(path)
        self.clients_state = params["aux"]["clients_state"]
        return params

    def evaluate(self):
        """
        构建带有正确前缀的完整 state_dicts 用于 BaseServer.evaluate() 的评估指标提取。
        这确保了在个性化测试期间 load_state_dict 功能的正常行为。
        """
        full_states = []
        # 获取最新的全局分类器并为其加载键前缀
        global_head_kv = {
            f"classifier.{k}": v for k, v in self.model.classifier.state_dict().items()
        }

        for i in range(self.num_clients):
            # 获取当前客户端的本地提取器并添加字典键前缀
            local_body_kv = {
                f"extractor.{k}": v for k, v in self.clients_state[i].items()
            }
            # 合并形成完整的状态字典
            full_state = local_body_kv
            full_state.update(global_head_kv)
            full_states.append(full_state)

        # 调用父类的智能评估方法
        super().evaluate(model_states=full_states)

    def save(self):
        # 准备用于保存或分析的完整模型 state_dicts
        client_states_full = []
        global_head_kv = {
            f"classifier.{k}": v for k, v in self.model.classifier.state_dict().items()
        }
        for i in range(self.num_clients):
            full_state = {f"extractor.{k}": v for k, v in self.clients_state[i].items()}
            full_state.update(global_head_kv)
            client_states_full.append(full_state)

        metrics = {"acc": self.acc, "loss": self.loss}
        params = {
            "global": self.model.classifier.state_dict(),
            "client": client_states_full,
        }
        self.deal_save(metrics, params)
