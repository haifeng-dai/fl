import os
import time

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


def train(params):
    """
    LG-FedAvg 本地训练流程:
    - 接收本地特征提取器状态与全局分类器状态。
    - 训练完整模型。
    - 训练完毕后返回更新后的提取器 (用于本地缓存) 和分类器 (用于全局聚合)。
    """
    (
        _,
        device,
        local_body_state,
        train_set,
        model_name,
        dataset_name,
        lr,
        batch_size,
        epochs,
        feature_dim,
        num_class,
        global_head_state,
    ) = params

    model = get_model(model_name, dataset_name, num_class, feature_dim).to(device)

    # 加载子模块 (注意：字典中的键不应带前缀)
    model.extractor.load_state_dict(local_body_state)
    model.classifier.load_state_dict(global_head_state)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    model.train()
    total_loss = 0.0
    num_batches = 0
    for _ in range(epochs):
        for x, y, *_ in loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = ce_loss(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    body_state = {
        k: model.extractor.state_dict()[k].cpu().detach().clone()
        for k in model.extractor.state_dict().keys()
    }
    head_state = {
        k: model.classifier.state_dict()[k].cpu().detach().clone()
        for k in model.classifier.state_dict().keys()
    }
    return {"loss": avg_loss, "body": body_state, "head": head_state}


class Server(BaseServer):
    def __init__(self, args):
        # pfl=True 表示此算法是个性化算法，评估时使用本地测试集
        super().__init__(True, args)
        # 覆盖 BaseServer 的初始化逻辑：LG-FedAvg 只需存储各客户端的特征提取器 (Extractor) 状态
        self.clients_state = [
            self.model.extractor.state_dict() for _ in range(self.num_clients)
        ]

    def fit(self):
        num_join = max(1, int(self.num_clients * self.args.join_ratio))

        for r in range(self.rounds):
            t0 = time.time()
            print(f"\n--- LG-FedAvg Round {r + 1}/{self.rounds} ---")
            selected = torch.randperm(self.num_clients)[:num_join].tolist()

            # 获取当前的全局共享分类头 (Head)
            global_head_state = self.model.classifier.state_dict()

            p = self.build_base_params(selected)
            for params, i in zip(p, selected):
                params[2] = self.clients_state[i]
                params.append(global_head_state)
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
        params = {"global": self.model.classifier.state_dict(), "client": client_states_full}
        self.deal_save(metrics, params)
