import os
import time

import numpy as np
import torch

from .utils import (
    BaseServer,
    ce_loss,
    evaluate_model,
    get_model,
    kl_loss,
    param_aggregate,
    _fmt_num,
)


def get_path(args):
    args.file_name = f"{args.common_name}_{_fmt_num(args.alpha_fml)}_{_fmt_num(args.beta_fml)}"
    return os.path.join(args.log_path, f"{args.file_name}_{args.cur_time}.log")


def train_worker(params):
    """
    FML (Federated Mutual Learning) 联邦互学习本地训练。
    """
    (
        _,
        device,
        global_state,
        train_set,
        local_state,
        model_name,
        dataset_name,
        lr,
        batch_size,
        epochs,
        alpha_fml,
        beta_fml,
        feature_dim,
    ) = params

    # 1. 初始化全局模型 (MEME)
    global_model = get_model(model_name, dataset_name, feature_dim).to(device)
    global_model.load_state_dict(global_state)

    # 2. 初始化本地模型 (个性化模型)
    local_model = get_model(model_name, dataset_name, feature_dim).to(device)
    local_model.load_state_dict(local_state)

    # 优化器设置
    opt_g = torch.optim.SGD(global_model.parameters(), lr=lr)
    opt_l = torch.optim.SGD(local_model.parameters(), lr=lr)

    loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    global_model.train()
    local_model.train()

    total_loss_g = 0.0
    total_loss_l = 0.0
    num_batches = 0

    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out_g = global_model(x)
            out_l = local_model(x)
            ce_g = ce_loss(out_g, y)
            ce_l = ce_loss(out_l, y)

            # 互学习损失 (KL 散度)
            loss_kl_g = kl_loss(out_g, out_l.detach())
            loss_kl_l = kl_loss(out_l, out_g.detach())

            loss_g = ce_g + beta_fml * loss_kl_g
            loss_l = ce_l + alpha_fml * loss_kl_l

            # 更新全局模型
            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()

            # 更新本地模型
            opt_l.zero_grad()
            loss_l.backward()
            opt_l.step()

            total_loss_g += loss_g.item()
            total_loss_l += loss_l.item()
            num_batches += 1

    avg_loss_g = total_loss_g / num_batches
    avg_loss_l = total_loss_l / num_batches
    global_state = {
        k: v.cpu().detach().clone() for k, v in global_model.state_dict().items()
    }
    local_state = {
        k: v.cpu().detach().clone() for k, v in local_model.state_dict().items()
    }
    return {
        "loss": avg_loss_l,
        "loss_global": avg_loss_g,
        "state": local_state,
        "state_global": global_state,
    }


class Server(BaseServer):
    def __init__(self, args):
        super().__init__(True, args)

        self.loss_g = []
        self.acc_g = []
        # 聚合所有客户端的测试集用于全局模型评估
        self.test_set_global = torch.utils.data.ConcatDataset(
            list(self.test_set.values())
        )

    def fit(self):
        num_join_clients = int(self.num_clients * self.args.join_ratio)
        num_join_clients = max(1, num_join_clients)

        for r in range(self.rounds):
            t0 = time.time()
            print(f"\n--- FML Round {r + 1}/{self.rounds} ---")

            selected_clients = np.random.choice(
                self.num_clients, num_join_clients, replace=False
            )
            print(f"Selected clients: {selected_clients}")

            p = [
                [
                    i,
                    self.client_gpu[i],
                    self.model.state_dict(),
                    self.train_sets[i],
                    self.clients_state[i],
                    self.args.model,
                    self.args.dataset,
                    self.args.lr,
                    self.args.batch_size,
                    self.args.epochs,
                    self.args.alpha_fml,
                    self.args.beta_fml,
                    self.args.feature_dim,
                ]
                for i in selected_clients
            ]
            results = self.run_clients(train_worker, p)

            total_loss = 0.0
            total_loss_g = 0.0
            selected_states = []
            selected_states_g = []
            current_weights = []
            for cid, res in results.items():
                total_loss += res["loss"]
                total_loss_g += res["loss_global"]
                selected_states.append(res["state"])
                self.clients_state[cid] = res["state"]
                selected_states_g.append(res["state_global"])
                current_weights.append(self.weights[cid])
            self.loss.append(total_loss / num_join_clients)
            self.loss_g.append(total_loss_g / num_join_clients)
            sum_weights = sum(current_weights)
            norm_weights = [w / sum_weights for w in current_weights]

            # 聚合全局模型参数
            self.model.load_state_dict(param_aggregate(selected_states_g, norm_weights))

            # 评估个性化模型准确率 (BaseServer.evaluate 在 pfl=True 时计算各客户端本地模型在其测试集上的均值)
            self.evaluate()
            # 评估聚合后的全局模型在全量测试集上的准确率
            acc_g = evaluate_model(self.model, self.test_set_global, self.device)
            self.acc_g.append(acc_g)

            print(
                f"Acc Global: {acc_g:.2f}%, Acc Local: {self.acc[-1]:.2f}%, "
                f"Loss Global: {self.loss_g[-1]:.4f}, Loss Local: {self.loss[-1]:.4f}"
            )
            print(f"Round finished in {time.time() - t0:.2f} seconds")

    def save(self):
        f = {
            "acc": {"local": self.acc, "global": self.acc_g},
            "loss": {"local": self.loss, "global": self.loss_g},
            "state_dict": {
                "global": self.model.state_dict(),
                "client": self.clients_state,
            },
        }
        self.deal_save(f)
