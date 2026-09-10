import os
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from .utils import (
    BaseParams,
    BaseServer,
    check_losses,
    clone_cpu_state,
    evaluate_model,
    fmt_num,
    get_model,
    param_aggregate,
)
from .utils.loss import kl_loss


def get_path(args):
    args.file_name = (
        f"{args.common_name}_{fmt_num(args.alpha_fml)}_{fmt_num(args.beta_fml)}"
    )
    return os.path.join(args.log_path, f"{args.file_name}_{args.cur_time}.log")


@dataclass
class Params(BaseParams):
    local_state: dict[str, torch.Tensor]
    alpha_fml: float
    beta_fml: float


def train(p: Params):
    """
    FML (Federated Mutual Learning) 联邦互学习本地训练。
    """
    device = torch.device(p.client_gpu)

    # 1. 初始化全局模型 (MEME)
    global_model = get_model(p).to(
        device
    )
    global_model.load_state_dict(p.model_state)

    # 2. 初始化本地模型 (个性化模型)
    local_model = get_model(p).to(
        device
    )
    local_model.load_state_dict(p.local_state)

    # 优化器设置
    opt_g = torch.optim.SGD(
        global_model.parameters(),
        lr=p.lr,
        momentum=p.momentum,
        weight_decay=p.weight_decay,
    )
    opt_l = torch.optim.SGD(
        local_model.parameters(),
        lr=p.lr,
        momentum=p.momentum,
        weight_decay=p.weight_decay,
    )

    loader = torch.utils.data.DataLoader(
        p.train_set, batch_size=p.batch_size, shuffle=True
    )

    global_model.train()
    local_model.train()

    total_loss_g = 0.0
    total_loss_l = 0.0
    num_batches = 0

    for _ in range(p.epochs):
        for x, y, *_ in loader:
            x, y = x.to(device), y.to(device)
            out_g = global_model(x)
            out_l = local_model(x)
            ce_g = F.cross_entropy(out_g, y)
            ce_l = F.cross_entropy(out_l, y)

            # 互学习损失 (KL 散度)
            loss_kl_g = kl_loss(out_g, out_l.detach())
            loss_kl_l = kl_loss(out_l, out_g.detach())

            loss_g = ce_g + p.beta_fml * loss_kl_g
            loss_l = ce_l + p.alpha_fml * loss_kl_l

            # 更新全局模型
            opt_g.zero_grad()
            check_losses(loss_g, locals())
            loss_g.backward()
            opt_g.step()

            # 更新本地模型
            opt_l.zero_grad()
            check_losses(loss_l, locals())
            loss_l.backward()
            opt_l.step()

            total_loss_g += loss_g.item()
            total_loss_l += loss_l.item()
            num_batches += 1

    avg_loss_g = total_loss_g / num_batches
    avg_loss_l = total_loss_l / num_batches
    global_state = clone_cpu_state(global_model.state_dict())
    local_state = clone_cpu_state(local_model.state_dict())
    return {
        "loss": avg_loss_l,
        "loss_global": avg_loss_g,
        "state": local_state,
        "state_global": global_state,
    }


class Server(BaseServer):
    def __init__(self, args):
        super().__init__(args, pfl=True)
        self.alpha_fml = args.alpha_fml
        self.beta_fml = args.beta_fml

        self.loss_g = []
        self.acc_g = []
        # 聚合所有客户端的测试集用于全局模型评估
        self.test_set_global = torch.utils.data.ConcatDataset(
            list(self.test_set.values())
        )

    def fit(self):
        num_join = max(1, int(self.num_clients * self.join_ratio))

        for r in range(self.start_round, self.rounds):
            t0 = time.time()
            print(f"\n--- FML Round {r + 1}/{self.rounds} ---")

            selected = torch.randperm(self.num_clients)[:num_join].tolist()
            print(f"Selected clients: {selected}")

            p = [
                Params(
                    **asdict(base),
                    local_state=self.clients_state[base.client_id],
                    alpha_fml=self.alpha_fml,
                    beta_fml=self.beta_fml,
                )
                for base in self.build_base_params(selected)
            ]
            results = self.run_clients(train, p)

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
            self.loss.append(total_loss / num_join)
            self.loss_g.append(total_loss_g / num_join)
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
            metrics = {
                "acc": self.acc,
                "acc_g": self.acc_g,
                "loss": self.loss,
                "loss_g": self.loss_g,
            }
            params = {
                "global": self.model.state_dict(),
                "client": self.clients_state,
            }
            self.save_checkpoint(r + 1, metrics, params)

    def save(self):
        metrics = {
            "acc": self.acc,
            "acc_global": self.acc_g,
            "loss": self.loss,
            "loss_global": self.loss_g,
        }
        params = {"global": self.model.state_dict(), "client": self.clients_state}
        self.deal_save(metrics, params)
