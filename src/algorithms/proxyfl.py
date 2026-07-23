import os
import time

import torch

from .utils import (
    BaseServer,
    fmt_num,
    ce_loss,
    evaluate_model,
    flattened_matrix_aggregate,
    generate_adjacency_matrix,
    get_model,
    kl_loss,
)


def get_path(args):
    adj_suffix = f"{args.adj_type}"
    if args.adj_type == "random":
        adj_suffix += f"_{fmt_num(args.edge_p)}"
    elif args.adj_type == "small_world":
        adj_suffix += f"_{fmt_num(args.k_small_world)}_{fmt_num(args.edge_p)}"
    elif args.adj_type == "scale_free":
        adj_suffix += f"_{fmt_num(args.m_scale_free)}"

    args.file_name = f"{args.common_name}_{adj_suffix}_{fmt_num(args.mu)}"
    return os.path.join(args.log_path, f"{args.file_name}_{args.cur_time}.log")


def train(params):
    """
    ProxyFL 本地训练流程，利用私有本地模型与共享代理模型之间的相互蒸馏机制 (Mutual Distillation)。
    """
    (
        _,
        device,
        proxy_state,
        train_set,
        model_name,
        dataset_name,
        lr,
        batch_size,
        epochs,
        feature_dim,
        local_state,
        mu,
    ) = params

    # 1. 初始化代理模型 (公共/共享模型)
    proxy_model = get_model(model_name, dataset_name, feature_dim).to(device)
    proxy_model.load_state_dict(proxy_state)

    # 2. 初始化本地模型 (私有/个性化模型)
    local_model = get_model(model_name, dataset_name, feature_dim).to(device)
    local_model.load_state_dict(local_state)

    # 优化器设置
    # 通常 ProxyFL 允许设置不同的学习率 LR，但为了简便我们在未指明时均使用相同学习率
    opt_p = torch.optim.SGD(proxy_model.parameters(), lr=lr)
    opt_l = torch.optim.SGD(local_model.parameters(), lr=lr)

    loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    proxy_model.train()
    local_model.train()

    total_loss_p = 0.0
    total_loss_l = 0.0
    num_batches = 0

    for _ in range(epochs):
        for x, y, *_ in loader:
            x, y = x.to(device), y.to(device)
            out_p = proxy_model(x)
            out_l = local_model(x)
            ce_p = ce_loss(out_p, y)
            ce_l = ce_loss(out_l, y)

            # 相互知识蒸馏 (Mutual Distillation, 基于 KL 散度)
            loss_kl_p = kl_loss(out_p, out_l.detach())
            loss_kl_l = kl_loss(out_l, out_p.detach())

            loss_p = ce_p + mu * loss_kl_p
            loss_l = ce_l + mu * loss_kl_l

            # 反向传播并更新代理模型
            opt_p.zero_grad()
            loss_p.backward()
            opt_p.step()

            # 反向传播并更新本地模型
            opt_l.zero_grad()
            loss_l.backward()
            opt_l.step()

            total_loss_p += loss_p.item()
            total_loss_l += loss_l.item()
            num_batches += 1

    avg_loss_l = total_loss_l / num_batches
    avg_loss_p = total_loss_p / num_batches
    local_state = {
        k: v.cpu().detach().clone() for k, v in local_model.state_dict().items()
    }
    proxy_state = {
        k: v.cpu().detach().clone() for k, v in proxy_model.state_dict().items()
    }
    return {
        "loss": avg_loss_l,
        "loss_proxy": avg_loss_p,
        "state": local_state,
        "state_proxy": proxy_state,
    }


class Server(BaseServer):
    def __init__(self, args):
        super().__init__(True, args)

        # 为每个客户端初始化对应的代理模型 (Public/Shared)
        self.client_states_p = [
            self.model.state_dict() for _ in range(self.num_clients)
        ]

        self.loss_p = []
        self.acc_p = []
        # 使用通用的邻接矩阵生成函数并进行行归一化处理
        # 必须归一化以防止在去中心化聚合时权重累加导致梯度爆炸 (NaN)
        A = generate_adjacency_matrix(args).to(self.device)
        self.adj_matrix = A / A.sum(dim=1, keepdim=True)

    def fit(self):
        num_join = max(1, int(self.num_clients * self.args.join_ratio))

        for r in range(self.rounds):
            t0 = time.time()
            print(f"\n--- ProxyFL Round {r + 1}/{self.rounds} ---")

            selected = torch.randperm(self.num_clients)[:num_join].tolist()
            print(f"Selected clients: {selected}")

            # 预计算所有客户端的聚合代理状态（GPU 矩阵乘法）
            proxy_list = [self.client_states_p[i] for i in range(self.num_clients)]
            agg_proxy_list = flattened_matrix_aggregate(
                proxy_list, self.adj_matrix, self.device
            )

            p = self.build_base_params(selected)
            for params, i in zip(p, selected):
                params[2] = agg_proxy_list[i]
                params.append(self.clients_state[i])
                params.append(self.args.mu)
            results = self.run_clients(train, p)

            # 3. 收集更新客户端状态数据与评估并计算平均损失
            total_loss = 0.0
            total_loss_p = 0.0
            for cid, res in results.items():
                total_loss += res["loss"]
                total_loss_p += res["loss_proxy"]
                self.clients_state[cid] = res["state"]
                self.client_states_p[cid] = res["state_proxy"]
            self.loss.append(total_loss / num_join)
            self.loss_p.append(total_loss_p / num_join)

            # 4. 执行预测评估 (基于最新状态的本地个性化模型)
            self.evaluate()

            # 5. 执行代理模型预测评估
            accs_p = []
            self.model.to(self.device)
            for i in range(self.num_clients):
                self.model.load_state_dict(self.client_states_p[i])
                accs_p.append(evaluate_model(self.model, self.test_set[i], self.device))
            self.model.cpu()
            self.acc_p.append(sum(accs_p) / len(accs_p) if accs_p else 0.0)

            print(
                f"Avg Local Acc: {self.acc[-1]:.2f}%, Avg Local Loss: {self.loss[-1]:.4f}\n"
                f"Avg Proxy Acc: {self.acc_p[-1]:.2f}%, Avg Proxy Loss: {self.loss_p[-1]:.4f}"
            )
            print(f"Round finished in {time.time() - t0:.2f} seconds")

    def save(self):
        metrics = {"acc": self.acc, "acc_p": self.acc_p, "loss": self.loss, "loss_p": self.loss_p}
        params = {"client": self.clients_state, "aux": {"proxy": self.client_states_p}}
        self.deal_save(metrics, params)
