import argparse
import time
import os

import numpy as np
import torch
import torch.nn.functional as F

from .utils import (
    BaseServer,
    ce_loss,
    evaluate_model,
    evaluate_prototype,
    get_model,
    mse_loss,
    param_aggregate,
    run_parallel_clients,
)


def add_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("FedDPL Specific Arguments")
    group.add_argument(
        "--lambda_", type=float, default=1.0, help="Weight for PLN Contrastive Loss"
    )
    group.add_argument(
        "--epoch_pln", type=int, default=2, help="Epochs for PLN learning"
    )
    group.add_argument(
        "--lr_pln", type=float, default=0.01, help="Learning rate for PLN learning"
    )
    group.add_argument(
        "--batch_size_pln", type=int, default=32, help="Batch size for PLN learning"
    )
    group.add_argument("--depth_pln", type=int, default=2, help="Depth of PLN network")
    group.add_argument(
        "--width_pln", type=int, default=128, help="Width of PLN network"
    )
    group.add_argument(
        "--mode",
        type=str,
        default="normal",
        choices=["normal", "pln", "model", "all"],
        help="Task mode",
    )
    group.add_argument(
        "--fixed_proto",
        type=int,
        default=0,
        help="Whether to fix the prototypes during training",
    )
    group.add_argument(
        "--init_emb",
        type=int,
        default=0,
        help="Initialization strategy for PLN embeddings",
    )
    group.add_argument("--har", type=int, default=0, help="Whether to use HAR dataset")
    return parser


def get_path(args):
    args.file_name = f"{args.name_pre}_{args.lambda_}_{args.epoch_pln}_{args.lr_pln}_{args.batch_size_pln}_{args.depth_pln}_{args.width_pln}_{args.mode}_{args.fixed_proto}_{args.init_emb}_{args.har}"
    return os.path.join(args.log_path, f"{args.file_name}_{args.times}.log")


class PLN(torch.nn.Module):
    def __init__(self, num_classes, width, feature_dim, depth=1, fixed=0, init_emb=0):
        super().__init__()
        self.embedings = torch.nn.Embedding(num_classes, width)
        self.__init_embedings(init_emb)
        if fixed:
            self.embedings.weight.requires_grad = False
        if depth < 1:
            raise ValueError("depth must be at least 1")
        layers = [
            torch.nn.Sequential(torch.nn.Linear(width, width), torch.nn.ReLU())
            for _ in range(depth)
        ]
        self.middle = torch.nn.Sequential(*layers)
        self.fc = torch.nn.Linear(width, feature_dim)

    def __init_embedings(self, init_emb):
        # 初始化策略说明：
        # init_emb == 0 : 使用 Embedding 的默认初始化
        # init_emb == 1 : Uniform(-0.1, 0.1)
        # init_emb == 2 : Normal(mean=0, std=0.1)
        # init_emb == 3 : Normal(mean=0, std=0.01)
        # init_emb == 4 : Xavier Uniform
        # init_emb == 5 : Xavier Normal
        # init_emb == 6 : Kaiming Uniform
        # init_emb == 7 : Orthogonal
        if init_emb == 0:
            pass
        elif init_emb == 1:
            torch.nn.init.uniform_(self.embedings.weight, -0.1, 0.1)
        elif init_emb == 2:
            torch.nn.init.normal_(self.embedings.weight, mean=0.0, std=0.1)
        elif init_emb == 3:
            torch.nn.init.normal_(self.embedings.weight, mean=0.0, std=0.01)
        elif init_emb == 4:
            torch.nn.init.xavier_uniform_(self.embedings.weight)
        elif init_emb == 5:
            torch.nn.init.xavier_normal_(self.embedings.weight)
        elif init_emb == 6:
            torch.nn.init.kaiming_uniform_(self.embedings.weight, nonlinearity="linear")
        elif init_emb == 7:
            torch.nn.init.orthogonal_(self.embedings.weight)
        else:
            raise ValueError("Unknown init_emb value")

    def forward(self, class_id: torch.Tensor):
        emb = self.embedings(class_id)
        mid = self.middle(emb)
        out = self.fc(mid)

        return out


def client_worker(params):
    """
    FedDPL local training with Dual Prototype Learning.
    """
    (
        _,
        device,
        model_state,
        pln_state,
        train_set,
        model_name,
        dataset_name,
        lr,
        batch_size,
        epochs,
        num_classes,
        lambda_,
        epoch_pln,
        lr_pln,
        batch_size_pln,
        feature_dim,
        depth_pln,
        width_pln,
        mode,
        fixed_proto,
        init_emb,
        har,
    ) = params

    # 1. 初始化核心模型与 PLN（原型网络）
    model = get_model(model_name, dataset_name, feature_dim).to(device)
    model.load_state_dict(model_state)

    pln = PLN(num_classes, width_pln, feature_dim, depth_pln, fixed_proto, init_emb).to(
        device
    )
    pln.load_state_dict(pln_state)

    all_classes = torch.arange(0, num_classes).to(device)

    # 2. 训练核心模型（特征提取器）
    avg_loss_m_m = 0.0
    avg_loss_m_p = 0.0
    if mode in ["model", "normal", "all"]:
        model.train()
        pln.eval()
        opt = torch.optim.SGD(model.parameters(), lr=lr)
        total_loss_m = 0.0
        total_loss_p = 0.0
        num_batches_m = 0
        loader = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, shuffle=True
        )

        for _ in range(epochs):
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                output, feature = model(x)
                loss_ce = ce_loss(output, y)

                # PLN 损失：促使特征向其对应类别的原型靠拢
                with torch.no_grad():
                    protos = pln(all_classes)
                loss_proto = mse_loss(feature, protos[y])

                loss = loss_ce + lambda_ * loss_proto

                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss_m += loss_ce.item()
                total_loss_p += loss_proto.item()
                num_batches_m += 1

        avg_loss_m_m = total_loss_m / num_batches_m if num_batches_m > 0 else 0.0
        avg_loss_m_p = total_loss_p / num_batches_m if num_batches_m > 0 else 0.0

    # 3. 训练 PLN 网络（优化类原型）
    avg_loss_p = 0.0
    if mode in ["pln", "normal", "all"]:
        model.eval()
        pln.train()
        opt_pln = torch.optim.SGD(pln.parameters(), lr=lr_pln)
        total_loss_p = 0.0
        num_batches_p = 0

        loader_pln = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size_pln, shuffle=True
        )

        for _ in range(epoch_pln):
            for x, y in loader_pln:
                x, y = x.to(device), y.to(device)
                protos = pln(all_classes)

                with torch.no_grad():
                    _, feature = model(x)

                # 更新原型，使其更贴近所在类的实例特征
                loss = mse_loss(feature, protos[y])

                opt_pln.zero_grad()
                loss.backward()
                opt_pln.step()
                total_loss_p += loss.item()
                num_batches_p += 1

        avg_loss_p = total_loss_p / num_batches_p if num_batches_p > 0 else 0.0

    model_state = {k: v.cpu() for k, v in model.state_dict().items()}
    pln_state = {k: v.cpu() for k, v in pln.state_dict().items()}
    return [avg_loss_m_m, avg_loss_m_p, avg_loss_p, model_state, pln_state]


class Server(BaseServer):
    def __init__(self, args: argparse.Namespace):
        super().__init__(True, args)

        self.pln = PLN(
            num_classes=self.num_class,
            width=args.width_pln,
            feature_dim=self.args.feature_dim,
            depth=args.depth_pln,
            fixed=args.fixed_proto,
            init_emb=args.init_emb,
        )

        self.all_classes = torch.arange(0, self.num_class)
        self.loss_p: list[float] = []
        self.loss_m_m: list[float] = []
        self.loss_m_p: list[float] = []

    def fit(self):
        num_join_clients = int(self.num_clients * self.args.join_ratio)
        num_join_clients = max(1, num_join_clients)

        for r in range(self.rounds):
            t0 = time.time()
            print(f"\n--- FedDPL Round {r + 1}/{self.rounds} ---")

            selected_clients = np.random.choice(
                self.num_clients, num_join_clients, replace=False
            )
            print(f"Selected clients: {selected_clients}")

            p = [
                [
                    i,
                    self.client_gpu[i],
                    self.clients_state[i],
                    self.pln.state_dict(),
                    self.train_sets[i],
                    self.args.model,
                    self.args.dataset,
                    self.args.lr,
                    self.args.batch_size,
                    self.args.epochs,
                    self.num_class,
                    self.args.lambda_,
                    self.args.epoch_pln,
                    self.args.lr_pln,
                    self.args.batch_size_pln,
                    self.args.feature_dim,
                    self.args.depth_pln,
                    self.args.width_pln,
                    self.args.mode,
                    self.args.fixed_proto,
                    self.args.init_emb,
                    self.args.har,
                ]
                for i in selected_clients
            ]

            results = run_parallel_clients(
                client_worker=client_worker,
                parameters=p,
                gpu_pools=self.gpu_pools,
                mp=self.mp,
            )

            # 汇集各客户端回传结果，以增量方式计算加权平均损失
            total_loss_model = 0.0
            total_loss_model_m = 0.0
            total_loss_model_p = 0.0
            total_loss_pln = 0.0
            plns_states = []
            for i in selected_clients:
                client_loss_m_m, client_loss_m_p, client_loss_pln, client_state, client_pln = results[i]
                total_loss_model += client_loss_m_m + client_loss_m_p
                total_loss_model_m += client_loss_m_m
                total_loss_model_p += client_loss_m_p
                total_loss_pln += client_loss_pln
                self.clients_state[i] = client_state
                plns_states.append(client_pln)
            self.loss.append(total_loss_model / num_join_clients)
            self.loss_m_m.append(total_loss_model_m / num_join_clients)
            self.loss_m_p.append(total_loss_model_p / num_join_clients)
            self.loss_p.append(total_loss_pln / num_join_clients)

            # 聚合各客户端学习到的 PLN 参数
            self.aggregate(plns_states, weights=norm_weights)

            # 使用统一接口进行全量客户端测试验证
            protos_tensor = self.pln(self.all_classes)
            self.evaluate(protos=protos_tensor)

            print(f"Loss: {self.loss[-1]:.4f}, PLN Loss: {self.loss_p[-1]:.4f}")
            print(f"Acc: {self.acc[-1]:.4f}, PLN ACC: {self.acc_proto[-1]:.4f}")
            print(f"Round finished in {time.time() - t0:.2f} seconds")

    def aggregate(self, pln_params, weights):
        self.pln.load_state_dict(param_aggregate(pln_params, weights))

    def save(self):
        f = {
            "acc": {"model": self.acc, "proto": self.acc_proto},
            "loss": {
                "model": self.loss,
                "proto": self.loss_p,
                "aux": {"model_m": self.loss_m_m, "model_p": self.loss_m_p},
            },
            "state_dict": {
                "client": self.clients_state,
                "proto": self.pln.state_dict(),
            },
        }
        self.deal_save(f)
