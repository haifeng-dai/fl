import os
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from .utils import (
    BaseParams,
    BaseServer,
    dist_contrastive_loss,
    fmt_num,
    get_model,
    param_aggregate,
)


def get_path(args):
    args.file_name = f"{args.common_name}_{fmt_num(args.lambda_)}_{fmt_num(args.epoch_pln)}_{fmt_num(args.lr_pln)}_{fmt_num(args.batch_size_pln)}_{fmt_num(args.depth_pln)}_{fmt_num(args.width_pln)}_{args.mode}_{fmt_num(args.fixed_proto)}_{fmt_num(args.init_emb)}_{fmt_num(args.har)}"
    return os.path.join(args.log_path, f"{args.file_name}_{args.cur_time}.log")


@dataclass
class Params(BaseParams):
    pln_state: dict[str, torch.Tensor]
    lambda_: float
    epoch_pln: int
    lr_pln: float
    batch_size_pln: int
    depth_pln: int
    width_pln: int
    mode: str
    fixed_proto: int
    init_emb: int
    har: int


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


def train(p: Params):
    """
    带有原型学习网络 (PLN) 的 FedPLN 本地训练流程。
    """
    device = torch.device(p.client_gpu)

    # 1. 初始化模型与 PLN 网络
    model = get_model(p.model_name, p.dataset, p.num_class, p.feature_dim).to(device)
    model.load_state_dict(p.model_state)
    pln = PLN(p.num_class, p.width_pln, p.feature_dim, p.depth_pln, p.fixed_proto, p.init_emb)
    pln.to(device)
    pln.load_state_dict(p.pln_state)
    all_classes = torch.arange(0, p.num_class).to(device)

    # 2. 阶段一：训练核心模型（特征提取器）
    avg_loss_m = 0.0
    model.train()
    pln.eval()
    opt = torch.optim.SGD(model.parameters(), lr=p.lr)
    total_loss_m = 0.0
    num_batches_m = 0
    loader = torch.utils.data.DataLoader(p.train_set, batch_size=p.batch_size, shuffle=True)

    # 原型损失：特征向量与 PLN 对应原型之间的欧式距离
    with torch.no_grad():
        protos = pln(all_classes)
    for _ in range(p.epochs):
        for x, y, *_ in loader:
            x, y = x.to(device), y.to(device)
            feature = model.extractor(x)
            output = model.classifier(feature)
            loss_ce = F.cross_entropy(output, y)

            loss_proto = dist_contrastive_loss(feature, protos, y)

            loss = loss_ce + p.lambda_ * loss_proto

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss_m += loss.item()
            num_batches_m += 1

    avg_loss_m = total_loss_m / num_batches_m if num_batches_m > 0 else 0.0

    # 3. 阶段二：训练 PLN 网络（优化类原型）
    avg_loss_p = 0.0
    model.eval()
    pln.train()
    opt_pln = torch.optim.SGD(pln.parameters(), lr=p.lr_pln)
    total_loss_p = 0.0
    num_batches_p = 0

    if p.batch_size_pln != p.batch_size:
        loader_pln = torch.utils.data.DataLoader(
            p.train_set, batch_size=p.batch_size_pln, shuffle=True
        )
    else:
        loader_pln = loader

    for _ in range(p.epoch_pln):
        for x, y, *_ in loader_pln:
            x, y = x.to(device), y.to(device)
            protos = pln(all_classes)

            with torch.no_grad():
                feature = model.extractor(x)

            # 损失计算：基于样本到原型距离的交差熵分类损失
            loss = dist_contrastive_loss(feature, protos, y)

            opt_pln.zero_grad()
            loss.backward()
            opt_pln.step()
            total_loss_p += loss.item()
            num_batches_p += 1

        avg_loss_p = total_loss_p / num_batches_p

    model_state = {k: v.cpu().detach().clone() for k, v in model.state_dict().items()}
    pln_state = {k: v.cpu().detach().clone() for k, v in pln.state_dict().items()}
    return {
        "loss": avg_loss_m,
        "loss_proto": avg_loss_p,
        "state": model_state,
        "pln_state": pln_state,
    }


class Server(BaseServer):
    def __init__(self, args):
        super().__init__(args)
        self.width_pln = args.width_pln
        self.depth_pln = args.depth_pln
        self.fixed_proto = args.fixed_proto
        self.init_emb = args.init_emb
        self.lambda_ = args.lambda_
        self.epoch_pln = args.epoch_pln
        self.lr_pln = args.lr_pln
        self.batch_size_pln = args.batch_size_pln
        self.mode = args.mode
        self.har = args.har

        self.pln = PLN(
            num_classes=self.num_class,
            width=self.width_pln,
            feature_dim=self.feature_dim,
            depth=self.depth_pln,
            fixed=self.fixed_proto,
            init_emb=self.init_emb,
        )
        self.all_classes = torch.arange(0, self.num_class)
        self.loss_p: list[float] = []

    def fit(self):
        num_join = max(1, int(self.num_clients * self.join_ratio))

        for r in range(self.rounds):
            t0 = time.time()
            print(f"\n--- FedPLN Round {r + 1}/{self.rounds} ---")

            selected = torch.randperm(self.num_clients)[:num_join].tolist()
            print(f"Selected clients: {selected}")

            p = [
                Params(
                    **asdict(base),
                    pln_state=self.pln.state_dict(),
                    lambda_=self.lambda_,
                    epoch_pln=self.epoch_pln,
                    lr_pln=self.lr_pln,
                    batch_size_pln=self.batch_size_pln,
                    depth_pln=self.depth_pln,
                    width_pln=self.width_pln,
                    mode=self.mode,
                    fixed_proto=self.fixed_proto,
                    init_emb=self.init_emb,
                    har=self.har,
                )
                for base in self.build_base_params(selected)
            ]
            results = self.run_clients(train, p)

            # 汇集各客户端的回传结果，计算模型与 PLN 的加权整体损失
            total_loss_model = 0.0
            total_loss_pln = 0.0
            selected_states = []
            selected_plns = []
            current_weights = []
            for cid, res in results.items():
                total_loss_model += res["loss"]
                total_loss_pln += res["loss_proto"]
                selected_states.append(res["state"])
                selected_plns.append(res["pln_state"])
                current_weights.append(self.weights[cid])
            self.loss.append(total_loss_model / num_join)
            self.loss_p.append(total_loss_pln / num_join)
            sum_weights = sum(current_weights)
            norm_weights = [w / sum_weights for w in current_weights]

            self.aggregate(selected_states, selected_plns, weights=norm_weights)
            # 直接使用聚合后的 PLN 模块输出作为当前全局原型进行评估
            protos_tensor = self.pln(self.all_classes)
            self.evaluate(protos=protos_tensor)

            print(f"Acc: {self.acc[-1]:.4f}, PLN ACC: {self.acc_proto[-1]:.4f}")
            print(f"Round finished in {time.time() - t0:.2f} seconds")

    def aggregate(self, clients_params, plns_params, weights):
        super().aggregate(clients_params, weights)
        self.pln.load_state_dict(param_aggregate(plns_params, weights))

    def save(self):
        metrics = {
            "acc": self.acc,
            "acc_p": self.acc_proto,
            "loss": self.loss,
            "loss_p": self.loss_p,
        }
        params = {"global": self.model.state_dict(), "proto": self.pln.state_dict()}
        self.deal_save(metrics, params)
