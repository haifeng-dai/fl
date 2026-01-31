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


class DCL(torch.nn.Module):
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(
        self, feature: torch.Tensor, protos: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        feature_norm = F.normalize(feature, dim=1)
        protos_norm = F.normalize(protos, dim=1)
        sim = torch.mm(feature_norm, protos_norm.t()) / self.temperature
        batch_indices = torch.arange(feature.size(0), device=feature.device)
        pos_sim = sim[batch_indices, y]
        mask = torch.ones_like(sim, dtype=torch.bool)
        mask[batch_indices, y] = False
        neg_sim = sim[mask].view(feature.size(0), -1)
        loss = -pos_sim + torch.logsumexp(neg_sim, dim=1)
        return loss.mean()


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

    # 1. Initialize Model and PLN (Prototype Learning Network)
    model = get_model(model_name, dataset_name, feature_dim).to(device)
    model.load_state_dict(model_state)

    pln = PLN(num_classes, width_pln, feature_dim, depth_pln, fixed_proto, init_emb).to(
        device
    )
    pln.load_state_dict(pln_state)

    all_classes = torch.arange(0, num_classes).to(device)

    # 2. Train Model (Feature Extractor)
    avg_loss_m = 0.0
    if mode in ["model", "normal", "all"]:
        model.train()
        pln.eval()
        opt = torch.optim.SGD(model.parameters(), lr=lr)
        total_loss_m = 0.0
        num_batches_m = 0
        loader = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, shuffle=True
        )

        for _ in range(epochs):
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                output, feature = model(x)
                loss_ce = ce_loss(output, y)

                # PLN Loss: Encourage features to be close to their class prototypes
                with torch.no_grad():
                    protos = pln(all_classes)
                dist = torch.cdist(feature, protos, p=2) ** 2
                loss_proto = ce_loss(-torch.sqrt(dist), y)

                loss = loss_ce + lambda_ * loss_proto

                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss_m += loss.item()
                num_batches_m += 1

        avg_loss_m = total_loss_m / num_batches_m if num_batches_m > 0 else 0.0

    # 3. Train PLN (Prototypes)
    avg_loss_p = 0.0
    if mode in ["pln", "normal", "all"]:
        model.eval()
        pln.train()
        opt_pln = torch.optim.SGD(pln.parameters(), lr=lr_pln)
        total_loss_p = 0.0
        num_batches_p = 0

        # Use separate loader if batch sizes differ, otherwise reuse or create new
        if batch_size_pln != batch_size:
            loader_pln = torch.utils.data.DataLoader(
                train_set, batch_size=batch_size_pln, shuffle=True
            )
        else:
            # Re-create loader to ensure full shuffle traversal for PLN epochs
            loader_pln = torch.utils.data.DataLoader(
                train_set, batch_size=batch_size, shuffle=True
            )

        for _ in range(epoch_pln):
            for x, y in loader_pln:
                x, y = x.to(device), y.to(device)
                protos = pln(all_classes)

                with torch.no_grad():
                    _, feature = model(x)

                # Update prototypes to be closer to features
                dist = torch.cdist(feature, protos, p=2) ** 2
                loss = ce_loss(-torch.sqrt(dist), y)

                opt_pln.zero_grad()
                loss.backward()
                opt_pln.step()
                total_loss_p += loss.item()
                num_batches_p += 1

        avg_loss_p = total_loss_p / num_batches_p if num_batches_p > 0 else 0.0

    model_state = {k: v.cpu() for k, v in model.state_dict().items()}
    pln_state = {k: v.cpu() for k, v in pln.state_dict().items()}
    return [avg_loss_m, avg_loss_p, model_state, pln_state]


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
        self.clients_state = {
            i: self.model.state_dict() for i in range(self.num_clients)
        }

        self.all_classes = torch.arange(0, self.pln.embedings.num_embeddings)
        self.acc_p: list[float] = []
        self.loss_p: list[float] = []

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

            # Calculate average losses using incremental summation
            total_loss_model = 0.0
            total_loss_pln = 0.0
            plns_states = []
            current_weights = []
            for i in selected_clients:
                total_loss_model += results[i][0]
                total_loss_pln += results[i][1]
                self.clients_state[i] = results[i][2]
                plns_states.append(results[i][3])
                current_weights.append(self.weights[i])
            self.loss.append(total_loss_model / num_join_clients)
            self.loss_p.append(total_loss_pln / num_join_clients)
            sum_weights = sum(current_weights)
            norm_weights = [w / sum_weights for w in current_weights]

            # Aggregate PLN parameters
            self.aggregate(plns_states, weights=norm_weights)

            self.evaluate()
            print(f"Acc: {self.acc[-1]:.4f}, PLN ACC: {self.acc_p[-1]:.4f}")
            print(f"Round finished in {time.time() - t0:.2f} seconds")

    def aggregate(self, pln_params, weights=None):
        self.pln.load_state_dict(param_aggregate(pln_params, weights))

    def evaluate(self):
        current_acc = []
        current_acc_p = []

        # Global PLN prototype for evaluation
        prototype = self.pln(self.all_classes)

        for i in range(self.num_clients):
            # Load local model
            self.model.load_state_dict(self.clients_state[i])

            # Evaluate model accuracy on local test set
            acc = evaluate_model(self.model, self.test_set[i], self.device)
            current_acc.append(acc)

            # Evaluate prototype accuracy on local test set
            acc_p = evaluate_prototype(
                self.model, prototype, self.test_set[i], self.device
            )
            current_acc_p.append(acc_p)

        self.acc.append(sum(current_acc) / len(current_acc))
        self.acc_p.append(sum(current_acc_p) / len(current_acc_p))

    def save(self):
        f = {
            "acc": {"model": self.acc, "prototype": self.acc_p},
            "loss": {"model": self.loss, "prototype": self.loss_p},
            "state_dict": {
                "model": self.clients_state,
                "pln": self.pln.state_dict(),
            },
        }
        self.deal_save(f)
