import copy
from pydoc import cli
import torch
import argparse

from .utils import BaseClient, BaseServer, run_parallel_clients, evaluate_prototype, param_aggregate


def add_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("FedPLN Specific Arguments")
    group.add_argument("--lambda_", type=float, default=10.0, help="Weight for PLN Contrastive Loss")
    group.add_argument("--epoch_pln", type=int, default=10, help="Epochs for PLN learning")
    group.add_argument("--lr_pln", type=float, default=0.01, help="Learning rate for PLN learning")
    group.add_argument("--batch_size_pln", type=int, default=32, help="Batch size for PLN learning")
    group.add_argument("--feature_dim", type=int, default=128, help="Feature dimension")
    group.add_argument("--depth_pln", type=int, default=2, help="Depth of PLN network")
    group.add_argument("--width_pln", type=int, default=128, help="Width of PLN network")
    group.add_argument("--mode", type=str, default="normal", choices=["normal", "pln", "model", "all"], help="Task mode")
    group.add_argument("--har", type=bool, default=False, help="Whether to use HAR dataset")
    group.add_argument("--fixed_proto", type=bool, default=False, help="Whether to fix the prototypes during training")
    group.add_argument("--init_emb", type=int, default=0, help="Initialization strategy for PLN embeddings")
    return parser


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
            torch.nn.Sequential(
                torch.nn.Linear(width, width), torch.nn.ReLU()
            ) for _ in range(depth)
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
            torch.nn.init.kaiming_uniform_(self.embedings.weight, nonlinearity='linear')
        elif init_emb == 7:
            torch.nn.init.orthogonal_(self.embedings.weight)
        else:
            raise ValueError("Unknown init_emb value")

    def forward(self, class_id: torch.Tensor):
        emb = self.embedings(class_id)
        mid = self.middle(emb)
        out = self.fc(mid)

        return out


class Client(BaseClient):
    def __init__(
            self,
            pln: PLN,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.lambda_ = self.args.lambda_
        self.epoch_pln = self.args.epoch_pln
        self.lr_pln = self.args.lr_pln
        self.batch_size_pln = self.args.batch_size_pln
        self.mode = self.args.mode
        self.har = self.args.har
        self.pln = copy.deepcopy(pln).to(self.device)
        self.all_classes = torch.arange(0, self.pln.embedings.num_embeddings).to(self.device)

    def train(self, *args, **kwargs):
        train_loader = self.build_train_loader()
        loss_m = self.train_model(train_loader)
        loss_p = self.pln_learning(train_loader)
        return loss_m, loss_p

    def train_model(self, train_loader):
        self.model.train()
        self.pln.eval()
        opt = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        loss_ = []
        for _ in range(self.epochs):
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                output, feature = self.model(x)
                loss_ce = self.ce(output, y)

                protos = self.pln(self.all_classes).detach()
                dist = torch.cdist(feature, protos, p=2) ** 2
                loss_proto = self.ce(-torch.sqrt(dist), y)

                loss = loss_ce + self.lambda_ * loss_proto

                opt.zero_grad()
                loss.backward()
                opt.step()
                loss_.append(loss.item())
        return sum(loss_) / len(loss_)

    def pln_learning(self, train_loader):
        self.model.eval()
        self.pln.train()
        opt_pln = torch.optim.SGD(self.pln.parameters(), lr=self.lr_pln)
        loss_ = []
        for _ in range(self.epoch_pln):
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                protos = self.pln(self.all_classes)

                _, feature = self.model(x)
                dist = torch.cdist(feature.detach(), protos, p=2) ** 2
                loss = self.ce(-torch.sqrt(dist), y)

                opt_pln.zero_grad()
                loss.backward()
                opt_pln.step()
                loss_.append(loss.item())
        return sum(loss_) / len(loss_)

    def evaluate(self, test_set, *args, **kwargs):
        prototype = self.pln(self.all_classes)
        acc = super().evaluate(test_set, *args, **kwargs)
        acc_p = evaluate_prototype(self.model, prototype, test_set, self.device)
        return acc, acc_p

    def set_client(self, parameters):
        model_params, pln_params = parameters
        self.model.load_state_dict(model_params)
        self.pln.load_state_dict(pln_params)


class Server(BaseServer):
    def __init__(
            self,
            model: torch.nn.Module,
            args: argparse.Namespace
    ):
        super().__init__(model, True, args)
        self.pln = PLN(
            num_classes=self.num_class,
            width=args.width_pln,
            feature_dim=args.feature_dim,
            depth=args.depth_pln,
            fixed=args.fixed_proto,
            init_emb=args.init_emb
        ).to(self.device)
        for i in range(args.num_clients):
            self.clients[i] = Client(
                client_id=i,
                model=model,
                train_set=self.train_sets[i],
                pln=self.pln,
                args=args
            )
        self.acc_p: list[float] = []
        self.loss_p: list[float] = []

    def fit(self):
        model_param = {k: v.cpu() for k, v in self.model.state_dict().items()}
        pln_param = {k: v.cpu() for k, v in self.pln.state_dict().items()}
        parameters_per_client = [(model_param, pln_param)] * self.num_clients
        for r in range(self.rounds):
            print(f"\n--- FedDPL Round {r + 1}/{self.rounds} ---")

            results = run_parallel_clients(
                clients=self.clients,
                parameters=parameters_per_client,
                gpu_pools=self.gpu_pools,
                no_mp=self.no_mp
            )
            loss_model_epoch = [res[0] for res in results]
            loss_pln_epoch = [res[1] for res in results]
            self.loss.append(sum(loss_model_epoch) / len(loss_model_epoch))
            self.loss_p.append(sum(loss_pln_epoch) / len(loss_pln_epoch))

            # # Update global PLN
            clients_params = [self.clients[i].model.state_dict() for i in range(self.num_clients)]
            plns_params = [self.clients[i].pln.state_dict() for i in range(self.num_clients)]  # type: ignore

            # Aggregate model parameters
            self.aggregate(plns_params)

            parameters_per_client = [
                (clients_params[i], self.pln.state_dict()) for i in range(self.num_clients)
            ]

            self.evaluate(clients_params, plns_params)

            print(f"Acc: {self.acc[-1]:.4f}, PLN ACC: {self.acc_p[-1]:.4f}")

    def aggregate(self, pln_params, *args, **kwargs):
        self.pln.load_state_dict(param_aggregate(pln_params, self.weights))

    def evaluate(self, clients_state, plns_state, *args, **kwargs):
        current_acc = []
        current_acc_p = []
        for client_id in self.clients:
            self.clients[client_id].set_client((
                clients_state[client_id], plns_state[client_id]
            ))
            acc, acc_p = self.clients[client_id].evaluate(
                self.test_set[client_id]
            )
            current_acc.append(acc)
            current_acc_p.append(acc_p)
        self.acc.append(sum(current_acc) / len(current_acc))
        self.acc_p.append(sum(current_acc_p) / len(current_acc_p))

    def save(self, test):
        file_name: str = f"{self.args.epochs}_{self.args.batch_size}_{self.args.lr}.pt"
        super().deal_save(test, self.model.state_dict(), file_name)
