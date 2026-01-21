import torch
import argparse

from .utils import (
    BaseServer,
    run_parallel_clients,
    evaluate_prototype,
    param_aggregate,
    ce_loss,
    get_model,
    evaluate_model,
)


def add_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("FedPLN Specific Arguments")
    group.add_argument(
        "--lambda_", type=float, default=10.0, help="Weight for PLN Contrastive Loss"
    )
    group.add_argument(
        "--epoch_pln", type=int, default=10, help="Epochs for PLN learning"
    )
    group.add_argument(
        "--lr_pln", type=float, default=0.01, help="Learning rate for PLN learning"
    )
    group.add_argument(
        "--batch_size_pln", type=int, default=32, help="Batch size for PLN learning"
    )
    group.add_argument("--feature_dim", type=int, default=128, help="Feature dimension")
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
        "--har", type=int, default=0, help="Whether to use HAR dataset"
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


def client_worker(client_id, params):
    device = params[0]
    model_state = params[1]
    pln_state = params[2]
    train_set = params[3]

    model_name = params[4]
    dataset_name = params[5]
    lr = params[6]
    batch_size = params[7]
    epochs = params[8]

    num_classes = params[9]
    width_pln = params[10]
    feature_dim = params[11]
    depth_pln = params[12]
    fixed_proto = params[13]
    init_emb = params[14]
    lambda_ = params[15]
    lr_pln = params[16]
    epoch_pln = params[17]

    model = get_model(model_name, dataset_name).to(device)
    model.load_state_dict(model_state)

    pln = PLN(
        num_classes, width_pln, feature_dim, depth_pln, fixed_proto, init_emb
    ).to(device)
    pln.load_state_dict(pln_state)

    all_classes = torch.arange(0, num_classes).to(device)

    # Train Model
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

    avg_loss_m = total_loss_m / num_batches_m

    # Train PLN
    model.eval()
    pln.train()
    opt_pln = torch.optim.SGD(pln.parameters(), lr=lr_pln)
    total_loss_p = 0.0
    num_batches_p = 0

    for _ in range(epoch_pln):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            protos = pln(all_classes)

            with torch.no_grad():
                _, feature = model(x)
            dist = torch.cdist(feature, protos, p=2) ** 2
            loss = ce_loss(-torch.sqrt(dist), y)

            opt_pln.zero_grad()
            loss.backward()
            opt_pln.step()
            total_loss_p += loss.item()
            num_batches_p += 1

    avg_loss_p = total_loss_p / num_batches_p

    return client_id, [avg_loss_m, avg_loss_p, model.state_dict(), pln.state_dict()]


class Server(BaseServer):
    def __init__(self, args: argparse.Namespace):
        model = get_model(args.model, args.dataset)
        super().__init__(model, False, args)
        self.pln = PLN(
            num_classes=self.num_class,
            width=args.width_pln,
            feature_dim=args.feature_dim,
            depth=args.depth_pln,
            fixed=args.fixed_proto,
            init_emb=args.init_emb,
        ).to(self.device)
        self.all_classes = torch.arange(0, self.pln.embedings.num_embeddings).to(
            self.device
        )
        self.acc_p: list[float] = []
        self.loss_p: list[float] = []

    def fit(self):
        for r in range(self.rounds):
            print(f"\n--- FedPLN Round {r + 1}/{self.rounds} ---")
            global_params = {k: v.cpu() for k, v in self.model.state_dict().items()}
            pln_params = {k: v.cpu() for k, v in self.pln.state_dict().items()}

            parameters_per_client = []
            for i in range(self.num_clients):
                p = [
                    self.client_gpu[i],
                    global_params,
                    pln_params,
                    self.train_sets[i],
                    self.args.model,
                    self.args.dataset,
                    self.args.lr,
                    self.args.batch_size,
                    self.args.epochs,
                    self.num_class,
                    self.args.width_pln,
                    self.args.feature_dim,
                    self.args.depth_pln,
                    self.args.fixed_proto,
                    self.args.init_emb,
                    self.args.lambda_,
                    self.args.lr_pln,
                    self.args.epoch_pln
                ]
                parameters_per_client.append(p)

            results = run_parallel_clients(
                client_worker=client_worker,
                num_clients=self.num_clients,
                parameters=parameters_per_client,
                gpu_pools=self.gpu_pools,
                mp=self.mp,
            )

            # Calculate average losses using incremental summation
            total_loss_model = 0.0
            total_loss_pln = 0.0
            for res in results:
                total_loss_model += res[0]
                total_loss_pln += res[1]
            self.loss.append(total_loss_model / self.num_clients)
            self.loss_p.append(total_loss_pln / self.num_clients)

            # Update global model and PLN
            clients_params = [res[2] for res in results]
            plns_params = [res[3] for res in results]
            
            self.aggregate(clients_params, plns_params)
            self.evaluate()

            print(f"Acc: {self.acc[-1]:.4f}, PLN ACC: {self.acc_p[-1]:.4f}")

    def aggregate(self, clients_params=None, plns_params=None, *args, **kwargs):
        # Handle optional arguments or direct passing
        if clients_params:
            self.model.load_state_dict(param_aggregate(clients_params, self.weights))
        if plns_params:
            self.pln.load_state_dict(param_aggregate(plns_params, self.weights))

    def evaluate(self):
        # Evaluate global model
        self.acc.append(evaluate_model(self.model, self.test_set, self.device))
        
        # Evaluate prototype
        prototype = self.pln(self.all_classes)
        acc_p = evaluate_prototype(self.model, prototype, self.test_set, self.device)
        self.acc_p.append(acc_p)

    def save(self, test):
        file_name: str = f"{self.args.epochs}_{self.args.batch_size}_{self.args.lr}.pt"
        super().deal_save(test, self.model.state_dict(), file_name)
