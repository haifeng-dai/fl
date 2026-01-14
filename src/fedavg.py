import torch.optim as optim
import torch
from src.utils.fed_utils import BaseClient, BaseServer, ClientInfo
from src.utils.parallel import run_parallel_clients


def add_args(parser):
    group = parser.add_argument_group("FedAvg Specific Arguments")
    group.add_argument("--weight_decay", type=float, default=1e-4)
    return parser


class FedAvgClient(BaseClient):
    def __init__(self, *args, weight_decay=1e-4, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_decay = weight_decay

    def train(self):
        self.model.train()
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        loss_ = []
        for epoch in range(self.epochs):
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output, _ = self.model(data)
                loss = self.ce(output, target)
                loss.backward()
                optimizer.step()
                loss_.append(loss.item())
        model_state = {k: v.detach().clone().cpu() for k, v in self.model.state_dict().items()}
        return sum(loss_) / len(loss_), model_state

    def set_client(self, parameters):
        self.model.load_state_dict(parameters)


class FedAvgServer(BaseServer):
    def __init__(
            self,
            model: torch.nn.Module,
            train_loader: dict[int, torch.utils.data.DataLoader],
            test_loader: torch.utils.data.DataLoader,
            clients_info: ClientInfo,
            rounds: int
    ):
        super().__init__(model, test_loader, clients_info, rounds)
        self.clients = {
            i: FedAvgClient(
                client_id=i,
                model=model,
                train_loader=train_loader[i],
                lr=clients_info.lr,
                epochs=clients_info.epochs,
                device=clients_info.cuda[i],
                weight_decay=clients_info.args.weight_decay if hasattr(clients_info.args, 'weight_decay') else 1e-4
            ) for i in range(len(clients_info.cuda))
        }

    def fit(self):
        loss = []
        for r in range(self.rounds):
            print(f"\n--- FedAvg Round {r + 1}/{self.rounds} ---")
            global_params = {k: v.cpu() for k, v in self.model.state_dict().items()}

            results = run_parallel_clients(
                clients=self.clients,
                parameters=global_params,
                gpu_pools=self.gpu_pools
            )
            loss_epoch = [res[0] for res in results]
            client_dicts = [res[1] for res in results]

            avg_loss = sum(loss_epoch) / len(loss_epoch)
            loss.append(avg_loss)
            self.aggregate(client_dicts)
            acc = self.evaluate()
            print(f"Global Accuracy: {acc:.2f}%, Avg Loss: {avg_loss:.4f}")
