import torch
import torch.nn as nn
import torch.optim as optim

from src.models import SimpleCNN
from src.utils.fed_utils import BaseClient, BaseServer, load_client_data
from src.utils.parallel import run_parallel_clients


def add_args(parser):
    """
    Add FedAvg specific arguments to the parser.
    """
    group = parser.add_argument_group("FedAvg Specific Arguments")
    # FedAvg usually doesn't have unique hyperparameters in this demo,
    # but we can add placeholders or specific settings like weight decay.
    group.add_argument("--weight_decay", type=float, default=1e-4)
    return parser


class FedAvgClient(BaseClient):
    def train(self, global_params):
        dataloader = load_client_data(self.client_id, self.args.dataset, self.args.partition)
        model = SimpleCNN().to(self.device)
        model.load_state_dict(global_params)
        model.train()

        optimizer = optim.SGD(
            model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay
        )
        criterion = nn.CrossEntropyLoss()

        for epoch in range(self.args.epochs):
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output, _ = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
        return model.cpu().state_dict()


class FedAvgServer(BaseServer):
    def fit(self):
        eval_device = self.clients_info[0][1]
        for r in range(self.args.rounds):
            print(f"\n--- FedAvg Round {r + 1}/{self.args.rounds} ---")
            global_params = self.model.state_dict()

            # Pack simple dict for worker if needed,
            # or just pass the whole args object if it's picklable (it usually is)
            client_dicts = run_parallel_clients(
                FedAvgClient, self.clients_info, global_params, self.args
            )

            self.aggregate(client_dicts)
            acc = self.evaluate(eval_device)
            print(f"Global Accuracy: {acc:.2f}%")
