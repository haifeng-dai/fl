import torch
import copy, os
import argparse

from .utils import (
    BaseClient,
    BaseServer,
    run_parallel_clients,
    compare_model_parameters,
)


def add_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("MOON Specific Arguments")
    group.add_argument(
        "--mu", type=float, default=1.0, help="Weight for contrastive loss"
    )
    group.add_argument(
        "--tau",
        type=float,
        default=0.5,
        help="Temperature parameter for contrastive loss",
    )
    return parser


class Client(BaseClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mu = self.args.mu
        self.tau = self.args.tau
        # MOON 需要两个额外的辅助模型
        self.global_model = copy.deepcopy(self.model).to(self.device)
        self.prev_model = copy.deepcopy(self.model).to(self.device)

        self.ce_moon = torch.nn.CosineSimilarity(dim=-1)

    def moon_loss(self, z, z_glob, z_prev):
        pos_sim = self.ce_moon(z, z_glob)
        neg_sim = self.ce_moon(z, z_prev)
        logits = torch.cat([pos_sim.reshape(-1, 1), neg_sim.reshape(-1, 1)], dim=1)
        logits /= self.tau
        labels = torch.zeros(z.size(0)).to(z.device).long()
        return self.ce(logits, labels)

    def train(self):
        self.model.train()
        self.global_model.eval()
        self.prev_model.eval()

        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.lr,
        )
        loss_ = []
        train_loader = self.build_train_loader()
        for _ in range(self.epochs):
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()

                y, z = self.model(data)
                with torch.no_grad():
                    _, z_glob = self.global_model(data)
                    _, z_prev = self.prev_model(data)

                loss_ce = self.ce(y, target)
                loss_con = self.moon_loss(z, z_glob, z_prev)
                loss = loss_ce + self.mu * loss_con

                loss.backward()
                optimizer.step()
                loss_.append(loss.item())
        self.prev_model.load_state_dict(self.model.state_dict())
        return sum(loss_) / len(loss_)

    def set_client(self, parameters):
        self.model.load_state_dict(parameters)
        self.global_model.load_state_dict(parameters)


class Server(BaseServer):
    def __init__(
        self,
        model: torch.nn.Module,
        args: argparse.Namespace,
    ):
        super().__init__(model, False, args)
        for i in range(args.num_clients):
            self.clients[i] = Client(
                client_id=i, model=model, train_set=self.train_sets[i], args=args
            )

    def fit(self):
        for r in range(self.rounds):
            print(f"\n--- MOON Round {r + 1}/{self.rounds} ---")
            global_params = {k: v.cpu() for k, v in self.model.state_dict().items()}
            # clients_params_old = copy.deepcopy([self.clients[i].model.state_dict() for i in range(self.num_clients)])

            # 准备每个客户端的个性化参数包
            parameters_per_client = [global_params] * self.num_clients

            results = run_parallel_clients(
                clients=self.clients,
                parameters=parameters_per_client,
                gpu_pools=self.gpu_pools,
                no_mp=self.no_mp,
            )
            loss_avg = sum(results) / len(results)
            self.loss.append(loss_avg)
            # updated = [compare_model_parameters(clients_params_old[i], self.clients[i].prev_model.state_dict()) for i in range(self.num_clients)]
            # print(updated)
            # updated_1 = [compare_model_parameters(global_params, self.clients[i].model.state_dict()) for i in range(self.num_clients)]

            clients_params = [
                self.clients[i].model.state_dict() for i in range(self.num_clients)
            ]
            self.aggregate(clients_params, weights=self.weights)
            self.evaluate()
            print(f"Global Accuracy: {self.acc[-1]:.2f}%, Avg Loss: {loss_avg:.4f}")
            # updated_2 = compare_model_parameters(global_params, self.model.state_dict())
            # updated_3 = [compare_model_parameters(clients_params_old[i], clients_params[i]) for i in range(self.num_clients)]
            # print(f"Updated Parameters: \nupdated: \n{updated}, \nupdated_1: \n{updated_1}, \nupdated_2: \n{updated_2}, \nupdated_3: \n{updated_3}")

    def save(self, test):
        file_name: str = f"{self.args.epochs}_{self.args.batch_size}_{self.args.lr}.pt"
        f = {"acc": self.acc, "loss": self.loss, "state_dict": self.model.state_dict()}
        super().deal_save(test, f, file_name)
