import torch
import copy,os
import argparse

from .utils import BaseClient, BaseServer, run_parallel_clients


def add_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("MOON Specific Arguments")
    group.add_argument("--mu", type=float, default=1.0, help="Weight for contrastive loss")
    group.add_argument("--tau", type=float, default=0.5, help="Temperature parameter for contrastive loss")
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

        model_state = {k: v.detach().clone().cpu() for k, v in self.model.state_dict().items()}
        return sum(loss_) / len(loss_), model_state

    def set_client(self, parameters):
        global_params, prev_local_params = parameters
        # 加载全局参数到本地模型和全局模型副本
        self.model.load_state_dict(global_params)
        self.global_model.load_state_dict(global_params)

        # 加载上轮本地参数
        if prev_local_params is not None:
            self.prev_model.load_state_dict(prev_local_params)
        else:
            self.prev_model.load_state_dict(global_params)

class Server(BaseServer):
    def __init__(
            self,
            model: torch.nn.Module,
            args: argparse.Namespace,
    ):
        super().__init__(model, False, args)
        for i in range(args.num_clients):
            self.clients[i] = Client(
                client_id=i,
                model=model,
                train_set=self.train_sets[i],
                args=args
            )

    def fit(self):
        prev_local_params_list = [None] * len(self.clients)

        for r in range(self.rounds):
            print(f"\n--- MOON Round {r + 1}/{self.rounds} ---")
            global_params = {k: v.cpu() for k, v in self.model.state_dict().items()}

            # 准备每个客户端的个性化参数包
            parameters_per_client = [
                (global_params, prev_local_params_list[i]) for i in range(self.num_clients)
            ]

            results = run_parallel_clients(
                clients=self.clients,
                parameters=parameters_per_client,
                gpu_pools=self.gpu_pools,
                no_mp=self.no_mp
            )

            loss_epoch = [res[0] for res in results]
            client_dicts = [res[1] for res in results]

            loss_avg = sum(loss_epoch) / len(loss_epoch)
            prev_local_params_list = client_dicts
            self.aggregate(client_dicts, weights=self.weights)
            acc = self.evaluate()
            self.acc.append(acc)
            self.loss.append(loss_avg)
            print(f"Global Accuracy: {acc:.2f}%, Avg Loss: {loss_avg:.4f}")

    def save(self, test):
        file_name: str = f"{self.args.epochs}_{self.args.batch_size}_{self.args.lr}.pt"
        f = {
            "acc": self.acc,
            "loss": self.loss,
            "state_dict": self.model.state_dict()
        }
        super().deal_save(test, f, file_name)
