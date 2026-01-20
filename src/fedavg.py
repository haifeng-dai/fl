import argparse
import copy

import torch

from .utils import BaseServer, run_parallel_clients, evaluate_model, ce_loss, get_model


# class Client(BaseClient):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     def train(self):
#         self.model.train()
#         optimizer = torch.optim.SGD(
#             self.model.parameters(),
#             lr=self.lr,
#         )
#         loss_ = []
#         train_loader = self.build_train_loader()
#         for _ in range(self.epochs):
#             for data, target in train_loader:
#                 data, target = data.to(self.device), target.to(self.device)
#                 optimizer.zero_grad()
#                 output, _ = self.model(data)
#                 loss = self.ce(output, target)
#                 loss.backward()
#                 optimizer.step()
#                 loss_.append(loss.item())
#         return sum(loss_) / len(loss_)

#     def set_client(self, parameters):
#         self.model.load_state_dict(parameters)


class Server(BaseServer):
    def __init__(self, args: argparse.Namespace):
        model = get_model(args.model, args.dataset)
        super().__init__(model, False, args)
        # for i in range(args.num_clients):
        #     self.clients[i] = Client(
        #         client_id=i, model=model, train_set=self.train_sets[i], args=args
        #     )

    def fit(self):
        for r in range(self.rounds):
            print(f"\n--- FedAvg Round {r + 1}/{self.rounds} ---")
            global_params = {k: v.cpu() for k, v in self.model.state_dict().items()}

            parameters_per_client = [
                [
                    v,
                    global_params,
                    self.train_sets[k],
                    self.args.model,
                    self.args.dataset,
                    self.args.lr,
                    self.args.batch_size,
                    self.args.epochs,
                ]
                for k, v in self.client_gpu.items()
            ]

            results = run_parallel_clients(
                client_worker=client_worker,
                num_clients=self.num_clients,
                parameters=parameters_per_client,
                gpu_pools=self.gpu_pools,
                no_mp=self.no_mp,
            )
            avg_loss = sum(results[0]) / self.num_clients
            self.loss.append(avg_loss)

            clients_params = [results[1][i] for i in range(self.num_clients)]
            self.aggregate(clients_params, weights=self.weights)
            self.evaluate()
            print(f"Global Accuracy: {self.acc[-1]:.2f}%, Avg Loss: {avg_loss:.4f}")

    def evaluate(self):
        self.acc.append(evaluate_model(self.model, self.test_set, self.device))

    def save(self, test):
        f = {
            "acc": self.acc,
            "loss": self.loss,
            "state_dict": self.model.state_dict(),
        }
        super().deal_save(test, f)


def client_worker(client_id, params):
    device = params[0]
    param_state = params[1]
    model = get_model(params[3], params[4])
    model = copy.deepcopy(model).to(device)
    model.load_state_dict(param_state)
    optimizer = torch.optim.SGD(model.parameters(), lr=params[5])
    loader = torch.utils.data.DataLoader(
        params[2],
        batch_size=params[6],
        shuffle=True,
    )
    loss = []
    for _ in range(params[7]):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            loss = ce_loss(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss.append(loss.item())
    return client_id, [sum(loss) / len(loss), model.state_dict()]
