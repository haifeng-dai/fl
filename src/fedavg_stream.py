import argparse

import torch

from .utils.fed_utils import StreamBaseClient, StreamBaseServer, param_aggregate


class Client(StreamBaseClient):
    """Stream FedAvg 客户端"""

    def train(self, *args, **kwargs) -> float:
        """训练一个 epoch"""
        print(self.device)
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)

        losses = []
        for data, target in self.train_loader:
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            output, _ = self.model(data)
            loss = self.ce(output, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        return sum(losses) / len(losses)


class Server(StreamBaseServer):
    """Stream FedAvg Server"""

    def __init__(self, model: torch.nn.Module, args: argparse.Namespace):
        super().__init__(model, False, args)
        assert isinstance(self.test_set, torch.utils.data.TensorDataset)
        for client_id in range(self.num_clients):
            self.clients[client_id] = Client(
                client_id=client_id,
                gpu_id=self.clients_gpu[client_id],
                model=model,
                train_set=self.test_set,
                args=self.args,
            )

    def fit(self):
        """执行联邦学习训练"""
        for r in range(self.rounds):
            print(f"\n--- FedAvg Round {r + 1}/{self.rounds} ---")

            # 下发参数并训练
            global_params = {k: v.cpu() for k, v in self.model.state_dict().items()}
            for client_id in range(self.num_clients):
                self.train_client(client_id, global_params)
            self.synchronize()

            # 收集所有客户端的参数
            client_params = []
            for client_id in range(self.num_clients):
                params = self.clients[client_id].get_parameters()
                client_params.append(params)

            # FedAvg 聚合
            self.aggregate(client_params, weights=self.weights)

            # 评估
            self.evaluate()

            avg_loss = self.loss[-1] if self.loss else 0
            print(f"Global Accuracy: {self.acc[-1]:.2f}%, Avg Loss: {avg_loss:.4f}")

    def save(self, test: bool):
        """保存结果"""
        f = {
            "acc": self.acc,
            "loss": self.loss,
            "state_dict": self.model.state_dict(),
        }
        self.deal_save(test, f)
