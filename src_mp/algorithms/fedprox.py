import torch

from .core import BaseClientExecutor, BaseServer, ClientResult, ce_loss, clone_state


class Client(BaseClientExecutor):
    """FedProx 客户端，在本地梯度中加入全局模型近端项。"""

    def train(self):
        """保存 Server 下发的 mu 和当前全局参数，再执行本地训练。"""
        self.mu: float = self.payload
        self.global_params = {
            name: parameter.detach().clone()
            for name, parameter in self.model.named_parameters()
        }
        self.model.train()
        total, batches = 0.0, 0
        for _ in range(self.epochs):
            total, batches = self.run_epoch(total, batches)
        return ClientResult(
            self.client_id,
            total / max(1, batches),
            clone_state(self.model.state_dict()),
        )

    def run_epoch(self, total, batches) -> tuple[float, int]:
        """执行一个 epoch，并将近端梯度项加入每个参数的梯度。"""
        for x, y, *_ in self.loader:
            loss = ce_loss(self.model(x.to(self.device)), y.to(self.device))
            self.check_nan(loss)
            self.optimizer.zero_grad()
            loss.backward()
            with torch.no_grad():
                for name, parameter in self.model.named_parameters():
                    if parameter.grad is not None:
                        parameter.grad.add_(
                            parameter - self.global_params[name], alpha=self.mu
                        )
            self.optimizer.step()
            total += loss.item()
            batches += 1
        return total, batches


class Server(BaseServer):
    """FedProx Server，通过训练 payload 向每个客户端传递 mu。"""

    client_cls = Client

    def __init__(self, args, devices):
        super().__init__(args, devices)
        self.mu: float = args.mu

    def train_payloads(self):
        """为当前选中的每个客户端构造相同的近端系数 payload。"""
        return {client_id: self.mu for client_id in self.selected}

    def apply_result(self, results):
        """按默认 FedAvg 权重聚合 FedProx 客户端返回的模型。"""
        self.aggregate_model(results)
