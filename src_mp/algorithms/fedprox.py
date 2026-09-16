import torch

from .core import BaseClientExecutor, BaseServer, ClientResult, ce_loss, clone_state


class Client(BaseClientExecutor):
    def train(self):
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
    client_cls = Client

    def __init__(self, args, devices):
        super().__init__(args, devices)
        self.mu: float = args.mu

    def train_payloads(self):
        return {client_id: self.mu for client_id in self.selected}

    def apply_result(self, results):
        self.aggregate_model(results)
