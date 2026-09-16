from .core import BaseClientExecutor, BaseServer, ce_loss


class Client(BaseClientExecutor):
    def run_epoch(self, total: float, batches: int):
        for x, y, *_ in self.loader:
            loss = ce_loss(self.model(x.to(self.device)), y.to(self.device))
            self.check_nan(loss)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total += loss.item()
            batches += 1
        return total, batches


class Server(BaseServer):
    client_cls = Client
    personalized = True

    def apply_result(self, results):
        self.update_client_states(results, self.selected)
