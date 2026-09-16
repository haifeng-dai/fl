from .core import BaseClientExecutor, BaseServer, ce_loss


class Client(BaseClientExecutor):
    """Local 算法客户端：只执行本地监督训练，不参与全局聚合。"""

    def run_epoch(self, total: float, batches: int):
        """使用当前客户端数据执行一个本地 epoch。"""
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
    """Local Server，为每个客户端维护独立的模型状态。"""

    client_cls = Client
    personalized = True

    def apply_result(self, results):
        """将客户端结果写回各自状态，而不是聚合成一个全局模型。"""
        self.update_client_states(results, self.selected)
