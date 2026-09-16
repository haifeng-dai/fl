from .core import BaseServer
from .local import Client


class Server(BaseServer):
    """FedAvg Server，使用客户端样本数加权聚合模型参数。

    FedAvg 不需要自定义 Client；继承基类客户端的本地监督训练即可。
    """

    client_cls = Client

    def apply_result(self, results):
        """将本轮客户端模型按训练样本数加权聚合到全局模型。"""
        self.aggregate_model(results)
