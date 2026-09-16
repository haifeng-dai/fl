from .core import BaseServer
from .local import Client


class Server(BaseServer):
    client_cls = Client

    def apply_result(self, results):
        self.aggregate_model(results)
