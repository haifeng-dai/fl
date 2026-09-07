import time
from dataclasses import asdict

import torch

from .fedproto import Params, get_path, train
from .utils import BaseServer, proto_aggregate


class Server(BaseServer):
    def __init__(self, args):
        # 全局模型联邦学习，pfl=False（基类默认）
        super().__init__(args)
        self.mu = args.mu

        self.global_protos = None

    def fit(self):
        num_join = max(1, int(self.num_clients * self.join_ratio))

        for r in range(self.rounds):
            t0 = time.time()
            print(f"\n--- FedProto_G Round {r + 1}/{self.rounds} ---")

            selected = torch.randperm(self.num_clients)[:num_join].tolist()
            print(f"Selected clients: {selected}")

            # 默认将全局模型 self.model.state_dict() 分发给各客户端
            base_params = self.build_base_params(selected)

            p = [
                Params(
                    **asdict(base),
                    mu=self.mu,
                    global_protos=(
                        self.global_protos.cpu()
                        if self.global_protos is not None
                        else None
                    ),
                )
                for base in base_params
            ]
            results = self.run_clients(train, p)

            total_loss = 0.0
            selected_states = []
            selected_protos = []
            selected_counts = []
            current_weights = []
            for cid, res in results.items():
                total_loss += res["loss"]
                selected_states.append(res["state"])
                selected_protos.append(res["protos"])
                selected_counts.append(res["counts"])
                current_weights.append(self.weights[cid])
            self.loss.append(total_loss / num_join)

            # 1. 聚合全局模型参数：按客户端样本量加权平均
            sum_weights = sum(current_weights)
            norm_weights = [w / sum_weights for w in current_weights]
            self.aggregate(selected_states, weights=norm_weights)

            # 2. 聚合原型向量：按样本计数加权
            self.global_protos = proto_aggregate(
                selected_protos,
                local_counts_list=selected_counts,
                old_global_protos=self.global_protos,
            )

            # 3. 评估全局模型及原型精度
            self.evaluate(protos=self.global_protos)

            print(
                f"Global Accuracy: {self.acc[-1]:.2f}%, Proto Accuracy: {self.acc_proto[-1]:.2f}%, Avg Loss: {self.loss[-1]:.4f}"
            )
            print(f"Round finished in {time.time() - t0:.2f} seconds")

    def save(self):
        metrics = {"acc": self.acc, "acc_proto": self.acc_proto, "loss": self.loss}
        params = {"global": self.model.state_dict(), "proto": self.global_protos}
        self.deal_save(metrics, params)
