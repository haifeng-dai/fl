import argparse
import os
import time
from collections import defaultdict

import numpy as np
import torch

from .utils import (
    BaseServer,
    ce_loss,
    get_model,
    mse_loss,
)


def add_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("FedTest Specific Arguments")
    group.add_argument(
        "--mu_test",
        type=float,
        default=0.1,
        help="Weight for prototype consistency loss",
    )
    return parser


def get_path(args):
    args.file_name = f"{args.name_pre}_{args.mu_test}"
    return os.path.join(args.log_path, f"{args.file_name}_{args.times}.log")


def proto_cluster(protos_list):
    """按类别汇总聚合对于多个客户端的原型向量。"""
    proto_clusters = defaultdict(list)
    for protos in protos_list:
        for k, v in protos.items():
            proto_clusters[k].append(v)

    avg_protos = {}
    for k, v in proto_clusters.items():
        protos = torch.stack(v)
        avg_protos[k] = torch.mean(protos, dim=0).detach()

    return avg_protos


def compute_prototype_similarity(client_protos_list, num_classes):
    """
    计算客户端之间每个类别的原型的余弦相似度。

    Args:
        client_protos_list: 字典列表, 每个 dict 为 {类别ID: 原型 Tensor}
        num_classes: 类别数量

    Returns:
        dict: {类别ID: 平均余弦相似度} 每个类别的平均余弦相似度
        dict: {类别ID: 相似度矩阵} 每个类别的完整相似度矩阵
    """
    class_similarities = {}
    class_matrices = {}

    for cls in range(num_classes):
        # 收集所有客户端该类别的原型
        protos = []
        for client_protos in client_protos_list:
            if cls in client_protos:
                protos.append(client_protos[cls])

        if len(protos) < 2:
            # 如果少于2个客户端有该类别的原型，无法计算相似度
            class_similarities[cls] = None
            class_matrices[cls] = None
            continue

        # 堆叠原型并归一化
        protos_tensor = torch.stack(protos)  # [num_clients_with_class, feature_dim]
        protos_norm = torch.nn.functional.normalize(protos_tensor, p=2, dim=1)

        # 计算余弦相似度矩阵
        similarity_matrix = torch.mm(protos_norm, protos_norm.t())  # [n, n]

        # 获取上三角元素（排除对角线），计算平均相似度
        n = similarity_matrix.size(0)
        upper_tri_indices = torch.triu_indices(n, n, offset=1)
        pairwise_similarities = similarity_matrix[
            upper_tri_indices[0], upper_tri_indices[1]
        ]

        avg_similarity = pairwise_similarities.mean().item()
        class_similarities[cls] = avg_similarity
        class_matrices[cls] = similarity_matrix

    return class_similarities, class_matrices


def client_worker(params):
    (
        _,
        device,
        model_state,
        train_set,
        model_name,
        dataset_name,
        lr,
        batch_size,
        epochs,
        num_classes,
        feature_dim,
        mu,
        global_protos,
    ) = params

    model = get_model(model_name, dataset_name, feature_dim).to(device)
    model.load_state_dict(model_state)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    global_protos_tensor = None
    if global_protos is not None:
        first_proto = next(iter(global_protos.values()))
        feat_dim = first_proto.shape[0]
        global_protos_tensor = torch.zeros(num_classes, feat_dim, device=device)
        for label, proto in global_protos.items():
            global_protos_tensor[label] = proto.to(device)

    total_loss = 0.0
    num_batches = 0
    model.train()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, feature, _ = model(x)

            loss_ce = ce_loss(logits, y)

            if global_protos_tensor is not None:
                target_protos = global_protos_tensor[y]
                loss_proto = mse_loss(feature, target_protos)
                loss = loss_ce + mu * loss_proto
            else:
                loss = loss_ce

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

    model.eval()
    proto_sum = torch.zeros(num_classes, feature_dim, device=device)
    proto_count = torch.zeros(num_classes, device=device)

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            _, features, _ = model(x)
            proto_sum.index_add_(0, y, features)
            ones = torch.ones_like(y, dtype=torch.float)
            proto_count.index_add_(0, y, ones)

    local_protos = {}
    present_classes = torch.nonzero(proto_count).squeeze(1)
    for cls_idx in present_classes:
        avg = proto_sum[cls_idx] / proto_count[cls_idx]
        local_protos[cls_idx.item()] = avg.cpu()

    model_state = {k: v.cpu() for k, v in model.state_dict().items()}
    return [avg_loss, model_state, local_protos]


class Server(BaseServer):
    def __init__(self, args: argparse.Namespace):
        super().__init__(True, args)

        self.global_protos = None
        self.proto_similarities: list[dict] = []

    def fit(self):
        num_join_clients = int(self.num_clients * self.args.join_ratio)
        num_join_clients = max(1, num_join_clients)

        for r in range(self.rounds):
            t0 = time.time()
            print(f"\n--- FedTest Round {r + 1}/{self.rounds} ---")

            selected_clients = np.random.choice(
                self.num_clients, num_join_clients, replace=False
            )
            print(f"Selected clients: {selected_clients}")

            global_protos_cpu = None
            if self.global_protos is not None:
                global_protos_cpu = {k: v.cpu() for k, v in self.global_protos.items()}

            p = [
                [
                    i,
                    self.client_gpu[i],
                    self.clients_state[i],
                    self.train_sets[i],
                    self.args.model,
                    self.args.dataset,
                    self.args.lr,
                    self.args.batch_size,
                    self.args.epochs,
                    self.num_class,
                    self.args.feature_dim,
                    self.args.mu_test,
                    global_protos_cpu,
                ]
                for i in selected_clients
            ]
            results = self.run_clients(client_worker, p)

            total_loss = 0.0
            selected_states = []
            current_weights = []
            selected_protos = []
            for i in selected_clients:
                client_loss, client_state, client_proto = results[i]
                total_loss += client_loss
                self.clients_state[i] = client_state
                selected_states.append(client_state)
                selected_protos.append(client_proto)
                current_weights.append(self.weights[i])
            self.loss.append(total_loss / num_join_clients)
            sum_weights = sum(current_weights)
            norm_weights = [w / sum_weights for w in current_weights]

            self.aggregate(selected_states, weights=norm_weights)
            self.global_protos = proto_cluster(selected_protos)

            # 计算客户端间原型相似度
            class_similarities, _ = compute_prototype_similarity(
                selected_protos, self.num_class
            )
            self.proto_similarities.append(class_similarities)

            # 打印相似度信息
            valid_sims = [v for v in class_similarities.values() if v is not None]
            if valid_sims:
                avg_sim = np.mean(valid_sims)
                print(f"Prototype Cosine Similarity (avg over classes): {avg_sim:.4f}")
                sim_str = ", ".join(
                    f"C{c}:{v:.3f}"
                    for c, v in sorted(class_similarities.items())
                    if v is not None
                )
                print(f"  Per-class: {sim_str}")

            self.evaluate(protos=self.global_protos)

            print(
                f"Personalized Accuracy: {self.acc[-1]:.2f}%, Proto Accuracy: {self.acc_proto[-1]:.2f}%, "
                f"Avg Loss: {self.loss[-1]:.4f}"
            )
            print(f"Round finished in {time.time() - t0:.2f} seconds")

    def save(self):
        f = {
            "acc": {"model": self.acc, "proto": self.acc_proto},
            "loss": self.loss,
            "state_dict": {
                "client": self.clients_state,
                "proto": self.global_protos,
            },
            "aux": self.proto_similarities,
        }
        self.deal_save(f)
