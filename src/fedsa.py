import argparse, os
import time

import numpy as np
import torch
import torch.nn.functional as F

from .utils import (
    BaseServer,
    ce_loss,
    get_model,
    mse_loss,
    param_aggregate,
    run_parallel_clients,
)


def add_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("FedSA Specific Arguments")
    group.add_argument(
        "--alpha_sa",
        type=float,
        default=0.5,
        help="Momentum factor for updating semantic anchors",
    )
    group.add_argument(
        "--lambda_r",
        type=float,
        default=0.1,
        help="Weight for regularization loss",
    )
    group.add_argument(
        "--lambda_mcl",
        type=float,
        default=0.1,
        help="Weight for margin-enhanced contrastive loss",
    )
    group.add_argument(
        "--lambda_cc",
        type=float,
        default=0.1,
        help="Weight for classifier calibration loss",
    )
    return parser


def get_path(args):
    args.file_name = f"{args.name_pre}_{args.alpha_sa}_{args.lambda_r}_{args.lambda_mcl}_{args.lambda_cc}"
    return os.path.join(args.log_path, f"{args.file_name}.log")


def mcl_loss(
    feature: torch.Tensor,
    protos: torch.Tensor,
    num_classes: int,
    y: torch.Tensor,
    d: float,
):
    one_hot = F.one_hot(y, num_classes)
    dist_final = torch.cdist(feature, protos) + one_hot * d
    return ce_loss(-dist_final, y)


def margin(anchor: torch.Tensor) -> float:
    d = torch.tensor(0.0, device=anchor.device, dtype=anchor.dtype)
    # anchor shape: [num_classes, feature_dim]
    for anchor_i in anchor:
        for anchor_j in anchor:
            d += torch.norm(anchor_i - anchor_j, p=2)

    denom = (anchor.shape[0] - 1) ** 2
    if denom > 0:
        d /= denom
    return d.item()


def client_worker(params):
    """
    FedSA local training with Semantic Anchors and multiple regularizations.
    """
    (
        _,
        device,
        model_state,
        local_anchors,
        global_anchors,
        train_set,
        model_name,
        dataset_name,
        lr,
        batch_size,
        epochs,
        lambda_r,
        lambda_mcl,
        lambda_cc,
        num_classes,
        feature_dim,
    ) = params

    # 1. Initialize Model
    model = get_model(model_name, dataset_name, feature_dim).to(device)
    model.load_state_dict(model_state)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    model.train()
    total_loss = 0.0
    num_batches = 0

    local_anchors = local_anchors.to(device)
    global_anchors = global_anchors.to(device)
    # Calculate margin 'd' for MCL loss
    d = max(margin(global_anchors), margin(local_anchors))

    # 2. Training Loop
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, features = model(x)
            # Classifier output for global anchors
            output = model.classifier(global_anchors)

            # Standard Cross Entropy
            loss_ce = ce_loss(logits, y)

            # Regression Loss (L_r): Align features with global anchors
            loss_r = mse_loss(features, global_anchors[y])

            # Margin-enhanced Contrastive Loss (L_mcl)
            loss_mcl = mcl_loss(features, global_anchors, num_classes, y, d)

            # Classifier Calibration Loss (L_cc)
            loss_cc = ce_loss(output, torch.arange(num_classes, device=device))

            loss = (
                loss_ce
                + lambda_r * loss_r
                + lambda_mcl * loss_mcl
                + lambda_cc * loss_cc
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches

    # 3. Calculate new local anchors (average features per class)
    with torch.no_grad():
        anchor_sums = {}
        anchor_counts = {}

        for x, y in loader:
            x, y = x.to(device), y.to(device)
            _, features = model(x)

            unique_labels = torch.unique(y)
            for label in unique_labels:
                label_item = label.item()
                mask = y == label
                feat_sum = features[mask].sum(dim=0)
                count = mask.sum().item()

                if label_item not in anchor_sums:
                    anchor_sums[label_item] = feat_sum
                    anchor_counts[label_item] = count
                else:
                    anchor_sums[label_item] += feat_sum
                    anchor_counts[label_item] += count

        local_anchors_dict = {}
        for label, total in anchor_sums.items():
            local_anchors_dict[label] = total.cpu() / anchor_counts[label]

    model_state = {k: v.cpu() for k, v in model.state_dict().items()}
    return [avg_loss, model_state, local_anchors_dict]


class Server(BaseServer):
    def __init__(self, args: argparse.Namespace):
        super().__init__(False, args)

        # 全局语义锚点 (Prototypes)
        self.anchors = torch.zeros(self.num_class, self.args.feature_dim)
        self.clients_anchors = [
            self.anchors.data.clone() for _ in range(self.num_clients)
        ]
        self.clients_state = [self.model.state_dict() for _ in range(self.num_clients)]

    def fit(self):
        num_join_clients = int(self.num_clients * self.args.join_ratio)
        num_join_clients = max(1, num_join_clients)

        print(f"FedSA Training with alpha_sa={self.args.alpha_sa}")

        for r in range(self.rounds):
            t0 = time.time()
            print(f"\n--- FedSA Round {r + 1}/{self.rounds} ---")

            selected_clients = np.random.choice(
                self.num_clients, num_join_clients, replace=False
            )
            print(f"Selected clients: {selected_clients}")

            p = [
                [
                    i,
                    self.client_gpu[i],
                    self.clients_state[i],
                    self.clients_anchors[i],
                    self.anchors,
                    self.train_sets[i],
                    self.args.model,
                    self.args.dataset,
                    self.args.lr,
                    self.args.batch_size,
                    self.args.epochs,
                    self.args.lambda_r,
                    self.args.lambda_mcl,
                    self.args.lambda_cc,
                    self.num_class,
                    self.args.feature_dim,
                ]
                for i in selected_clients
            ]

            results = run_parallel_clients(
                client_worker=client_worker,
                parameters=p,
                gpu_pools=self.gpu_pools,
                mp=self.mp,
            )

            total_loss = 0.0
            selected_states = []
            current_weights = []
            local_anchors = []
            for i in selected_clients:
                total_loss += results[i][0]
                self.clients_state[i] = results[i][1]
                selected_states.append(results[i][1])
                current_weights.append(self.weights[i])
                local_anchors.append(results[i][2])
            self.loss.append(total_loss / num_join_clients)
            sum_weights = sum(current_weights)
            norm_weights = [w / sum_weights for w in current_weights]

            # 1. 聚合全局模型
            self.model.load_state_dict(param_aggregate(selected_states, norm_weights))

            # 2. 更新全局语义锚点
            self.update_global_anchors(local_anchors)

            # 3. 更新 Server 端保存的 Client Anchors
            # for i in selected_clients:
            #     self.clients_anchors[i] = local_anchors[i].data.clone()
            for i, anchors_dict in enumerate(local_anchors):
                client_idx = selected_clients[i]
                for label, anchor in anchors_dict.items():
                    # Update local copy on server device
                    self.clients_anchors[client_idx][label] = anchor.data.clone()

            self.evaluate()
            print(
                f"Global Accuracy: {self.acc[-1]:.2f}%, Avg Loss: {self.loss[-1]:.4f}"
            )
            print(f"Round finished in {time.time() - t0:.2f} seconds")

    def update_global_anchors(self, client_anchors_list):
        new_anchors = {}
        counts = {}

        for client_anchors in client_anchors_list:
            for label, anchor in client_anchors.items():
                anchor = anchor.to(self.device)
                if label not in new_anchors:
                    new_anchors[label] = anchor
                    counts[label] = 1
                else:
                    new_anchors[label] += anchor
                    counts[label] += 1

        for label in new_anchors:
            new_anchors[label] /= counts[label]

        alpha = self.args.alpha_sa
        for label, new_anchor in new_anchors.items():
            self.anchors[label] = (1 - alpha) * self.anchors[
                label
            ] + alpha * new_anchor.cpu()

    def save(self):
        f = {
            "acc": self.acc,
            "loss": self.loss,
            "state_dict": {
                "anchors": self.anchors.data,
                "clients": self.clients_state,
            },
        }
        super().deal_save(f)
