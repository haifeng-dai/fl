import copy
import argparse
import torch

from .utils import (
    BaseServer,
    run_parallel_clients,
    ce_loss,
    evaluate_model,
    get_model,
    param_aggregate,
    kl_loss,
)


def add_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("ProxyFL Specific Arguments")
    group.add_argument(
        "--mu",
        type=float,
        default=1.0,
        help="Weight for Mutual Learning Distillation",
    )
    group.add_argument(
        "--adj_type",
        type=str,
        default="ring",
        choices=["ring", "centralized"],
        help="Topology of the decentralized network",
    )
    return parser


def client_worker(params):
    device = params[0]
    proxy_state = params[1]
    local_state = params[2]
    train_set = params[3]

    model_name = params[4]
    dataset_name = params[5]
    lr = params[6]
    batch_size = params[7]
    epochs = params[8]
    mu = params[9]

    # 1. Initialize Proxy Model (Shared)
    proxy_model = get_model(model_name, dataset_name).to(device)
    proxy_model.load_state_dict(proxy_state)

    # 2. Initialize Local Model (Private)
    local_model = get_model(model_name, dataset_name).to(device)
    local_model.load_state_dict(local_state)

    # Optimizers
    # Usually ProxyFL allows different LRs, but we use the same for simplicity unless specified
    opt_p = torch.optim.SGD(proxy_model.parameters(), lr=lr)
    opt_l = torch.optim.SGD(local_model.parameters(), lr=lr)

    loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    proxy_model.train()
    local_model.train()

    total_loss_p = 0.0
    total_loss_l = 0.0
    num_batches = 0

    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            # Forward pass
            out_p, _ = proxy_model(x)
            out_l, _ = local_model(x)

            # Cross Entropy
            ce_p = ce_loss(out_p, y)
            ce_l = ce_loss(out_l, y)

            # Mutual Distillation
            # KL(Local || Proxy) -> Proxy learns from Local
            loss_kl_p = kl_loss(out_p, out_l.detach())

            # KL(Proxy || Local) -> Local learns from Proxy
            loss_kl_l = kl_loss(out_l, out_p.detach())

            loss_p = ce_p + mu * loss_kl_p
            loss_l = ce_l + mu * loss_kl_l

            # Update Proxy
            opt_p.zero_grad()
            loss_p.backward()
            opt_p.step()

            # Update Local
            opt_l.zero_grad()
            loss_l.backward()
            opt_l.step()

            total_loss_p += loss_p.item()
            total_loss_l += loss_l.item()
            num_batches += 1

    avg_loss_p = total_loss_p / num_batches
    avg_loss_l = total_loss_l / num_batches

    # Return: client_id, [avg_loss_l, new_proxy_state, new_local_state]
    # We track local loss usually, but server aggregates proxy
    return [avg_loss_p, avg_loss_l, proxy_model.state_dict(), local_model.state_dict()]


class Server(BaseServer):
    def __init__(self, args: argparse.Namespace):
        model = get_model(args.model, args.dataset)
        # ProxyFL is a personalized method
        super().__init__(model, True, args)

        # Initialize local models for each client (Private)
        self.client_model_states = [
            copy.deepcopy(self.model.state_dict()) for _ in range(self.num_clients)
        ]
        # Initialize proxy models for each client (Public/Shared)
        self.proxy_model_states = [
            copy.deepcopy(self.model.state_dict()) for _ in range(self.num_clients)
        ]
        
        self.loss_p = []
        self.adj_matrix = self.generate_adj_matrix()

    def generate_adj_matrix(self):
        num_clients = self.num_clients
        adj = torch.zeros(num_clients, num_clients)
        if self.args.adj_type == "ring":
            for i in range(num_clients):
                adj[i, i] = 1.0
                adj[i, (i - 1) % num_clients] = 1.0
                adj[i, (i + 1) % num_clients] = 1.0
        elif self.args.adj_type == "centralized":
            adj.fill_(1.0)
        
        # Normalize weights for each client
        row_sums = adj.sum(dim=1, keepdim=True)
        adj = adj / row_sums
        return adj

    def fit(self):
        for r in range(self.rounds):
            print(f"\n--- ProxyFL Round {r + 1}/{self.rounds} ---")

            # 1. Neighbor Aggregation for each client
            # Each client aggregates models from its neighbors based on the adjacency matrix
            parameters_per_client = []
            for i in range(self.num_clients):
                # Identify neighbors and their weights
                neighbor_indices = torch.where(self.adj_matrix[i] > 0)[0].tolist()
                neighbor_weights = self.adj_matrix[i, neighbor_indices].tolist()
                
                # Perform local aggregation
                neighbor_states = [self.proxy_model_states[j] for j in neighbor_indices]
                aggregated_proxy_state = param_aggregate(neighbor_states, neighbor_weights)

                p = [
                    self.client_gpu[i],
                    aggregated_proxy_state,
                    self.client_model_states[i],
                    self.train_sets[i],
                    self.args.model,
                    self.args.dataset,
                    self.args.lr,
                    self.args.batch_size,
                    self.args.epochs,
                    self.args.mu,
                ]
                parameters_per_client.append(p)

            # 2. Parallel Client Training
            results = run_parallel_clients(
                client_worker=client_worker,
                num_clients=self.num_clients,
                parameters=parameters_per_client,
                gpu_pools=self.gpu_pools,
                mp=self.mp,
            )

            # 3. Update states and calculate average losses
            total_loss_p = 0.0
            total_loss_l = 0.0

            for i, res in enumerate(results):
                loss_p = res[0]
                loss_l = res[1]
                new_proxy_state = res[2]
                new_local_state = res[3]

                total_loss_p += loss_p
                total_loss_l += loss_l

                # Update stored states (move to CPU to save GPU memory)
                self.proxy_model_states[i] = {
                    k: v.cpu() for k, v in new_proxy_state.items()
                }
                self.client_model_states[i] = {
                    k: v.cpu() for k, v in new_local_state.items()
                }

            self.loss.append(total_loss_l / self.num_clients)
            self.loss_p.append(total_loss_p / self.num_clients)

            # 4. Evaluation (using local personalized models)
            self.evaluate()
            print(
                f"Avg Local Accuracy: {self.acc[-1]:.2f}%, Avg Local Loss: {self.loss[-1]:.4f}"
            )

    def evaluate(self):
        # Evaluate Personalized Local Models on Local Test Sets
        accs = []
        for i in range(self.num_clients):
            self.model.load_state_dict(self.client_model_states[i])
            acc = evaluate_model(self.model, self.test_set[i], self.device)
            accs.append(acc)

        avg_acc = sum(accs) / len(accs)
        self.acc.append(avg_acc)

    def save(self, test):
        file_name: str = f"{self.args.epochs}_{self.args.batch_size}_{self.args.lr}.pt"
        f = {"acc": self.acc, "loss": self.loss, "state_dict": self.model.state_dict()}
        super().deal_save(test, f, file_name)
