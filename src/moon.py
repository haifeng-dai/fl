import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.models import SimpleCNN
from src.utils.fed_utils import BaseClient, BaseServer, load_client_data
from src.utils.parallel import run_parallel_clients


def add_args(parser):
    """
    Add MOON specific arguments to the parser.
    """
    group = parser.add_argument_group("MOON Specific Arguments")
    group.add_argument(
        "--mu", type=float, default=1.0, help="Weight for MOON contrastive loss"
    )
    group.add_argument(
        "--tau", type=float, default=0.5, help="Temperature for contrastive loss"
    )
    return parser


class MOONClient(BaseClient):
    def moon_loss(self, z, z_glob, z_prev, temperature=0.5):
        pos_sim = F.cosine_similarity(z, z_glob, dim=-1)
        neg_sim = F.cosine_similarity(z, z_prev, dim=-1)
        logits = torch.cat([pos_sim.reshape(-1, 1), neg_sim.reshape(-1, 1)], dim=1)
        logits /= temperature
        labels = torch.zeros(z.size(0)).to(z.device).long()
        return F.cross_entropy(logits, labels)

    def train(self, global_params):
        dataloader = load_client_data(self.client_id, self.args.dataset, self.args.partition)
        model = SimpleCNN().to(self.device)
        model.load_state_dict(global_params)

        global_model = SimpleCNN().to(self.device)
        global_model.load_state_dict(global_params)
        global_model.eval()

        prev_model = SimpleCNN().to(self.device)
        # MOON specifically needs the previous round's local model parameters
        if self.args.prev_local_params[self.client_id] is not None:
            prev_model.load_state_dict(self.args.prev_local_params[self.client_id])
        else:
            prev_model.load_state_dict(global_params)
        prev_model.eval()

        optimizer = optim.SGD(model.parameters(), lr=self.args.lr)
        criterion = nn.CrossEntropyLoss()

        model.train()
        for epoch in range(self.args.epochs):
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                y, z = model(data)
                with torch.no_grad():
                    _, z_glob = global_model(data)
                    _, z_prev = prev_model(data)

                loss_ce = criterion(y, target)
                loss_con = self.moon_loss(z, z_glob, z_prev, temperature=self.args.tau)
                loss = loss_ce + self.args.mu * loss_con
                loss.backward()
                optimizer.step()
        return model.cpu().state_dict()


class MOONServer(BaseServer):
    def fit(self):
        eval_device = self.clients_info[0][1]
        # In MOON, we need to pass previous local params to clients
        self.args.prev_local_params = [None] * self.args.num_clients

        for r in range(self.args.rounds):
            print(f"\n--- MOON Round {r + 1}/{self.args.rounds} ---")
            global_params = self.model.state_dict()

            client_dicts = run_parallel_clients(
                MOONClient, self.clients_info, global_params, self.args
            )

            self.args.prev_local_params = client_dicts
            self.aggregate(client_dicts)
            acc = self.evaluate(eval_device)
            print(f"Global Accuracy: {acc:.2f}%")