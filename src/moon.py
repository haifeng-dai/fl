import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy

from src.utils.fed_utils import BaseClient, BaseServer, ClientInfo
from src.utils.parallel import run_parallel_clients


def add_args(parser):
    group = parser.add_argument_group("MOON Specific Arguments")
    group.add_argument("--mu", type=float, default=1.0)
    group.add_argument("--tau", type=float, default=0.5)
    return parser


class MOONClient(BaseClient):
    def __init__(self, *args, mu=1.0, tau=0.5, weight_decay=1e-4, **kwargs):
        super().__init__(*args, **kwargs)
        self.mu = mu
        self.tau = tau
        self.weight_decay = weight_decay
        # MOON 需要两个额外的辅助模型
        self.global_model = copy.deepcopy(self.model).to(self.device)
        self.prev_model = copy.deepcopy(self.model).to(self.device)

    def moon_loss(self, z, z_glob, z_prev):
        pos_sim = F.cosine_similarity(z, z_glob, dim=-1)
        neg_sim = F.cosine_similarity(z, z_prev, dim=-1)
        logits = torch.cat([pos_sim.reshape(-1, 1), neg_sim.reshape(-1, 1)], dim=1)
        logits /= self.tau
        labels = torch.zeros(z.size(0)).to(z.device).long()
        return F.cross_entropy(logits, labels)

    def train(self):
        self.model.train()
        self.global_model.eval()
        self.prev_model.eval()
        
        optimizer = optim.SGD(
            self.model.parameters(), 
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        loss_list = []
        
        for epoch in range(self.epochs):
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                
                y, z = self.model(data)
                with torch.no_grad():
                    _, z_glob = self.global_model(data)
                    _, z_prev = self.prev_model(data)

                loss_ce = self.ce(y, target)
                loss_con = self.moon_loss(z, z_glob, z_prev)
                loss = loss_ce + self.mu * loss_con
                
                loss.backward()
                optimizer.step()
                loss_list.append(loss.item())
                
        model_state = {k: v.detach().clone().cpu() for k, v in self.model.state_dict().items()}
        return sum(loss_list) / len(loss_list), model_state

    def set_client(self, parameters):
        global_params, prev_local_params = parameters
        # 加载全局参数到本地模型和全局模型副本
        self.model.load_state_dict(global_params)
        self.global_model.load_state_dict(global_params)
        
        # 加载上轮本地参数
        if prev_local_params is not None:
            self.prev_model.load_state_dict(prev_local_params)
        else:
            self.prev_model.load_state_dict(global_params)

class MOONServer(BaseServer):
    def __init__(self, model, train_loader, test_loader, clients_info, rounds):
        super().__init__(model, test_loader, clients_info, rounds)
        self.args = clients_info.args
        self.clients = {
            i: MOONClient(
                client_id=i,
                model=model,
                train_loader=train_loader[i],
                lr=clients_info.lr,
                epochs=clients_info.epochs,
                device=clients_info.cuda[i],
                mu=self.args.mu if hasattr(self.args, 'mu') else 1.0,
                tau=self.args.tau if hasattr(self.args, 'tau') else 0.5,
                weight_decay=self.args.weight_decay if hasattr(self.args, 'weight_decay') else 1e-4
            ) for i in range(len(clients_info.cuda))
        }

    def fit(self):
        prev_local_params_list = [None] * len(self.clients)
        
        for r in range(self.rounds):
            print(f"\n--- MOON Round {r + 1}/{self.rounds} ---")
            global_params = {k: v.cpu() for k, v in self.model.state_dict().items()}

            # 准备每个客户端的个性化参数包
            parameters_per_client = {
                i: (global_params, prev_local_params_list[i]) 
                for i in self.clients.keys()
            }

            results = run_parallel_clients(
                clients=self.clients,
                parameters=parameters_per_client,
                gpu_pools=self.gpu_pools
            )
            
            loss_epoch = [res[0] for res in results]
            client_dicts = [res[1] for res in results]

            prev_local_params_list = client_dicts
            self.aggregate(client_dicts)
            acc = self.evaluate()
            print(f"Global Accuracy: {acc:.2f}%, Avg Loss: {sum(loss_epoch)/len(loss_epoch):.4f}")