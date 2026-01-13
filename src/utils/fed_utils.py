import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os

class BaseClient:
    def __init__(self, client_id, device, args):
        self.client_id = client_id
        self.device = device
        self.args = args

    def train(self, global_params):
        raise NotImplementedError

class BaseServer:
    def __init__(self, model, test_loader, clients_info, args):
        self.model = model
        self.test_loader = test_loader
        self.clients_info = clients_info
        self.args = args

    def aggregate(self, client_state_dicts):
        weights = [1.0 / len(client_state_dicts)] * len(client_state_dicts)
        global_dict = self.model.state_dict()
        for key in global_dict.keys():
            if global_dict[key].dtype == torch.float32:
                temp = torch.zeros_like(global_dict[key])
                for i, state_dict in enumerate(client_state_dicts):
                    temp += state_dict[key] * weights[i]
                global_dict[key].copy_(temp)
        self.model.load_state_dict(global_dict)

    def evaluate(self, device):
        self.model.to(device)
        self.model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(device), target.to(device)
                output, _ = self.model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        return 100. * correct / len(self.test_loader.dataset)

    def fit(self):
        raise NotImplementedError

def load_client_data(client_id, dataset_name, partition):
    data_path = f'./datasets/{dataset_name}/{partition}/client_{client_id}.pt'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data for client {client_id} not found at {data_path}.")
    data = torch.load(data_path)
    dataset = TensorDataset(data['x'], data['y'])
    return DataLoader(dataset, batch_size=64, shuffle=True)

def load_test_data(dataset_name, partition):
    test_path = f'./datasets/{dataset_name}/{partition}/test_data.pt'
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test data not found at {test_path}.")
    data = torch.load(test_path)
    dataset = TensorDataset(data['x'], data['y'])
    return DataLoader(dataset, batch_size=1000, shuffle=False)