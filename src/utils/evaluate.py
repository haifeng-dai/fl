import copy
import torch


def evaluate_model(model, test_loader, device):
    eval_model = copy.deepcopy(model)
    eval_model.to(device)
    eval_model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, _ = eval_model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    accuracy = 100.0 * correct / len(test_loader.dataset)
    return accuracy