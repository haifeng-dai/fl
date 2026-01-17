import copy
import torch


def evaluate_model(model, test_set, device) -> float:
    loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False)

    eval_model = copy.deepcopy(model)
    eval_model.to(device)
    eval_model.eval()
    correct = 0.
    count = 0.
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output, _ = eval_model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            count += target.size(0)

    return 100.0 * correct / count


def evaluate_prototype(model, prototypes, test_set, device) -> float:
    loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False)

    eval_model = copy.deepcopy(model)
    eval_model.to(device)
    eval_model.eval()
    correct = 0.
    count = 0.
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            _, features = eval_model(data)

            dist = torch.cdist(features, prototypes, p=2)
            pred = dist.argmin(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            count += target.size(0)

    return 100.0 * correct / count