import torch

from src.models import CNN, HARCNN, HARMLP, ResNet18, ResNet50

from .protocol import ModelParam


def build_model(param: ModelParam) -> torch.nn.Module:
    rgb = {
        "tiny_imagenet",
        "flowers102",
        "cars",
        "gtsrb",
        "cinic10",
        "svhn",
        "pacs",
        "officehome",
        "vlcs",
        "domainnet",
    }
    if param.model == "cnn":
        return CNN(
            3 if "cifar" in param.dataset or param.dataset in rgb else 1,
            param.num_class,
            param.feature_dim,
            param.dataset,
        )
    if param.model == "resnet18":
        return ResNet18(param.num_class, param.feature_dim, param.dataset)
    if param.model == "resnet50":
        return ResNet50(param.num_class, param.feature_dim, param.dataset)
    if param.model == "harcnn":
        return HARCNN(9, param.num_class, param.feature_dim)
    if param.model == "harmlp":
        return HARMLP(561, param.num_class, param.feature_dim)
    raise ValueError(f"Unknown model: {param.model}")
