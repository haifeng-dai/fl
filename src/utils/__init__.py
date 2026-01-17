from .fed_utils import BaseClient, BaseServer
from .parallel import run_parallel_clients
from .aggregate import param_aggregate
from .evaluate import evaluate_model, evaluate_prototype
import torch


def compare_model_parameters(params1: dict, params2: dict) -> bool:
    """
    对比两个结构相同的模型参数字典是否完全相同

    Args:
        params1: 第一个模型参数字典
        params2: 第二个模型参数字典

    Returns:
        bool: 如果所有参数都相同则返回True，否则返回False
    """
    # 检查键是否相同
    if set(params1.keys()) != set(params2.keys()):
        return False

    # 逐个对比每个参数张量
    for key in params1.keys():
        tensor1 = params1[key].cpu()
        tensor2 = params2[key].cpu()

        # 检查形状是否相同
        if tensor1.shape != tensor2.shape:
            return False

        # 检查数值是否完全相同
        if not torch.equal(tensor1, tensor2):
            return False

    return True
