from .fed_utils import BaseServer, get_model
from .parallel import run_parallel_clients
from .aggregate import param_aggregate
from .evaluate import evaluate_model, evaluate_prototype
import torch, os


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
        tensor1 = params1[key]
        tensor2 = params2[key]

        # 检查形状是否相同
        if tensor1.shape != tensor2.shape:
            return False

        # 检查数值是否完全相同
        # 如果设备不同，移到 CPU 比较；如果相同，直接在原设备比较（更快）
        if tensor1.device != tensor2.device:
            if not torch.equal(tensor1.cpu(), tensor2.cpu()):
                return False
        else:
            if not torch.equal(tensor1, tensor2):
                return False

    return True


def ce_loss(predictions, targets):
    """
    计算交叉熵损失

    Args:
        predictions: 预测值，形状为(batch_size, num_classes)
        targets: 真实标签，形状为(batch_size,)

    Returns:
        float: 交叉熵损失值
    """
    return torch.nn.functional.cross_entropy(predictions, targets)


def mse_loss(predictions, targets):
    """
    计算均方误差损失

    Args:
        predictions: 预测值，形状为(batch_size, num_classes)
        targets: 真实标签，形状为(batch_size,)

    Returns:
        float: 均方误差损失值
    """
    return torch.nn.functional.mse_loss(predictions, targets)


def kl_loss(student_logits, teacher_logits, temperature=1.0):
    """
    计算知识蒸馏的 KL 散度损失

    Args:
        student_logits: 学生模型的输出 logits
        teacher_logits: 教师模型的输出 logits (已 detach)
        temperature: 软化分布的温度参数 (默认: 1.0)

    Returns:
        KL 散度损失值
    """
    if temperature != 1.0:
        student_logits = student_logits / temperature
        teacher_logits = teacher_logits / temperature

    student_soft = torch.nn.functional.log_softmax(student_logits, dim=1)
    teacher_soft = torch.nn.functional.softmax(teacher_logits, dim=1)

    loss = torch.nn.functional.kl_div(
        student_soft, teacher_soft, reduction="batchmean"
    ) * (temperature**2)

    return loss


def get_pre_name(args):
    fold_path = os.path.join(
        f"{args.algo}",
        f"{args.dataset}_{args.partition}_{args.num_clients}",
    )
    if args.partition == "dirichlet":
        fold_path += f"_{args.alpha}"
    elif args.partition == "pathological":
        fold_path += f"_{args.n_class}"
    args.save_path = os.path.join("results", fold_path)
    args.log_path = os.path.join("logs", fold_path)
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.log_path, exist_ok=True)
    args.name_pre = f"{args.epochs}_{args.batch_size}_{args.lr}"
