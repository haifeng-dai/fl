import logging
import os
import random

import numpy as np
import ray
import torch


def setup_runtime_env():
    """
    早期环境配置，解决 Ray 与 uv 配合时的库路径问题。
    必须在 main.py 启动初期调用。
    """
    # 1. 禁止 Ray 自动检测并修改虚拟环境路径（解决 libcublas 找不到的问题）
    os.environ["RAY_RUNTIME_ENV_DETECT_VENV"] = "0"
    # 2. 优化 Ray 日志显示
    os.environ["RAY_DEDUP_LOGS"] = "0"
    # 3. 解决特定驱动下的显存初始化冲突，消除 Ray 的 FutureWarning
    os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
    # 4. 彻底禁止 Ray 尝试在 Worker 端重建 uv 环境
    os.environ["RAY_RUNTIME_ENV_IGNORE_PYPROJECT"] = "1"


def init_ray(args):
    """
    初始化仅使用 NVIDIA GPU 的 Ray 分布式环境。
    """
    gpus_str = str(args.gpus).strip()
    if not gpus_str:
        raise RuntimeError("必须通过 YAML 配置项 gpus 指定至少一张 NVIDIA GPU")

    try:
        gpu_ids = [int(gpu_id) for gpu_id in gpus_str.split(",")]
    except ValueError as exc:
        raise ValueError(f"gpus 配置格式错误: {gpus_str}") from exc
    if any(gpu_id < 0 for gpu_id in gpu_ids) or len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError(f"gpus 必须是互不重复的非负整数: {gpus_str}")
    num_gpus = len(gpu_ids)

    # 1. 在初始化前设置全局可见设备，让 Ray 仅管理这些卡
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus_str
    if not torch.cuda.is_available():
        raise RuntimeError("未检测到可用的 NVIDIA CUDA GPU；本项目不支持 CPU 训练模式")
    visible_gpus = torch.cuda.device_count()
    if visible_gpus < num_gpus:
        raise RuntimeError(
            f"配置了 {num_gpus} 张 GPU，但当前仅检测到 {visible_gpus} 张可见 GPU"
        )

    print(f"-> Initializing Ray Framework | GPUs: {num_gpus} ({gpus_str})")

    runtime_env = {
        "working_dir": ".",
        "excludes": [
            ".venv",
            ".git",
            "__pycache__",
            "results",
            "logs",
            "datasets",
            "*.pth",
            "*.pt",
            "pyproject.toml",
            "uv.lock",
        ],
        "env_vars": {
            "RAY_RUNTIME_ENV_IGNORE_PYPROJECT": "1",
        },
    }

    ray.init(
        num_gpus=num_gpus,
        num_cpus=max(num_gpus, 1) * args.max_workers_per_gpu + 4,
        ignore_reinit_error=True,
        logging_level=logging.ERROR,
        runtime_env=runtime_env,
    )


def shutdown_ray():
    """安全关闭 Ray。"""
    if ray.is_initialized():
        ray.shutdown()


def set_seed(seed):
    """设置所有随机种子以确保实验可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
