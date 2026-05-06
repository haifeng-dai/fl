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
    初始化 Ray 分布式环境，采用 fl_ray 风格的鲁棒性配置。
    """
    gpus_str = str(args.gpus).strip()
    gpu_ids = [int(i) for i in gpus_str.split(",")] if gpus_str else []
    num_gpus = len(gpu_ids)

    resource_msg = f"GPUs: {num_gpus} ({gpus_str})" if num_gpus > 0 else "CPU Only"
    print(f"-> Initializing Ray Framework (Mandatory) | {resource_msg}")

    # 1. 在初始化前设置全局可见设备，让 Ray 仅管理这些卡
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus_str

    runtime_env = {
        "working_dir": ".",
        "excludes": [
            ".venv",
            ".git",
            "__pycache__",
            "results",
            "logs",
            "results_ray",
            "logs_ray",
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
