import datetime
import importlib
import json
import os
import random
import sys
import time
import traceback

import numpy as np
import torch

from src import (
    get_config,
    get_pre_name,
    init_ray,
    prepare_data,
    setup_runtime_env,
    shutdown_ray,
)


def set_seed(seed):
    """设置所有随机种子以确保实验可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def run_experiment(args, algo_module, t, total_times, base_seed):
    """
    运行单次实验的核心逻辑
    """
    # 更新当前运行的索引和随机种子
    args.times = t
    args.seed = base_seed + t
    start_time_stamp = time.time()

    set_seed(args.seed)

    # 获取日志路径
    log_path = algo_module.get_path(args)

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log_f = None

    if not args.test:
        # 打开日志文件并将 stdout/stderr 重定向
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        log_f = open(log_path, "w", encoding="utf-8", buffering=1)
        sys.stdout = log_f
        sys.stderr = log_f

    try:
        # 记录本次实验的完整配置到日志中
        print(f"\n{'=' * 30} Experiment Config {t + 1}/{total_times} {'=' * 30}")
        args_dict = vars(args)
        print(f"Full Configuration:\n{json.dumps(args_dict, indent=4, ensure_ascii=False)}")
        print(f"{'-' * 80}\n")
        
        print(f"Start Seed: {args.seed}")
        print(
            f"Start time: {datetime.datetime.fromtimestamp(start_time_stamp).strftime('%Y-%m-%d %H:%M:%S')}\n"
        )

        prepare_data(
            dataset_name=args.dataset,
            partition_method=args.partition,
            num_clients=args.num_clients,
            alpha=args.alpha,
            n_classes=args.n_class,
            test_ratio=args.test_ratio,
        )

        server = algo_module.Server(args=args)
        server.fit()
        server.save()

        end_time_stamp = time.time()
        print(
            f"End time: {datetime.datetime.fromtimestamp(end_time_stamp).strftime('%Y-%m-%d %H:%M:%S')}"
        )
        delta = datetime.timedelta(seconds=int(end_time_stamp - start_time_stamp))
        print(f"\nTotal time: {delta}")
    except Exception:
        traceback.print_exc()
        raise
    finally:
        # 恢复 stdout/stderr 并关闭日志文件
        if not args.test:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            if log_f:
                log_f.close()


def main():
    # 1. 环境与配置初始化
    setup_runtime_env()
    configs = get_config()
    total_configs = len(configs)

    # 2. 初始化全局 Ray 资源（以第一个配置的 GPU 设定为准）
    init_ray(configs[0])

    try:
        for cfg_idx, args in enumerate(configs):
            # 1. 实验分割线（主控制台可见）
            print(f"\n{'=' * 30} Running Experiment {cfg_idx + 1}/{total_configs} {'=' * 30}")

            # 2. 动态加载算法模块
            try:
                algo_module = importlib.import_module(f"src.{args.algo}")
            except ModuleNotFoundError:
                raise ValueError(f"Algorithm module src.{args.algo} not found.")

            # 3. 自动根据数据集设置 n_class (当 n_class 为 0 时)
            if args.n_class == 0:
                if args.dataset == "cifar100":
                    args.n_class = 10
                elif args.dataset == "tiny_imagenet":
                    args.n_class = 20
                else:
                    args.n_class = 2

            # 4. 生成实验路径
            get_pre_name(args)

            # 5. 执行多次实验 (times)
            total_times = args.times
            base_seed = args.seed
            for t in range(total_times):
                run_experiment(args, algo_module, t, total_times, base_seed)
    finally:
        shutdown_ray()


if __name__ == "__main__":
    main()
