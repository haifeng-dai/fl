import argparse
import itertools
import os
from types import SimpleNamespace

import yaml


def load_yaml(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_config():
    """
    核心配置加载逻辑：
    1. 加载 default.yaml
    2. 解析命令行 -a 参数，加载 algorithms.yaml 对应区块
    3. 解析命令行其他参数进行覆盖
    4. 展开 sweep 列表
    """
    # 1. 基础解析器：仅获取算法名和核心控制参数
    base_parser = argparse.ArgumentParser(
        description="Federated Learning Framework", add_help=False
    )
    base_parser.add_argument(
        "-a", "--algo", type=str, default=None, help="Algorithm name"
    )
    base_parser.add_argument(
        "-t", "--test", action="store_true", help="Enable test mode"
    )

    # 解析命令行参数
    args = base_parser.parse_args()

    # 2. 加载 YAML 基础配置
    config_dict = load_yaml("configs/default.yaml")

    # 3. 如果指定了算法，加载算法特定配置
    target_algos = args.algo

    # 支持逗号分隔的多个算法 (如 -a fedavg,fedprox)
    if "," in target_algos:
        target_algos = [a.strip() for a in target_algos.split(",")]
        config_dict["algo"] = target_algos
    else:
        config_dict["algo"] = target_algos
        # 如果是单算法，立即加载其特定配置
        algo_configs = load_yaml("configs/algorithms.yaml")
        if target_algos in algo_configs:
            config_dict.update(algo_configs[target_algos])

    # 4. 命令行参数覆盖
    if args.test:
        config_dict["test"] = 1

    # 如果开启测试模式，自动强制缩减实验规模以实现“极速测试”
    if config_dict.get("test") == 1:
        config_dict["rounds"] = 2
        config_dict["epochs"] = 2
        config_dict["times"] = 2
        print("-> Fast Test Mode Active: rounds=2, epochs=2, times=2")

    # 5. 展开参数搜索 (Sweep)
    return expand_sweep(config_dict)


def expand_sweep(config_dict):
    """
    如果配置项中存在列表，则展开为多个实验配置。
    """
    sweep_keys = [k for k, v in config_dict.items() if isinstance(v, list)]
    if not sweep_keys:
        return [SimpleNamespace(**config_dict)]

    print(f"-> Detected parameter sweep for keys: {sweep_keys}")

    lists_to_product = [config_dict[k] for k in sweep_keys]
    combinations = list(itertools.product(*lists_to_product))

    configs = []
    algo_configs = load_yaml("configs/algorithms.yaml")

    for combo in combinations:
        new_config_dict = config_dict.copy()
        for i, k in enumerate(sweep_keys):
            new_config_dict[k] = combo[i]

        # 动态加载该算法对应的特定配置
        current_algo = new_config_dict.get("algo")
        if isinstance(current_algo, str) and current_algo in algo_configs:
            # 优先级说明：Sweep 组合值 > 算法特定配置 > 全局默认配置
            for spec_k, spec_v in algo_configs[current_algo].items():
                if spec_k not in sweep_keys:
                    # 如果不是正在 sweep 的键，则使用算法专属配置覆盖默认配置
                    new_config_dict[spec_k] = spec_v

        configs.append(SimpleNamespace(**new_config_dict))

    return configs


def get_pre_name(args):
    """设置并创建实验所需的保存路径和日志路径"""
    fold_path = os.path.join(
        f"{args.algo}",
        f"{args.dataset}_{args.partition}_{args.num_clients}",
    )
    if args.partition == "dirichlet":
        fold_path += f"_{args.alpha}"
    elif args.partition == "pathological":
        fold_path += f"_{args.n_class}"
    args.common_name = f"{args.epochs}_{args.batch_size}_{args.lr}"
    args.save_path = os.path.join("results_ray", fold_path)
    args.log_path = os.path.join("logs_ray", fold_path)
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.log_path, exist_ok=True)
