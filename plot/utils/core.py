"""结果读取基础设施：命名解析、严格直读、递归平均与缓存。"""

import numbers
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from types import SimpleNamespace

import numpy as np
import torch

import src
from src.naming import build_common_name, build_result_folder


class ResultLoader:
    def __init__(self, base_dir="results"):
        self.base_dir = base_dir

    def average_recursive(self, data_list):
        if not data_list:
            return None
        first = data_list[0]
        if all(isinstance(data, numbers.Number) for data in data_list):
            return float(np.mean(data_list))
        if isinstance(first, list):
            min_len = min(len(d) for d in data_list)
            return [
                self.average_recursive([data[i] for data in data_list])
                for i in range(min_len)
            ]
        elif isinstance(first, dict):
            res = {}
            keys = set().union(
                *(data.keys() for data in data_list if isinstance(data, dict))
            )
            for key in keys:
                sub_list = [
                    d[key] for d in data_list if isinstance(d, dict) and key in d
                ]
                if sub_list:
                    res[key] = self.average_recursive(sub_list)
            return res
        return first

    def build_naming_args(self, algo, kwargs):
        values = dict(kwargs)
        values["algo"] = algo
        values.setdefault("model", "cnn")
        values.setdefault("ssl", "none")
        values.setdefault("fdg", False)
        required = [
            "dataset",
            "model",
            "partition",
            "num_clients",
            "epochs",
            "batch_size",
            "lr",
            "momentum",
            "weight_decay",
        ]
        if values["partition"] == "dirichlet":
            required.append("alpha")
        elif values["partition"] == "pathological":
            required.append("n_class")
        if values["ssl"] != "none":
            required.extend(["unlabeled_ratio", "label_ratio", "lam", "confidence"])
            if values["ssl"] == "sfd":
                required.extend(["label_domain", "unlabel_domain"])
        elif values["fdg"]:
            required.extend(["selected_domains", "target_domain"])
        missing = [key for key in required if key not in values]
        if missing:
            raise ValueError(f"缺少结果命名参数: {', '.join(missing)}")
        args = SimpleNamespace(**values)
        args.common_name = build_common_name(args)
        return args

    def resolve_folder_path(self, args, ablate_name):
        folder_path = os.path.join(self.base_dir, build_result_folder(args))
        if ablate_name:
            folder_path = os.path.join(folder_path, ablate_name)
        return folder_path

    def resolve_file_name(self, args):
        try:
            _, get_path = src.load_algorithm(args.algo)
            args.log_path = ""
            args.cur_time = 0
            get_path(args)
        except (AttributeError, KeyError) as exc:
            raise ValueError(f"算法 {args.algo} 无法生成结果文件名，缺少参数") from exc
        file_name = getattr(args, "file_name", "")
        if not file_name:
            raise ValueError(f"算法 {args.algo} 未生成有效结果文件名")
        return file_name

    def resolve_result_path(self, algo, dataset, partition, num_clients, ablate_name=None, **kwargs):
        """解析某算法结果的目录与文件名前缀（复用算法自身的 get_path 命名逻辑）。"""
        args = self.build_naming_args(
            algo,
            {
                **kwargs,
                "dataset": dataset,
                "partition": partition,
                "num_clients": num_clients,
            },
        )
        folder_path = self.resolve_folder_path(args, ablate_name)
        file_name = self.resolve_file_name(args)
        return folder_path, file_name

    def run_file_path(self, folder_path, file_name, run):
        """第 run 次 run 的结果文件完整路径。"""
        return os.path.join(folder_path, f"{file_name}_{run}.pt")

    def load_metrics_file(self, file_path, keys=None):
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"结果文件不存在: {file_path}")
        try:
            data = torch.load(file_path, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise ValueError(f"结果文件损坏: {file_path}") from exc
        if keys is None:
            return data
        return {key: data[key] for key in keys if key in data}

    def load_runs(
        self,
        algo,
        dataset,
        partition,
        num_clients,
        runs,
        ablate_name=None,
        keys=None,
        **kwargs,
    ):
        """严格直读第 0..runs-1 次 run 的结果文件；任一缺失立即抛 FileNotFoundError。"""
        folder_path, file_name = self.resolve_result_path(
            algo, dataset, partition, num_clients, ablate_name, **kwargs
        )
        return [
            self.load_metrics_file(self.run_file_path(folder_path, file_name, run), keys)
            for run in range(runs)
        ]

    def load(
        self,
        algo,
        dataset,
        partition,
        num_clients,
        specific_run=None,
        runs=None,
        ablate_name=None,
        keys=None,
        **kwargs,
    ):
        """读取结果：specific_run=k 读单次；runs=N 读第 0..N-1 次并求均值。

        两者必须显式二选一，文件缺失或损坏直接抛错，不做任何静默兜底。
        """
        if (specific_run is None) == (runs is None):
            raise ValueError(
                "必须显式指定 specific_run=k（单次 run）或 runs=N（0..N-1 求均值）之一"
            )
        folder_path, file_name = self.resolve_result_path(
            algo, dataset, partition, num_clients, ablate_name, **kwargs
        )
        if specific_run is not None:
            return self.load_metrics_file(
                self.run_file_path(folder_path, file_name, specific_run), keys
            )
        data_list = [
            self.load_metrics_file(self.run_file_path(folder_path, file_name, run), keys)
            for run in range(runs)
        ]
        keys_to_average = set().union(*(data.keys() for data in data_list))
        return {
            key: self.average_recursive(
                [data[key] for data in data_list if key in data]
            )
            for key in keys_to_average
        }

    def load_file(self, metrics_path, keys=None):
        if metrics_path.endswith("_params.pt"):
            raise ValueError("load_file 只接受指标文件，不能加载 _params.pt")
        if not metrics_path.endswith(".pt"):
            raise ValueError("load_file 只接受 .pt 指标文件")
        if not os.path.isfile(metrics_path):
            raise FileNotFoundError(metrics_path)
        return self.load_metrics_file(metrics_path, keys)


def freeze_cache_value(value):
    if isinstance(value, dict):
        return tuple(
            sorted((key, freeze_cache_value(item)) for key, item in value.items())
        )
    if isinstance(value, (list, tuple)):
        return tuple(freeze_cache_value(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted(freeze_cache_value(item) for item in value))
    return value


def batch_cached_load(loader, algo, specs, data_cache=None, keys=None, max_workers=8):
    """并行批量加载结果文件，兼容现有 data_cache。

    Args:
        loader: ResultLoader 实例
        algo: 算法名
        specs: list of kwargs dict（每个 dict 传给 loader.load）
        data_cache: 外部缓存 dict（可选），有则写入
        keys: 要提取的字段
        max_workers: 线程数
    Returns:
        list of (data or None)，顺序与 specs 一致
    """
    if data_cache is None:
        data_cache = {}

    cache_keys = []
    cached = [None] * len(specs)
    missing_idx = []
    missing_specs = []

    for i, kw in enumerate(specs):
        kt = freeze_cache_value(kw)
        kk = tuple(keys) if keys else None
        ck = (algo, kk, kt)
        cache_keys.append(ck)
        if ck in data_cache:
            cached[i] = data_cache[ck]
        else:
            missing_idx.append(i)
            missing_specs.append(kw)

    if not missing_specs:
        return cached

    def load_one(kw):
        return loader.load(algo, keys=keys, **kw)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        fut_map = {
            pool.submit(load_one, kw): i for i, kw in zip(missing_idx, missing_specs)
        }
        for fut in as_completed(fut_map):
            i = fut_map[fut]
            data = fut.result()
            cached[i] = data
            if data is not None:
                data_cache[cache_keys[i]] = data

    return cached
