import argparse
import logging
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

from .algorithms import load_algorithm

logger = logging.getLogger(__name__)


def configure_logging(level, log_file):
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s\n%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(log_path, mode="w", encoding="utf-8")],
    )


def select_devices(gpu_spec, workers_spec):
    try:
        gpu_ids = [int(value.strip()) for value in gpu_spec.split(",")]
    except ValueError as exc:
        raise ValueError("--gpus must be comma-separated GPU indices") from exc
    if not gpu_ids or any(gpu_id < 0 for gpu_id in gpu_ids):
        raise ValueError("--gpus must contain non-negative GPU indices")
    if len(gpu_ids) != len(set(gpu_ids)):
        raise ValueError("--gpus must not contain duplicates")

    if ":" not in workers_spec:
        try:
            count = int(workers_spec)
        except ValueError as exc:
            raise ValueError("--workers-per-gpu must be a positive integer") from exc
        if count < 1:
            raise ValueError("--workers-per-gpu must be at least 1")
        return gpu_ids, [f"cuda:{gpu_id}" for gpu_id in gpu_ids for _ in range(count)]

    try:
        counts = {}
        for item in workers_spec.split(","):
            gpu_id, count = item.split(":", maxsplit=1)
            gpu_id, count = int(gpu_id.strip()), int(count.strip())
            if gpu_id in counts:
                raise ValueError("duplicate GPU in --workers-per-gpu mapping")
            counts[gpu_id] = count
    except ValueError as exc:
        raise ValueError("--workers-per-gpu mapping must look like 0:2,1:4") from exc
    if set(counts) != set(gpu_ids) or any(count < 1 for count in counts.values()):
        raise ValueError(
            "--workers-per-gpu must give every selected GPU a positive worker count"
        )
    return gpu_ids, [
        f"cuda:{gpu_id}" for gpu_id in gpu_ids for _ in range(counts[gpu_id])
    ]


def main():
    parser = argparse.ArgumentParser(
        description="基于 torch.multiprocessing 的原生联邦学习框架"
    )
    parser.add_argument("-a", "--algo", default="fedavg")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--mu", type=float, help="FedProx 的近端正则系数")
    parser.add_argument("--gpus", help="GPU 编号，例如 0,1,3；默认读取配置文件")
    parser.add_argument(
        "--workers-per-gpu",
        help="每张 GPU 的 worker 数量，或使用映射格式，例如 0:2,1:4",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="日志级别",
    )
    parser.add_argument(
        "--log-file",
        help="日志文件路径，默认保存到 logs/src_mp/<algorithm>.log",
    )
    cli = parser.parse_args()
    configure_logging(
        cli.log_level,
        cli.log_file or Path("logs/src_mp") / f"{cli.algo.lower()}.log",
    )
    try:
        # FedMatch 会通过进程队列传输大量 sigma/psi Tensor；使用文件系统
        # 共享策略，避免 file_descriptor 策略耗尽进程文件描述符。
        torch.multiprocessing.set_sharing_strategy("file_system")
        with open(cli.config, encoding="utf-8") as file:
            raw = yaml.safe_load(file) or {}
        with open("configs/algorithms.yaml", encoding="utf-8") as file:
            algorithm_configs = yaml.safe_load(file) or {}
        cfg = {
            key: value[0] if isinstance(value, list) else value
            for key, value in raw.items()
        }
        cfg.update(algorithm_configs.get(cli.algo.lower(), {}))
        configured_ssl = cfg.get("ssl", "none")
        cfg.update(algo=cli.algo, ssl=configured_ssl)
        for key in ("rounds", "epochs", "mu"):
            if getattr(cli, key) is not None:
                cfg[key] = getattr(cli, key)
        if not torch.cuda.is_available():
            raise RuntimeError("src_mp requires CUDA")
        gpu_ids, devices = select_devices(
            cli.gpus or str(cfg["gpus"]),
            cli.workers_per_gpu or str(cfg["max_workers_per_gpu"]),
        )
        if max(gpu_ids) >= torch.cuda.device_count():
            raise ValueError("requested GPU is not visible")
        torch.manual_seed(cfg["seed"])
        load_algorithm(cli.algo)(SimpleNamespace(**cfg), devices).fit()
    except Exception:
        logger.exception("Training failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
