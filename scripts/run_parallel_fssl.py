#!/usr/bin/env python3
"""
FSSL 算法多任务并发运行调度器 (4 卡全共享模式)
支持配置最大并发数 MAX_JOBS，自动管理算法队列并实时汇报进度。
"""

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ALGORITHMS = [
    "fedavg_lpl",
    "fedavg_gpl",
    "fedavg_flexmatch",
    "fedlabel",
    "feddure",
    "fedloke",
    "feddb",
    "fedmatch",
    "proxyfl",
    "sage",
]


def run_experiment(algo: str, args) -> tuple[str, bool, float, str]:
    log_dir = Path("logs/parallel_runs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{algo}_{args.model}_{args.dataset}.log"

    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "src_mp",
        "-a",
        algo,
        "-m",
        args.model,
        "-d",
        args.dataset,
        "--rounds",
        str(args.rounds),
        "--gpus",
        args.gpus,
        "--workers-per-gpu",
        str(args.workers_per_gpu),
        "--log-file",
        str(log_file),
    ]

    start_time = time.time()
    print(f"[{time.strftime('%X')}] 🚀 [启动] {algo} (日志: {log_file})")

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    elapsed = time.time() - start_time
    success = proc.returncode == 0

    if success:
        print(f"[{time.strftime('%X')}] ✅ [完成] {algo} (耗时: {elapsed:.1f}s)")
    else:
        print(
            f"[{time.strftime('%X')}] ❌ [失败] {algo} (退出码: {proc.returncode})\n{proc.stdout[-500:]}"
        )

    return algo, success, elapsed, str(log_file)


def main():
    parser = argparse.ArgumentParser(description="FSSL 多任务并发运行器")
    parser.add_argument("-m", "--model", default="cnn", help="模型名称")
    parser.add_argument("-d", "--dataset", default="cifar10", help="数据集名称")
    parser.add_argument("-r", "--rounds", type=int, default=2, help="训练轮数")
    parser.add_argument(
        "--gpus",
        default="0,1,2,3",
        help="使用的 GPU 编号列表，如 0,1,2,3",
    )
    parser.add_argument(
        "-w",
        "--workers-per-gpu",
        type=int,
        default=5,
        help="每卡 Worker 数",
    )
    parser.add_argument(
        "-j",
        "--max-jobs",
        type=int,
        default=2,
        help="同时并发运行的算法数",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  FSSL 算法多任务并发调度器 (4 卡全共享)")
    print(
        f"  模型: {args.model} | 数据集: {args.dataset} | 训练轮数: {args.rounds}"
    )
    print(f"  GPU: {args.gpus} | 每卡 Worker 数: {args.workers_per_gpu}")
    print(f"  并发算法数: {args.max_jobs} | 待跑算法总数: {len(ALGORITHMS)}")
    print("=" * 60)

    start_all = time.time()
    results = []

    with ThreadPoolExecutor(max_workers=args.max_jobs) as executor:
        future_to_alg = {
            executor.submit(run_experiment, alg, args): alg
            for alg in ALGORITHMS
        }
        for future in as_completed(future_to_alg):
            results.append(future.result())

    total_elapsed = time.time() - start_all
    print("\n" + "=" * 60)
    print("  所有并发实验执行完毕！统计汇总:")
    print("=" * 60)
    for algo, success, elapsed, log_file in sorted(
        results, key=lambda x: x[0]
    ):
        status = "✅ 成功" if success else "❌ 失败"
        print(f"  - {algo:18s}: {status} | 耗时: {elapsed:5.1f}s | {log_file}")
    print("=" * 60)
    print(f"  总耗时: {total_elapsed:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
