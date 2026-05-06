import datetime
import json
import os
import sys
import time
import traceback
from contextlib import ExitStack, redirect_stderr, redirect_stdout

import src

def run_experiment(args, t):
    """
    运行单次实验的核心逻辑
    """
    # 动态加载当前算法所需的组件
    server_cls, get_path = src.load_algorithm(args.algo)

    # 记录当前实验索引
    args.cur_time = t
    # 设置随机种子
    src.set_seed(args.seed)

    start_time = time.time()
    log_path = get_path(args)

    # 资源管理：自动处理日志文件打开与标准流重定向
    with ExitStack() as stack:
        if not args.test:
            # 创建日志目录并将 stdout/stderr 导向文件
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            log_f = stack.enter_context(open(log_path, "w", encoding="utf-8", buffering=1))
            stack.enter_context(redirect_stdout(log_f))
            stack.enter_context(redirect_stderr(log_f))

        try:
            # 记录本次实验的完整配置到日志中
            print(f"\n{'=' * 30} Experiment {t + 1}/{args.times} {'=' * 30}")
            print(f"Config:\n{json.dumps(vars(args), indent=4, ensure_ascii=False)}\n{'-' * 80}")
            print(f"Seed: {args.seed} | Start Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

            # 4. 数据准备（如果是首次运行则会生成划分，否则直接加载）
            src.prepare_data(args)

            # 5. 实例化 Server 并执行训练与保存
            server = server_cls(args=args)
            server.fit()
            server.save()

            end_time = time.time()
            print(f"\nEnd Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Total Duration: {datetime.timedelta(seconds=int(end_time - start_time))}")
        except Exception:
            # 捕获异常并打印追踪信息，确保即使失败也能在日志中看到原因
            traceback.print_exc()
            raise

def main():
    # 1. 环境初始化：加载系统配置与运行时环境
    src.setup_runtime_env()
    configs = src.get_config()

    # 2. 初始化全局 Ray 资源（以第一个配置的 GPU 设定为准）
    src.init_ray(configs[0])

    try:
        # 遍历配置列表，依次执行实验任务
        for cfg_idx, args in enumerate(configs):
            # 1. 实验分割线（主控制台可见）
            print(f"\n{'#' * 40}\n# Running Task {cfg_idx + 1}/{len(configs)}: {args.algo}\n{'#' * 40}")

            # 2. 初始化实验保存路径
            src.get_pre_name(args)

            # 3. 循环执行多次实验：只需传入 args 对象和当前索引 t
            for t in range(args.times):
                run_experiment(args, t)
    finally:
        # 确保无论实验是否成功，最后都关闭 Ray 集群释放资源
        src.shutdown_ray()

if __name__ == "__main__":
    main()
