#!/usr/bin/env bash
# ==============================================================================
# FSSL 算法多任务并发运行脚本 (4 卡全共享模式)
# ==============================================================================
set -e

# 待跑算法列表
ALGORITHMS=(
    "fedavg_lpl"
    "fedavg_gpl"
    "fedavg_flexmatch"
    "fedlabel"
    "feddure"
    "fedloke"
    "feddb"
    "fedmatch"
    "proxyfl"
    "sage"
)

# 默认参数 (支持通过环境变量覆盖)
MODEL="${MODEL:-cnn}"
DATASET="${DATASET:-cifar10}"
ROUNDS="${ROUNDS:-2}"
GPUS="${GPUS:-0,1,2,3}"
WORKERS="${WORKERS:-5}"
MAX_JOBS="${MAX_JOBS:-2}"       # 同时并发运行的算法数量，默认 2 个

mkdir -p logs/parallel_runs

echo "=========================================================="
echo "  FSSL 算法多任务并发套件 (4 卡全共享)"
echo "  模型: $MODEL | 数据集: $DATASET | 训练轮数: $ROUNDS"
echo "  GPU: $GPUS | 每卡 Worker 数: $WORKERS"
echo "  同时并发算法数: $MAX_JOBS | 待跑算法总数: ${#ALGORITHMS[@]}"
echo "=========================================================="

# 运行单个算法的函数
run_single_algorithm() {
    local alg="$1"
    local log_file="logs/parallel_runs/${alg}_${MODEL}_${DATASET}.log"
    local start_time
    start_time=$(date +%s)
    
    echo ">>> [$(date '+%Y-%m-%d %H:%M:%S')] [启动] 算法: $alg (日志: $log_file)"
    
    uv run python -m src_mp \
        -a "$alg" \
        -m "$MODEL" \
        -d "$DATASET" \
        --rounds "$ROUNDS" \
        --gpus "$GPUS" \
        --workers-per-gpu "$WORKERS" \
        --log-file "$log_file"
        
    local end_time
    end_time=$(date +%s)
    local elapsed=$((end_time - start_time))
    echo ">>> [$(date '+%Y-%m-%d %H:%M:%S')] [完成] 算法: $alg (耗时: ${elapsed}s)"
}

# 并发调度队列控制
job_count=0
pids=()

for alg in "${ALGORITHMS[@]}"; do
    run_single_algorithm "$alg" &
    pids+=($!)
    ((job_count++))
    
    # 当达到最大并发任务数时，等待其中一个任务完成
    if (( job_count >= MAX_JOBS )); then
        wait -n
        ((job_count--))
    fi
done

# 等待剩余后台任务全部完成
wait

echo ""
echo "=========================================================="
echo "  所有 ${#ALGORITHMS[@]} 个算法已全部并发执行完成！"
echo "=========================================================="
