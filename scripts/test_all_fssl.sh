#!/usr/bin/env bash
# ==============================================================================
# 批量顺序测试所有 FSSL 算法运行脚本 (src_mp)
# ==============================================================================
set -e

# 测试算法列表
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

# 默认测试参数 (可通过环境变量或传参覆盖)
MODEL="${MODEL:-cnn}"
DATASET="${DATASET:-cifar10}"
ROUNDS="${ROUNDS:-2}"
GPUS="${GPUS:-0,1,2,3}"
WORKERS="${WORKERS:-5}"

echo "=========================================================="
echo "  FSSL 算法批量测试套件"
echo "  模型: $MODEL | 数据集: $DATASET | 测试轮数: $ROUNDS"
echo "  GPU: $GPUS | 每卡 Worker 数: $WORKERS"
echo "  待测算法总数: ${#ALGORITHMS[@]}"
echo "=========================================================="

mkdir -p logs/test_runs

for alg in "${ALGORITHMS[@]}"; do
    echo ""
    echo "----------------------------------------------------------"
    echo ">>> [$(date '+%Y-%m-%d %H:%M:%S')] 正在测试算法: $alg"
    echo "----------------------------------------------------------"
    
    LOG_FILE="logs/test_runs/${alg}_${MODEL}_${DATASET}.log"
    
    uv run python -m src_mp \
        -a "$alg" \
        -m "$MODEL" \
        -d "$DATASET" \
        --rounds "$ROUNDS" \
        --gpus "$GPUS" \
        --workers-per-gpu "$WORKERS" \
        --log-file "$LOG_FILE"
        
    echo ">>> [$(date '+%Y-%m-%d %H:%M:%S')] 算法 $alg 测试通过！日志已存至: $LOG_FILE"
done

echo ""
echo "=========================================================="
echo "  所有 ${#ALGORITHMS[@]} 个算法已全部顺序测试完毕并成功通过！"
echo "=========================================================="
