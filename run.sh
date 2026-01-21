#!/bin/bash

# 通过环境变量配置实验参数
export ALGO="fedkd" # fedavg,moon,fedpln,feddpl,fedproto,fedkd
export DATASETS="mnist"
export MODELS="cnn"
export NUM_CLIENTS="10"
export PARTITIONS="iid"
export ROUND="2"
export EPOCHS="10"
export LRS="0.01"
export GPUS="1,2,3"  # 0,1,2,3
export MP=1
export TEST=1
export MAX_WORKERS_PER_GPU=1  # 每个GPU的最大并行worker数，避免OOM

export ALPHAS="0.1"
export N_CLASS="2"

# 算法特定参数
export MUS="1.0"
export TAUS="0.5"

export TEMPERATURES="3.0"
export ALPHA_KDS="0.5"
export BETA_KDS="0.5"
export T_STARTS="0.9"
export T_ENDS="0.95"
export HIDDEN_DIMS="512"

export LAMBDAS="1.0"
export EPOCH_PLNS="2"
export LR_PLNS="0.01"
export MODES="normal"
export BATCH_SIZE_PLNS="32"
export FEATURE_DIMS="64"
export DEPTH_PLNS="2"
export WIDTH_PLNS="12"
export FIXED_PROTOS=0
export INIT_EMBS="0"
export HARS=0

# 根据算法选择启动对应的脚本
# 将逗号分隔的 ALGO 转换为数组并遍历
for ALG in ${ALGO//,/ }; do
    echo "Starting experiment for algorithm: $ALG"
    case $ALG in
        "fedavg")
            bash ./scripts/fedavg.sh
            ;;
        "moon")
            bash ./scripts/moon.sh
            ;;
        "fedpln")
            bash ./scripts/fedpln.sh
            ;;
        "feddpl")
            bash ./scripts/feddpl.sh
            ;;
        "fedproto")
            bash ./scripts/fedproto.sh
            ;;
        "fedkd")
            bash ./scripts/fedkd.sh
            ;;
        *)
            echo "未知算法: $ALG. 支持的算法: fedavg, moon, fedpln, feddpl, fedproto, fedkd"
            exit 1
            ;;
    esac
    echo "Finished experiment for algorithm: $ALG"
    echo "------------------------------------------------"
done
