#!/bin/bash

# 通过环境变量配置实验参数
export ALGO="fedavg"
export DATASETS="mnist"
export MODELS="cnn"
export NUM_CLIENTS="10"
export PARTITIONS="iid"
export ROUND="2"
export EPOCHS="2"
export LRS="0.01"
export GPUS="0,1,2,3"
# export NO_MP="--no_mp"
export TEST=True

export ALPHAS="0.1"
export N_CLASS="2"

# 算法特定参数
export MUS="1.0"
export TAUS="0.5"

export LAMBDAS="1.0"
export EPOCH_PLNS="2"
export LR_PLNS="0.01"
export MODES="normal"
export BATCH_SIZE_PLNS="32"
export FEATURE_DIMS="64"
export DEPTH_PLNS="2"
export WIDTH_PLNS="12"
export FIXED_PROTOS="False"
export INIT_EMBS="0"
export HARS="False"

# 根据算法选择启动对应的脚本
case $ALGO in
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
    *)
        echo "未知算法: $ALGO. 支持的算法: fedavg, moon, fedpln, feddpl, fedproto"
        exit 1
        ;;
esac
