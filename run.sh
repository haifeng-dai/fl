#!/bin/bash

# 通过环境变量配置实验参数
export DATASETS=("mnist" "cifar10")
export NUM_CLIENTS=(10)
export PARTITIONS=("iid" "pathological" "dirichlet")
export ROUND=2
export EPOCHS=(2)
export LRS=(0.01)
export GPUS="0,1,2,3"
export TEST=True

export ALPHAS="0.1"
export N_CLASS="2"

# 启动 FedAvg 脚本
# bash ./scripts/fedavg.sh

# 启动 MOON 脚本
export MUS="1.0"
export TAUS="0.5"
# bash ./scripts/moon.sh

# 启动 FedPLN 脚本
export LAMBDAS="1.0, 10.0"
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
# bash ./scripts/fedpln.sh
bash ./scripts/feddpl.sh
