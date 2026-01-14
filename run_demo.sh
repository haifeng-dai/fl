#!/bin/bash

# 多GPU联邦学习执行脚本 (使用 uv)

# --- 配置 ---
DATASET="mnist"
NUM_CLIENTS=10
ROUNDS=10
EPOCHS=1
LR=0.01
GPUS="0,1"  # 根据你可用的GPU进行修改，例如 "0" 或 "0,1,2,3"

# # --- 算法 1: FedAvg (Dirichlet 分区) ---
# echo "使用 uv 启动 FedAvg (Dirichlet 分区)..."
# uv run main.py \
#     --algo fedavg \
#     --dataset $DATASET \
#     --partition dirichlet \
#     --alpha 0.5 \
#     --num_clients $NUM_CLIENTS \
#     --rounds $ROUNDS \
#     --epochs $EPOCHS \
#     --lr $LR \
#     --gpus $GPUS

# --- 算法 2: MOON (病态分区) ---
取消注释以运行 MOON
echo "使用 uv 启动 MOON (病态分区)..."
uv run main.py \
    --algo moon \
    --dataset $DATASET \
    --partition pathological \
    --n_classes 2 \
    --mu 1.0 \
    --tau 0.5 \
    --num_clients $NUM_CLIENTS \
    --rounds $ROUNDS \
    --epochs $EPOCHS \
    --lr $LR \
    --gpus $GPUS
