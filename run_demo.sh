#!/bin/bash

# Multi-GPU Federated Learning Execution Script (using uv)

# --- Configuration ---
DATASET="mnist"
NUM_CLIENTS=10
ROUNDS=10
EPOCHS=1
LR=0.01
GPUS="0,1"  # Change this to match your available GPUs, e.g., "0" or "0,1,2,3"

# --- Algorithm 1: FedAvg with Dirichlet Partition ---
echo "Starting FedAvg with Dirichlet partition using uv..."
uv run main.py \
    --algo fedavg \
    --dataset $DATASET \
    --partition dirichlet \
    --alpha 0.5 \
    --num_clients $NUM_CLIENTS \
    --rounds $ROUNDS \
    --epochs $EPOCHS \
    --lr $LR \
    --gpus $GPUS

# --- Algorithm 2: MOON with Pathological Partition ---
# Uncomment below to run MOON
# echo "Starting MOON with pathological partition using uv..."
# uv run main.py \
#     --algo moon \
#     --dataset $DATASET \
#     --partition pathological \
#     --n_classes 2 \
#     --mu 1.0 \
#     --tau 0.5 \
#     --num_clients $NUM_CLIENTS \
#     --rounds $ROUNDS \
#     --epochs $EPOCHS \
#     --lr $LR \
#     --gpus $GPUS