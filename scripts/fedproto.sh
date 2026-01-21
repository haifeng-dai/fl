#!/bin/bash

# --- 配置 ---
DATASETS=${DATASETS:-"mnist"}
MODELS=${MODELS:-"cnn"}
NUM_CLIENTS=${NUM_CLIENTS:-10}
PARTITIONS=${PARTITIONS:-"iid"}
ROUND=${ROUND:-2}
EPOCHS=${EPOCHS:-1}
LRS=${LRS:-0.01}
GPUS=${GPUS:-"0"}
TEST=${TEST:-1}
MP=${MP:-0}
MAX_WORKERS_PER_GPU=${MAX_WORKERS_PER_GPU:-1}  # 默认为1

ALPHAS=${ALPHAS:-0.5}
N_CLASS=${N_CLASS:-2}

# FedProto 专属参数
MUS=${MUS:-1.0}

for DATASET in ${DATASETS//,/ }; do
    for MODEL in ${MODELS//,/ }; do
        for NUM_CLIENT in ${NUM_CLIENTS//,/ }; do
            for PARTITION in ${PARTITIONS//,/ }; do
                for EPOCH in ${EPOCHS//,/ }; do
                    for LR in ${LRS//,/ }; do
                        for MU in ${MUS//,/ }; do
                            uv run main.py \
                                --algo fedproto \
                                --dataset $DATASET \
                                --model $MODEL \
                                --partition $PARTITION \
                                --num_clients $NUM_CLIENT \
                                --rounds $ROUND \
                                --epochs $EPOCH \
                                --lr $LR \
                                --mu $MU \
                                --gpus $GPUS \
                                --mp $MP \
                                --test $TEST \
                                --max_workers_per_gpu $MAX_WORKERS_PER_GPU
                        done
                    done
                done
            done
        done
    done
done
