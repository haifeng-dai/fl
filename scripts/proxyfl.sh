#!/bin/bash

# --- 配置 ---
DATASETS=${DATASETS:-"mnist"}
MODELS=${MODELS:-"cnn"}
NUM_CLIENTS=${NUM_CLIENTS:-10}
PARTITIONS=${PARTITIONS:-"iid"}
EPOCHS=${EPOCHS:-1}
LRS=${LRS:-0.01}
ROUNDS=${ROUNDS:-2}
GPUS=${GPUS:-"0"}
TEST=${TEST:-1}
MP=${MP:-0}
MAX_WORKERS_PER_GPU=${MAX_WORKERS_PER_GPU:-1}

# ProxyFL Specific
MUS_PROXY=${MUS_PROXY:-1.0}

for DATASET in ${DATASETS//,/ }; do
    for MODEL in ${MODELS//,/ }; do
        for NUM_CLIENT in ${NUM_CLIENTS//,/ }; do
            for PARTITION in ${PARTITIONS//,/ }; do
                for EPOCH in ${EPOCHS//,/ }; do
                    for LR in ${LRS//,/ }; do
                        for ROUND in ${ROUNDS//,/ }; do
                            for MU_PROXY in ${MUS_PROXY//,/ }; do
                                uv run main.py \
                                    --algo proxyfl \
                                    --test $TEST \
                                    --dataset $DATASET \
                                    --model $MODEL \
                                    --num_clients $NUM_CLIENT \
                                    --partition $PARTITION \
                                    --epochs $EPOCH \
                                    --lr $LR \
                                    --rounds $ROUND \
                                    --gpus $GPUS \
                                    --mp $MP \
                                    --max_workers_per_gpu $MAX_WORKERS_PER_GPU \
                                    --mu $MU_PROXY
                            done
                        done
                    done
                done
            done
        done
    done
done
