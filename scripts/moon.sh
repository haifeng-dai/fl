#!/bin/bash

# --- 配置 ---
DATASETS=${DATASETS:-"mnist"}
NUM_CLIENTS=${NUM_CLIENTS:-10}
PARTITIONS=${PARTITIONS:-"iid"}
ROUND=${ROUND:-2}
EPOCHS=${EPOCHS:-1}
LRS=${LRS:-0.01}
GPUS=${GPUS:-"0,1,2,3"}
TEST=${TEST:-True}

ALPHAS=${ALPHAS:-0.1}
N_CLASS=${N_CLASS:-2}

# MOON 专属参数
MUS=${MUS:-1.0}
TAUS=${TAUS:-0.5}

for DATASET in ${DATASETS//,/ }; do
    for NUM_CLIENT in ${NUM_CLIENTS//,/ }; do
        for PARTITION in ${PARTITIONS//,/ }; do
            for EPOCH in ${EPOCHS//,/ }; do
                for LR in ${LRS//,/ }; do
                    for MU in ${MUS//,/ }; do
                        for TAU in ${TAUS//,/ }; do
                            uv run main.py \
                                --algo moon \
                                --dataset $DATASET \
                                --partition $PARTITION \
                                --num_clients $NUM_CLIENT \
                                --rounds $ROUND \
                                --epochs $EPOCH \
                                --lr $LR \
                                --mu $MU \
                                --tau $TAU \
                                --gpus $GPUS \
                                --test $TEST
                        done
                    done
                done
            done
        done
    done
done