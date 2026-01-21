#!/bin/bash

# --- 配置 ---
DATASETS=${DATASETS:-"mnist"}
MODELS=${MODELS:-"cnn"}
NUM_CLIENTS=${NUM_CLIENTS:-10}
PARTITIONS=${PARTITIONS:-"iid"}
ROUND=${ROUND:-2}
EPOCHS=${EPOCHS:-1}
LRS=${LRS:-0.01}
GPUS=${GPUS:-"0,1,2,3"}
TEST=${TEST:-1}
MP=${MP:-0}
MAX_WORKERS_PER_GPU=${MAX_WORKERS_PER_GPU:-1}  # 默认为1

ALPHAS=${ALPHAS:-0.5}
N_CLASS=${N_CLASS:-2}

# FedKD 特定参数
MENTEE_LRS=${MENTEE_LRS:-0.005}
LR_DECAY_GAMMAS=${LR_DECAY_GAMMAS:-0.99}
ENERGIES=${ENERGIES:-0.95}

for DATASET in ${DATASETS//,/ }; do
    for MODEL in ${MODELS//,/ }; do
        for NUM_CLIENT in ${NUM_CLIENTS//,/ }; do
            for PARTITION in ${PARTITIONS//,/ }; do
                for EPOCH in ${EPOCHS//,/ }; do
                    for LR in ${LRS//,/ }; do
                        for MENTEE_LR in ${MENTEE_LRS//,/ }; do
                            for GAMMA in ${LR_DECAY_GAMMAS//,/ }; do
                                for ENERGY in ${ENERGIES//,/ }; do
                                    uv run main.py \
                                        --algo fedkd \
                                        --dataset $DATASET \
                                        --model $MODEL \
                                        --partition $PARTITION \
                                        --num_clients $NUM_CLIENT \
                                        --rounds $ROUND \
                                        --epochs $EPOCH \
                                        --lr $LR \
                                        --gpus $GPUS \
                                        --mp $MP \
                                        --test $TEST \
                                        --max_workers_per_gpu $MAX_WORKERS_PER_GPU \
                                        --mentee_learning_rate $MENTEE_LR \
                                        --learning_rate_decay_gamma $GAMMA \
                                        --energy $ENERGY
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done
