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

# FML Specific
ALPHA_FMLS=${ALPHAS_FML:-1.0}
BETA_FMLS=${BETAS_FML:-1.0}

for DATASET in ${DATASETS//,/ }; do
    for MODEL in ${MODELS//,/ }; do
        for NUM_CLIENT in ${NUM_CLIENTS//,/ }; do
            for PARTITION in ${PARTITIONS//,/ }; do
                for EPOCH in ${EPOCHS//,/ }; do
                    for LR in ${LRS//,/ }; do
                        for ROUND in ${ROUNDS//,/ }; do
                            for ALPHA_FML in ${ALPHA_FMLS//,/ }; do
                                for BETA_FML in ${BETA_FMLS//,/ }; do
                                    uv run main.py \
                                        --algo fml \
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
                                        --alpha_fml $ALPHA_FML \
                                        --beta_fml $BETA_FML
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done
