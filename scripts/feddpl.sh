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

# FedDPL 专属参数
LAMBDAS=${LAMBDAS:-10.0}
EPOCH_PLNS=${EPOCH_PLNS:-10}
LR_PLNS=${LR_PLNS:-0.01}
MODES=${MODES:-"normal"}
BATCH_SIZE_PLNS=${BATCH_SIZE_PLNS:-32}
FEATURE_DIMS=${FEATURE_DIMS:-128}
DEPTH_PLNS=${DEPTH_PLNS:-2}
WIDTH_PLNS=${WIDTH_PLNS:-128}
FIXED_PROTOS=${FIXED_PROTOS:-0}
INIT_EMBS=${INIT_EMBS:-0}
HARS=${HARS:-0}

for DATASET in ${DATASETS//,/ }; do
    for MODEL in ${MODELS//,/ }; do
        for NUM_CLIENT in ${NUM_CLIENTS//,/ }; do
            for PARTITION in ${PARTITIONS//,/ }; do
                for EPOCH in ${EPOCHS//,/ }; do
                    for LR in ${LRS//,/ }; do
                        for LAMBDA in ${LAMBDAS//,/ }; do
                            for EPOCH_PLN in ${EPOCH_PLNS//,/ }; do
                                for LR_PLN in ${LR_PLNS//,/ }; do
                                    for MODE in ${MODES//,/ }; do
                                        for BATCH_SIZE_PLN in ${BATCH_SIZE_PLNS//,/ }; do
                                            for FEATURE_DIM in ${FEATURE_DIMS//,/ }; do
                                                for DEPTH_PLN in ${DEPTH_PLNS//,/ }; do
                                                    for WIDTH_PLN in ${WIDTH_PLNS//,/ }; do
                                                        for FIXED_PROTO in ${FIXED_PROTOS//,/ }; do
                                                            for INIT_EMB in ${INIT_EMBS//,/ }; do
                                                                for HAR in ${HARS//,/ }; do
                                                                    uv run main.py \
                                                                        --algo feddpl \
                                                                        --dataset $DATASET \
                                                                        --model $MODEL \
                                                                        --partition $PARTITION \
                                                                        --num_clients $NUM_CLIENT \
                                                                        --rounds $ROUND \
                                                                        --epochs $EPOCH \
                                                                        --lr $LR \
                                                                        --lambda_ $LAMBDA \
                                                                        --epoch_pln $EPOCH_PLN \
                                                                        --lr_pln $LR_PLN \
                                                                        --batch_size_pln $BATCH_SIZE_PLN \
                                                                        --feature_dim $FEATURE_DIM \
                                                                        --depth_pln $DEPTH_PLN \
                                                                        --width_pln $WIDTH_PLN \
                                                                        --fixed_proto $FIXED_PROTO \
                                                                        --init_emb $INIT_EMB \
                                                                        --har $HAR \
                                                                        --mode $MODE \
                                                                        --alpha $ALPHAS \
                                                                        --n_classes $N_CLASS \
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
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done
