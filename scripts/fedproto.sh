#!/bin/bash

# --- Configuration ---
DATASETS=${DATASETS:-"mnist"}
MODELS=${MODELS:-"cnn"}
NUM_CLIENTS=${NUM_CLIENTS:-10}
PARTITIONS=${PARTITIONS:-"iid"}
ALPHAS=${ALPHAS:-0.5}
N_CLASSES=${N_CLASSES:-2}
EPOCHS=${EPOCHS:-1}
LRS=${LRS:-0.01}
ROUNDS=${ROUNDS:-2}
BATCH_SIZES=${BATCH_SIZES:-32}
GPUS=${GPUS:-"0"}
MP=${MP:-0}
MAX_WORKERS_PER_GPU=${MAX_WORKERS_PER_GPU:-1}
PARALLEL_MODE=${PARALLEL_MODE:-"sequential"}
TEST=${TEST:-1}

# FedProto Specific Args
MUS_PROTO=${MUS_PROTO:-1.0}

for DATASET in ${DATASETS//,/ }; do
    for MODEL in ${MODELS//,/ }; do
        for NUM_CLIENT in ${NUM_CLIENTS//,/ }; do
            for PARTITION in ${PARTITIONS//,/ }; do
                for ALPHA in ${ALPHAS//,/ }; do
                    for N_CLASS in ${N_CLASSES//,/ }; do
                        for EPOCH in ${EPOCHS//,/ }; do
                            for LR in ${LRS//,/ }; do
                                for BATCH_SIZE in ${BATCH_SIZES//,/ }; do
                                    for MU_PROTO in ${MUS_PROTO//,/ }; do
                                        uv run main.py \
                                            --algo fedproto \
                                            --test $TEST \
                                            --dataset $DATASET \
                                            --model $MODEL \
                                            --num_clients $NUM_CLIENT \
                                            --partition $PARTITION \
                                            --alpha $ALPHA \
                                            --n_class $N_CLASS \
                                            --epochs $EPOCH \
                                            --lr $LR \
                                            --rounds $ROUNDS \
                                            --batch_size $BATCH_SIZE \
                                            --gpus $GPUS \
                                            --mp $MP \
                                            --max_workers_per_gpu $MAX_WORKERS_PER_GPU \
                                            --parallel_mode $PARALLEL_MODE \
                                            --mu $MU_PROTO
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
