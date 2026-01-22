#!/bin/bash

# Read defaults from env vars
DATASETS=${DATASETS:-"mnist"}
MODELS=${MODELS:-"cnn"}
NUM_CLIENTS=${NUM_CLIENTS:-10}
PARTITIONS=${PARTITIONS:-"iid"}
ALPHAS=${ALPHAS:-0.5}
N_CLASSES=${N_CLASSES:-2}
EPOCHS=${EPOCHS:-1}
LRS=${LRS:-0.01}
ROUNDS=${ROUNDS:-2}
BATCH_SIZES=${BATCH_SIZES:-64}
GPUS=${GPUS:-"0,1,2,3"}
MP=${MP:-0}
MAX_WORKERS_PER_GPU=${MAX_WORKERS_PER_GPU:-1}
PARALLEL_MODES=${PARALLEL_MODES:-"stream"}
TEST=${TEST:-1}

for DATASET in ${DATASETS//,/ }; do
    for MODEL in ${MODELS//,/ }; do
        for NUM_CLIENT in ${NUM_CLIENTS//,/ }; do
            for PARTITION in ${PARTITIONS//,/ }; do
                for ALPHA in ${ALPHAS//,/ }; do
                    for N_CLASS in ${N_CLASSES//,/ }; do
                        for EPOCH in ${EPOCHS//,/ }; do
                            for LR in ${LRS//,/ }; do
                                for ROUND in ${ROUNDS//,/ }; do
                                    for BATCH_SIZE in ${BATCH_SIZES//,/ }; do
                                        for PARALLEL_MODE in ${PARALLEL_MODES//,/ }; do
                                            uv run main.py \
                                                --algo fedavg \
                                                --test $TEST \
                                                --dataset $DATASET \
                                                --model $MODEL \
                                                --num_clients $NUM_CLIENT \
                                                --partition $PARTITION \
                                                --alpha $ALPHA \
                                                --n_class $N_CLASS \
                                                --epochs $EPOCH \
                                                --lr $LR \
                                                --rounds $ROUND \
                                                --batch_size $BATCH_SIZE \
                                                --gpus $GPUS \
                                                --mp $MP \
                                                --max_workers_per_gpu $MAX_WORKERS_PER_GPU \
                                                --parallel_mode $PARALLEL_MODE
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
