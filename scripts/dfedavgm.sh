#!/bin/bash

# 默认支持的拓扑类型，如果需要可以在运行前通过环境变量 ADJ_TYPES 覆盖
: ${ADJ_TYPES:="ring"}

for DATASET in ${DATASETS//,/ }; do
    for MODEL in ${MODELS//,/ }; do
        for FEATURE_DIM in ${FEATURE_DIMS//,/ }; do
            for NUM_CLIENT in ${NUM_CLIENTS//,/ }; do
                for PARTITION in ${PARTITIONS//,/ }; do
                    for ALPHA in ${ALPHAS//,/ }; do
                        for N_CLASS in ${N_CLASSES//,/ }; do
                            for EPOCH in ${EPOCHS//,/ }; do
                                for LR in ${LRS//,/ }; do
                                    for BATCH_SIZE in ${BATCH_SIZES//,/ }; do
                                        for JOIN_RATIO in ${JOIN_RATIOS//,/ }; do
                                            for ADJ_TYPE in ${ADJ_TYPES//,/ }; do
                                                uv run main.py \
                                                    --algo dfedavgm \
                                                    --dataset $DATASET \
                                                    --model $MODEL \
                                                    --feature_dim $FEATURE_DIM \
                                                    --num_clients $NUM_CLIENT \
                                                    --partition $PARTITION \
                                                    --alpha $ALPHA \
                                                    --n_class $N_CLASS \
                                                    --epochs $EPOCH \
                                                    --lr $LR \
                                                    --rounds $ROUNDS \
                                                    --batch_size $BATCH_SIZE \
                                                    --join_ratio $JOIN_RATIO \
                                                    --gpus $GPUS \
                                                    --max_workers_per_gpu $MAX_WORKERS_PER_GPU \
                                                    --times $TIMES \
                                                    --test $TEST \
                                                    --adj_type $ADJ_TYPE \
                                                    --edge_p $EDGE_P \
                                                    --k $K_SMALL_WORLD \
                                                    --m $M_SCALE_FREE
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
