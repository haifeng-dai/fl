#!/bin/bash

# 变量已从 run.sh 通过 export 继承

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
                                            for ALPHA_FML in ${ALPHAS_FML//,/ }; do
                                                for BETA_FML in ${BETAS_FML//,/ }; do
                                                    uv run main.py \
                                                        --algo fml \
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
                                                        --parallel_mode $PARALLEL_MODE \
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
            done
        done
    done
done