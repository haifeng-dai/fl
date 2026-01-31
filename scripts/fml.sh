#!/bin/bash

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
                                            for ALPHA_FML in ${ALPHAS_FML//,/ }; do
                                                for BETA_FML in ${BETAS_FML//,/ }; do
                                                    uv run main.py \
                                                    --algo fml \
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
                                                        --mp $MP \
                                                        --max_workers_per_gpu $MAX_WORKERS_PER_GPU \
                                                        --times $TIMES \
                                                        --test $TEST \
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
