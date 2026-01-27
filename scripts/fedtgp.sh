#!/bin/bash

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
                                        for JOIN_RATIO in ${JOIN_RATIOS//,/ }; do
                                            for PARALLEL_MODE in ${PARALLEL_MODES//,/ }; do
                                                for LAMDA_TGP in ${LAMDAS_TGP//,/ }; do
                                                    for SERVER_EPOCH_TGP in ${SERVER_EPOCHS_TGP//,/ }; do
                                                        for SERVER_LR_TGP in ${SERVER_LRS_TGP//,/ }; do
                                                            for MARGIN_THRESHOLD_TGP in ${MARGIN_THRESHOLDS_TGP//,/ }; do
                                                                for FEATURE_DIM_TGP in ${FEATURE_DIMS_TGP//,/ }; do
                                                                    uv run main.py \
                                                                        --algo fedtgp \
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
                                                                        --join_ratio $JOIN_RATIO \
                                                                        --gpus $GPUS \
                                                                        --mp $MP \
                                                                        --max_workers_per_gpu $MAX_WORKERS_PER_GPU \
                                                                        --parallel_mode $PARALLEL_MODE \
                                                                        --lamda $LAMDA_TGP \
                                                                        --server_epochs $SERVER_EPOCH_TGP \
                                                                        --server_lr $SERVER_LR_TGP \
                                                                        --margin_threshold $MARGIN_THRESHOLD_TGP \
                                                                        --feature_dim $FEATURE_DIM_TGP
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
