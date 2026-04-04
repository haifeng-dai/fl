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
                                            for LAMDA_TGP in ${LAMDAS_TGP//,/ }; do
                                                for SERVER_EPOCH_TGP in ${SERVER_EPOCHS_TGP//,/ }; do
                                                    for SERVER_LR_TGP in ${SERVER_LRS_TGP//,/ }; do
                                                        for MARGIN_THRESHOLD_TGP in ${MARGIN_THRESHOLDS_TGP//,/ }; do
                                                            uv run main.py \
                                                                --algo fedtgp2 \
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
                                                                --lamda_ $LAMDA_TGP \
                                                                --server_epochs $SERVER_EPOCH_TGP \
                                                                --server_lr $SERVER_LR_TGP \
                                                                --margin_threshold $MARGIN_THRESHOLD_TGP
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
