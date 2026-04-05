#!/bin/bash

for DATASET in ${DATASETS//,/ }; do
    for MODEL in ${MODELS//,/ }; do
        for FEATURE_DIM in ${FEATURE_DIMS//,/ }; do
            for NUM_CLIENT in ${NUM_CLIENTS//,/ }; do
                for PARTITION in ${PARTITIONS//,/ }; do
                    for ALPHA in ${ALPHAS//,/ }; do
                        for N_CLASS in ${N_CLASSES//,/ }; do
                            for BATCH_SIZE in ${BATCH_SIZES//,/ }; do
                                for JOIN_RATIO in ${JOIN_RATIOS//,/ }; do
                                    for LAMDA_DPC in ${LAMDAS_DPC//,/ }; do
                                        for HEAD_EPOCH in ${HEAD_EPOCHS_DPC//,/ }; do
                                            for BODY_EPOCH in ${BODY_EPOCHS_DPC//,/ }; do
                                                for LR_HEAD in ${LR_HEAD_DPC//,/ }; do
                                                    for LR_BODY in ${LR_BODY_DPC//,/ }; do
                                                        for SERVER_EPOCH_DPC in ${SERVER_EPOCHS_DPC//,/ }; do
                                                            for SERVER_LR_DPC in ${SERVER_LRS_DPC//,/ }; do
                                                                for MARGIN_THRESHOLD_DPC in ${MARGIN_THRESHOLDS_DPC//,/ }; do
                                                                    uv run main.py \
                                                                        --algo feddpc \
                                                                        --dataset $DATASET \
                                                                        --model $MODEL \
                                                                        --feature_dim $FEATURE_DIM \
                                                                        --num_clients $NUM_CLIENT \
                                                                        --partition $PARTITION \
                                                                        --alpha $ALPHA \
                                                                        --n_class $N_CLASS \
                                                                        --rounds $ROUNDS \
                                                                        --batch_size $BATCH_SIZE \
                                                                        --join_ratio $JOIN_RATIO \
                                                                        --gpus $GPUS \
                                                                        --mp $MP \
                                                                        --max_workers_per_gpu $MAX_WORKERS_PER_GPU \
                                                                        --times $TIMES \
                                                                        --test $TEST \
                                                                        --lamda_ $LAMDA_DPC \
                                                                        --head_epochs $HEAD_EPOCH \
                                                                        --body_epochs $BODY_EPOCH \
                                                                        --lr_head $LR_HEAD \
                                                                        --lr_body $LR_BODY \
                                                                        --server_epochs $SERVER_EPOCH_DPC \
                                                                        --server_lr $SERVER_LR_DPC \
                                                                        --margin_threshold $MARGIN_THRESHOLD_DPC
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
