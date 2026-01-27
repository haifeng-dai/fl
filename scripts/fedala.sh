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
                                                for ETA_ALA in ${ETAS_ALA//,/ }; do
                                                    for RAND_PERCENT_ALA in ${RAND_PERCENTS_ALA//,/ }; do
                                                        for LAYER_IDX_ALA in ${LAYER_IDXS_ALA//,/ }; do
                                                            for ALA_THRESHOLD_ALA in ${ALA_THRESHOLDS_ALA//,/ }; do
                                                                for NUM_PRE_LOSS_ALA in ${NUM_PRE_LOSSES_ALA//,/ }; do
                                                                    uv run main.py \
                                                                        --algo fedala \
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
                                                                        --eta $ETA_ALA \
                                                                        --rand_percent $RAND_PERCENT_ALA \
                                                                        --layer_idx $LAYER_IDX_ALA \
                                                                        --ala_threshold $ALA_THRESHOLD_ALA \
                                                                        --num_pre_loss $NUM_PRE_LOSS_ALA
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
