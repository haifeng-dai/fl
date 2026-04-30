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
                                            for LAMBDA_PLN in ${LAMBDAS_PLN//,/ }; do
                                                for EPOCH_PLN_PLN in ${EPOCH_PLNS_PLN//,/ }; do
                                                    for LR_PLN_PLN in ${LR_PLNS_PLN//,/ }; do
                                                        for BATCH_SIZE_PLN_PLN in ${BATCH_SIZE_PLNS_PLN//,/ }; do
                                                            for DEPTH_PLN_PLN in ${DEPTH_PLNS_PLN//,/ }; do
                                                                for WIDTH_PLN_PLN in ${WIDTH_PLNS_PLN//,/ }; do
                                                                    for MODE_PLN in ${MODES_PLN//,/ }; do
                                                                        for FIXED_PROTO_PLN in ${FIXED_PROTOS_PLN//,/ }; do
                                                                            for INIT_EMB_PLN in ${INIT_EMBS_PLN//,/ }; do
                                                                                for HAR_PLN in ${HARS_PLN//,/ }; do
                                                                                    uv run main.py \
                                                                                        --algo fedpln \
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
                                                                                        --lambda_ $LAMBDA_PLN \
                                                                                        --epoch_pln $EPOCH_PLN_PLN \
                                                                                        --lr_pln $LR_PLN_PLN \
                                                                                        --batch_size_pln $BATCH_SIZE_PLN_PLN \
                                                                                        --depth_pln $DEPTH_PLN_PLN \
                                                                                        --width_pln $WIDTH_PLN_PLN \
                                                                                        --mode $MODE_PLN \
                                                                                        --fixed_proto $FIXED_PROTO_PLN \
                                                                                        --init_emb $INIT_EMB_PLN \
                                                                                        --har $HAR_PLN
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
            done
        done
    done
done
