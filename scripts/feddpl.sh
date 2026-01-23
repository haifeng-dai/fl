#!/bin/bash

for DATASET in ${DATASETS//,/ }; do
    for MODEL in ${MODELS//,/ }; do
        for NUM_CLIENT in ${NUM_CLIENTS//,/ }; do
            for PARTITION in ${PARTITIONS//,/ }; do
                for ALPHA in ${ALPHAS//,/ }; do
                    for N_CLASS in ${N_CLASSES//,/ }; do
                        for EPOCH in ${EPOCHS//,/ }; do
                            for LR in ${LRS//,/ }; do
                                for BATCH_SIZE in ${BATCH_SIZES//,/ }; do
                                    for PARALLEL_MODE in ${PARALLEL_MODES//,/ }; do
                                        for LAMBDA_DPL in ${LAMBDAS_DPL//,/ }; do
                                            for EPOCH_PLN_DPL in ${EPOCH_PLNS_DPL//,/ }; do
                                                for LR_PLN_DPL in ${LRS_DPL//,/ }; do
                                                    for BATCH_SIZE_PLN_DPL in ${BATCH_SIZE_PLNS_DPL//,/ }; do
                                                        for FEATURE_DIM_DPL in ${FEATURE_DIMS_DPL//,/ }; do
                                                            for DEPTH_PLN_DPL in ${DEPTH_PLNS_DPL//,/ }; do
                                                                for WIDTH_PLN_DPL in ${WIDTH_PLNS_DPL//,/ }; do
                                                                    for MODE_DPL in ${MODES_DPL//,/ }; do
                                                                        for FIXED_PROTO_DPL in ${FIXED_PROTOS_DPL//,/ }; do
                                                                            for INIT_EMB_DPL in ${INIT_EMBS_DPL//,/ }; do
                                                                                for HAR_DPL in ${HARS_DPL//,/ }; do
                                                                                    uv run main.py \
                                                                                        --algo feddpl \
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
                                                                                        --lambda_ $LAMBDA_DPL \
                                                                                        --epoch_pln $EPOCH_PLN_DPL \
                                                                                        --lr_pln $LR_PLN_DPL \
                                                                                        --batch_size_pln $BATCH_SIZE_PLN_DPL \
                                                                                        --feature_dim $FEATURE_DIM_DPL \
                                                                                        --depth_pln $DEPTH_PLN_DPL \
                                                                                        --width_pln $WIDTH_PLN_DPL \
                                                                                        --mode $MODE_DPL \
                                                                                        --fixed_proto $FIXED_PROTO_DPL \
                                                                                        --init_emb $INIT_EMB_DPL \
                                                                                        --har $HAR_DPL
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