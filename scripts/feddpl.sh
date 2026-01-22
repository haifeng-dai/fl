#!/bin/bash

# --- Configuration ---
DATASETS=${DATASETS:-"mnist"}
MODELS=${MODELS:-"cnn"}
NUM_CLIENTS=${NUM_CLIENTS:-10}
PARTITIONS=${PARTITIONS:-"iid"}
ALPHAS=${ALPHAS:-0.5}
N_CLASSES=${N_CLASSES:-2}
EPOCHS=${EPOCHS:-1}
LRS=${LRS:-0.01}
ROUNDS=${ROUNDS:-2}
BATCH_SIZES=${BATCH_SIZES:-32}
GPUS=${GPUS:-"0,1,2,3"}
MP=${MP:-0}
MAX_WORKERS_PER_GPU=${MAX_WORKERS_PER_GPU:-1}
PARALLEL_MODE=${PARALLEL_MODE:-"sequential"}
TEST=${TEST:-1}

# FedDPL Specific Args
LAMBDAS_DPL=${LAMBDAS_DPL:-1.0}
EPOCH_PLNS_DPL=${EPOCH_PLNS_DPL:-10}
LRS_DPL=${LRS_DPL:-0.01}
BATCH_SIZE_PLNS_DPL=${BATCH_SIZE_PLNS_DPL:-32}
FEATURE_DIMS_DPL=${FEATURE_DIMS_DPL:-128}
DEPTH_PLNS_DPL=${DEPTH_PLNS_DPL:-2}
WIDTH_PLNS_DPL=${WIDTH_PLNS_DPL:-128}
MODES_DPL=${MODES_DPL:-"normal"}
FIXED_PROTOS_DPL=${FIXED_PROTOS_DPL:-0}
INIT_EMBS_DPL=${INIT_EMBS_DPL:-0}
HARS_DPL=${HARS_DPL:-0}

for DATASET in ${DATASETS//,/ }; do
    for MODEL in ${MODELS//,/ }; do
        for NUM_CLIENT in ${NUM_CLIENTS//,/ }; do
            for PARTITION in ${PARTITIONS//,/ }; do
                for ALPHA in ${ALPHAS//,/ }; do
                    for N_CLASS in ${N_CLASSES//,/ }; do
                        for EPOCH in ${EPOCHS//,/ }; do
                            for LR in ${LRS//,/ }; do
                                for BATCH_SIZE in ${BATCH_SIZES//,/ }; do
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
                                                                                    --mode $MODE_DPL \
                                                                                    --batch_size_pln $BATCH_SIZE_PLN_DPL \
                                                                                    --feature_dim $FEATURE_DIM_DPL \
                                                                                    --depth_pln $DEPTH_PLN_DPL \
                                                                                    --width_pln $WIDTH_PLN_DPL \
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
