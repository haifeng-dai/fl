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

# FedPLN Specific Args
LAMBDAS_PLN=${LAMBDAS_PLN:-1.0}
EPOCH_PLNS_PLN=${EPOCH_PLNS_PLN:-10}
LR_PLNS_PLN=${LR_PLNS_PLN:-0.01}
MODES_PLN=${MODES_PLN:-"normal"}
BATCH_SIZE_PLNS_PLN=${BATCH_SIZE_PLNS_PLN:-32}
FEATURE_DIMS_PLN=${FEATURE_DIMS_PLN:-64}
DEPTH_PLNS_PLN=${DEPTH_PLNS_PLN:-2}
WIDTH_PLNS_PLN=${WIDTH_PLNS_PLN:-12}
FIXED_PROTOS_PLN=${FIXED_PROTOS_PLN:-0}
INIT_EMBS_PLN=${INIT_EMBS_PLN:-0}
HARS_PLN=${HARS_PLN:-0}

for DATASET in ${DATASETS//,/ }; do
    for MODEL in ${MODELS//,/ }; do
        for NUM_CLIENT in ${NUM_CLIENTS//,/ }; do
            for PARTITION in ${PARTITIONS//,/ }; do
                for ALPHA in ${ALPHAS//,/ }; do
                    for N_CLASS in ${N_CLASSES//,/ }; do
                        for EPOCH in ${EPOCHS//,/ }; do
                            for LR in ${LRS//,/ }; do
                                for BATCH_SIZE in ${BATCH_SIZES//,/ }; do
                                    for LAMBDA_PLN in ${LAMBDAS_PLN//,/ }; do
                                        for EPOCH_PLN_PLN in ${EPOCH_PLNS_PLN//,/ }; do
                                            for LR_PLN_PLN in ${LR_PLNS_PLN//,/ }; do
                                                for MODE_PLN in ${MODES_PLN//,/ }; do
                                                    for BATCH_SIZE_PLN_PLN in ${BATCH_SIZE_PLNS_PLN//,/ }; do
                                                        for FEATURE_DIM_PLN in ${FEATURE_DIMS_PLN//,/ }; do
                                                            for DEPTH_PLN_PLN in ${DEPTH_PLNS_PLN//,/ }; do
                                                                for WIDTH_PLN_PLN in ${WIDTH_PLNS_PLN//,/ }; do
                                                                    for FIXED_PROTO_PLN in ${FIXED_PROTOS_PLN//,/ }; do
                                                                        for INIT_EMB_PLN in ${INIT_EMBS_PLN//,/ }; do
                                                                            for HAR_PLN in ${HARS_PLN//,/ }; do
                                                                                uv run main.py \
                                                                                    --algo fedpln \
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
                                                                                    --lambda_ $LAMBDA_PLN \
                                                                                    --epoch_pln $EPOCH_PLN_PLN \
                                                                                    --lr_pln $LR_PLN_PLN \
                                                                                    --mode $MODE_PLN \
                                                                                    --batch_size_pln $BATCH_SIZE_PLN_PLN \
                                                                                    --feature_dim $FEATURE_DIM_PLN \
                                                                                    --depth_pln $DEPTH_PLN_PLN \
                                                                                    --width_pln $WIDTH_PLN_PLN \
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
