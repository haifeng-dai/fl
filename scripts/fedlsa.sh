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
                                                for LAMBDA_COM in ${LAMBDAS_COM_LSA//,/ }; do
                                                    for ALPHA_SEP in ${ALPHAS_SEP_LSA//,/ }; do
                                                        for SERVER_EPOCH in ${SERVER_EPOCHS_LSA//,/ }; do
                                                            for SERVER_LR in ${SERVER_LRS_LSA//,/ }; do
                                                                for TAU in ${TAUS_LSA//,/ }; do
                                                                    uv run main.py \
                                                                        --algo fedlsa \
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
                                                                        --lambda_com $LAMBDA_COM \
                                                                        --alpha_sep $ALPHA_SEP \
                                                                        --server_epochs $SERVER_EPOCH \
                                                                        --server_lr $SERVER_LR \
                                                                        --tau $TAU
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
