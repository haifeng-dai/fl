#!/bin/bash

# 变量已从 run.sh 通过 export 继承

# 设置默认值，防止未定义
ALPHAS_SA=${ALPHAS_SA:-0.5}
LAMBDAS_R_SA=${LAMBDAS_R_SA:-0.1}
LAMBDAS_MCL_SA=${LAMBDAS_MCL_SA:-0.1}
LAMBDAS_CC_SA=${LAMBDAS_CC_SA:-0.1}

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
                                        for PARALLEL_MODE in ${PARALLEL_MODES//,/ }; do
                                            # FedSA specific hyperparameters
                                            for ALPHA_SA in ${ALPHAS_SA//,/ }; do
                                                for LAMBDA_R in ${LAMBDAS_R_SA//,/ }; do
                                                    for LAMBDA_MCL in ${LAMBDAS_MCL_SA//,/ }; do
                                                        for LAMBDA_CC in ${LAMBDAS_CC_SA//,/ }; do
                                                            uv run main.py \
                                                                --algo fedsa \
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
                                                                --gpus $GPUS \
                                                                --mp $MP \
                                                                --max_workers_per_gpu $MAX_WORKERS_PER_GPU \
                                                                --parallel_mode $PARALLEL_MODE \
                                                                --alpha_sa $ALPHA_SA \
                                                                --lambda_r $LAMBDA_R \
                                                                --lambda_mcl $LAMBDA_MCL \
                                                                --lambda_cc $LAMBDA_CC
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
