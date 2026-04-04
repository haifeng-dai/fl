#!/bin/bash

# =============
# Global Config
# =============
# export ALGOS="fedala,fedavg,feddpl,feddyn,fedfm,fedkd,fedlsa,fedper,fedpln,fedproc,fedproto,fedprox,fedrep,fedsa,fedtgp,fml,lgfedavg,moon,proxyfl,scaffold"

# # traditional algorithms
# export ALGOS="fedavg,feddyn,fedfm,fedlsa,fedpln,fedproc,fedprox,moon,scaffold"

# personalized algorithms
# export ALGOS="fedala,feddpl,fedkd,fedper,fedproto,fedrep,fedsa,fedtgp,fml,lgfedavg,local,proxyfl"

export ALGOS="fedtgp1"

# =============
# Data
# =============
# mnist,cifar10,cifar100,har,har_feat
export DATASETS="cifar10"
# cnn,resnet18,resnet50,harcnn,harmlp
export MODELS="cnn"
export FEATURE_DIMS="512"
export NUM_CLIENTS="10"

# =============
# Partition
# =============
# iid,dirichlet,pathological
export PARTITIONS="dirichlet"
export ALPHAS="0.1"
export N_CLASSES="2"

# =============
# Training
# =============
export EPOCHS="10"
export LRS="0.01"
export ROUNDS="100"
export BATCH_SIZES="64"
export JOIN_RATIOS="1.0"
export TIMES=1

# =============
# Compute
# =============
# 3,2,1,0  0,1,2,3
# export GPUS="0,1,2,3"
export GPUS="2,3,0,1"
export MP=1
export MAX_WORKERS_PER_GPU=10

# =============
# Test
# =============
export TEST=0

if [ "${TEST}" -eq 1 ]; then
    export ROUNDS="10"
fi

# FedTGP1 (FedTGP-Dec)
export LAMDAS_TGP1="100.0"
export HEAD_EPOCHS_TGP1="3"
export BODY_EPOCHS_TGP1="3"
export LR_HEAD_TGP1="0.01"
export LR_BODY_TGP1="0.01"
export SERVER_EPOCHS_TGP1="100"
export SERVER_LRS_TGP1="0.01"
export MARGIN_THRESHOLDS_TGP1="100.0"

# =============
# Algorithm Specific
# =============

# FedALA
export ETAS_ALA="1.0"
export RAND_PERCENTS_ALA="20"
export LAYER_IDXS_ALA="1"
export ALA_THRESHOLDS_ALA="0.1"
export NUM_PRE_LOSSES_ALA="10"

# FedDPL
export LAMBDAS_DPL="10.0"
export EPOCH_PLNS_DPL="10"
export LRS_DPL="0.01"
export BATCH_SIZE_PLNS_DPL="64"
export DEPTH_PLNS_DPL="1"
export WIDTH_PLNS_DPL="512"
export MODES_DPL="normal"
export FIXED_PROTOS_DPL=0
export INIT_EMBS_DPL="0"
export HARS_DPL="0"
export MUS_DPL="1.0"

# FedKD
export LR_GS_KD="0.01"
export ENERGIES_KD="0.9"

# FedLSA
export LAMBDAS_COM_LSA="0.1"
export ALPHAS_SEP_LSA="0.1"
export SERVER_EPOCHS_LSA="10"
export SERVER_LRS_LSA="0.01"
export TAUS_LSA="0.1"

# FedPLN
export LAMBDAS_PLN="10.0"
export EPOCH_PLNS_PLN="10"
export LR_PLNS_PLN="0.01"
export BATCH_SIZE_PLNS_PLN="64"
export DEPTH_PLNS_PLN="1"
export WIDTH_PLNS_PLN="512"
export MODES_PLN="normal"
export FIXED_PROTOS_PLN=0
export INIT_EMBS_PLN="0"
export HARS_PLN="0"

# FedProto
export MUS_PROTO="1.0"

# FedProx
export MUS_PROX="0.01"

# FedRep
export EPOCHS_HEAD_REP="5"

# FedSA
export ALPHAS_SA="0.9999"
export LAMBDAS_R_SA="0.1"
export LAMBDAS_MCL_SA="0.01"
export LAMBDAS_CC_SA="1.0"

# FedTGP
export LAMDAS_TGP="10.0"
export SERVER_EPOCHS_TGP="10"
export SERVER_LRS_TGP="0.01"
export MARGIN_THRESHOLDS_TGP="1.0"

# FML
export ALPHAS_FML="1.0"
export BETAS_FML="1.0"

# MOON
export MUS_MOON="0.01"
export TAUS_MOON="0.5"

# ProxyFL
export MUS_PROXY="1.0"

# FedTest
export MUS_TEST="10.0,1.0,0.1,0.01"

# SCAFFOLD
export GLOBAL_LRS_SCAFFOLD="1.0"

# FedDyn
export ALPHA_COEFS_DYNN="0.1"

# FedFM
export MUS_FM="1.0"

# =============
# Execution
# =============

for ALGO in ${ALGOS//,/ }; do
    echo "Starting experiment for algorithm: $ALGO"
    if [ -f "./scripts/${ALGO}.sh" ]; then
        bash ./scripts/${ALGO}.sh
    else
        echo "Script not found: ./scripts/${ALGO}.sh"
        exit 1
    fi
    echo "Finished experiment for algorithm: $ALGO"
    echo "------------------------------------------------"
done
