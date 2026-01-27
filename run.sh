#!/bin/bash

# =============
# Global Config
# =============
# fedavg,moon,fedpln,feddpl,fedproto,fedkd,fml,proxyfl,fedper,fedprox,fedsa,fedlsa,lgfedavg,fedrep,fedala,fedtgp
export ALGOS="fedavg,moon,fedpln,feddpl,fedproto,fedkd,fml,proxyfl,fedper,fedprox,fedsa,fedlsa,lgfedavg,fedrep,fedala,fedtgp"

# =============
# Data
# =============
# mnist,cifar10,cifar100,har,har_feat
export DATASETS="mnist,cifar10"
# cnn,resnet18,resnet50,harcnn,harmlp
export MODELS="cnn"
export NUM_CLIENTS="10"

# =============
# Partition
# =============
# iid,dirichlet,pathological
export PARTITIONS="pathological"
export ALPHAS="0.1,0.5"
export N_CLASSES="2"

# =============
# Training
# =============
export EPOCHS="10"
export LRS="0.01"
export ROUNDS="1000"
export BATCH_SIZES="64"
export JOIN_RATIOS="1.0"

# =============
# Compute
# =============
# 3,2,1,0  0,1,2,3
export GPUS="0,1,2,3"
export MP=1
export MAX_WORKERS_PER_GPU=10
# sequential, stream, multi_stream
export PARALLEL_MODES="multi_stream"

# =============
# Test
# =============
export TEST=0

# =============
# Algorithm Specific
# =============

# FedDPL
export LAMBDAS_DPL="10.0"
export EPOCH_PLNS_DPL="10"
export LRS_DPL="0.01"
export BATCH_SIZE_PLNS_DPL="64"
export FEATURE_DIMS_DPL="512"
export DEPTH_PLNS_DPL="1"
export WIDTH_PLNS_DPL="512"
export MODES_DPL="normal"
export FIXED_PROTOS_DPL=0
export INIT_EMBS_DPL="0"
export HARS_DPL="0"

# FedKD
export LR_GS_KD="0.01"
export ENERGIES_KD="0.9"

# FedPLN
export LAMBDAS_PLN="10.0"
export EPOCH_PLNS_PLN="10"
export LR_PLNS_PLN="0.01"
export BATCH_SIZE_PLNS_PLN="64"
export FEATURE_DIMS_PLN="512"
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

# FedSA
export ALPHAS_SA="0.5"
export LAMBDAS_R_SA="0.1"
export LAMBDAS_MCL_SA="0.1"
export LAMBDAS_CC_SA="0.1"

# FedLSA
export LAMBDAS_COM_LSA="0.1"
export ALPHAS_SEP_LSA="0.1"
export SERVER_EPOCHS_LSA="1"
export SERVER_LRS_LSA="0.01"
export TAUS_LSA="0.1"

# FML
export ALPHAS_FML="1.0"
export BETAS_FML="1.0"

# MOON
export MUS_MOON="1.0"
export TAUS_MOON="0.5"

# ProxyFL
export MUS_PROXY="1.0"

# FedRep
export EPOCHS_HEAD_REP="5"

# FedALA
export ETAS_ALA="1.0"
export RAND_PERCENTS_ALA="80"
export LAYER_IDXS_ALA="2"
export ALA_THRESHOLDS_ALA="0.1"
export NUM_PRE_LOSSES_ALA="10"

# FedTGP
export LAMDAS_TGP="10.0"
export SERVER_EPOCHS_TGP="10"
export SERVER_LRS_TGP="0.01"
export MARGIN_THRESHOLDS_TGP="1.0"
export FEATURE_DIMS_TGP="512"

# ==============================================================================
# Execution
# ==============================================================================

for ALGO in ${ALGOS//,/ }; do
    echo "Starting experiment for algorithm: $ALGO"
    case $ALGO in
        "fedavg")
            bash ./scripts/fedavg.sh
            ;;
        "moon")
            bash ./scripts/moon.sh
            ;;
        "fedpln")
            bash ./scripts/fedpln.sh
            ;;
        "feddpl")
            bash ./scripts/feddpl.sh
            ;;
        "fedproto")
            bash ./scripts/fedproto.sh
            ;;
        "fedper")
            bash ./scripts/fedper.sh
            ;;
        "fedkd")
            bash ./scripts/fedkd.sh
            ;;
        "fml")
            bash ./scripts/fml.sh
            ;;
        "proxyfl")
            bash ./scripts/proxyfl.sh
            ;;
        "fedprox")
            bash ./scripts/fedprox.sh
            ;;
        "fedsa")
            bash ./scripts/fedsa.sh
            ;;
        "fedlsa")
            bash ./scripts/fedlsa.sh
            ;;
        "lgfedavg")
            bash ./scripts/lgfedavg.sh
            ;;
        "fedrep")
            bash ./scripts/fedrep.sh
            ;;
        "fedala")
            bash ./scripts/fedala.sh
            ;;
        "fedtgp")
            bash ./scripts/fedtgp.sh
            ;;
        *)
            echo "Unknown algorithm: $ALGO. Supported: fedavg, moon, fedpln, feddpl, fedproto, fedkd, fml, proxyfl, fedper, fedprox, fedsa, fedlsa, lgfedavg, fedrep, fedala, fedtgp"
            exit 1
            ;;
    esac
    echo "Finished experiment for algorithm: $ALGO"
    echo "------------------------------------------------"
done
