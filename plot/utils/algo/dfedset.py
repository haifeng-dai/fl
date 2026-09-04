"""DFedSET 的专属绘图辅助：通信量模型与消融实验配置（自 notebook 上收）。"""

import numpy as np

# 参数量统计（完整 CNN 与去中心化对比涉及的模块）
MODEL_PARAMS = 2122186  # 完整 CNN 参数量
BODY_PARAMS = 2117056  # DFedPGP 提取器（不含分类头）
PROTOS_PARAMS = 5120  # 原型 [10, 512]
DFEDSET_EXTRA = 5130  # S[10,512] + W[10]
PARAM_TO_GB = 4 / (1024 * 1024 * 1024)  # 每个参数 4 bytes → GB

# DFedSET 消融实验配置：标签 → (消融子目录名, 组件类别)
ABLATION_CONFIGS = {
    "Base": {"ablate_name": None, "category": "DFedSET"},
    "-": {"ablate_name": "relay", "category": "Relay"},
    "Plain Aggr.": {"ablate_name": "aggregator", "category": "Aggregation"},
    "count": {"ablate_name": "confidence_count", "category": "Weighting"},
    "none": {"ablate_name": "confidence_none", "category": "Weighting"},
    "global (0.001)": {
        "ablate_name": "trigger_global_gamma_0.001",
        "category": "Trigger",
    },
    "global (0.005)": {
        "ablate_name": "trigger_global_gamma_0.005",
        "category": "Trigger",
    },
    "all": {"ablate_name": "trigger_all", "category": "Trigger"},
}


def calc_cum_comm(data, algo, num_clients, join_ratio):
    """返回累计通信量（累计参数传输数量），覆盖去中心化对比中的各算法。"""
    acc_dict = data.get("acc", [])
    n_rounds = len(acc_dict)

    if algo == "dfedset":
        num_tr = data.get("num_triggered", [])
        per_round = (
            np.array(num_tr[:n_rounds]) * BODY_PARAMS + num_clients * DFEDSET_EXTRA
        )
    elif algo == "efhc":
        num_tr = data.get("num_triggered", [])
        per_round = np.array(num_tr[:n_rounds]) * MODEL_PARAMS
    elif algo == "local":
        per_round = np.zeros(n_rounds)
    elif algo == "l2c":
        per_round = np.full(n_rounds, num_clients * 2 * MODEL_PARAMS)
    elif algo == "dispfl":
        per_round = np.full(n_rounds, int(num_clients * MODEL_PARAMS * 0.5))
    elif algo == "dfedpgp":
        per_round = np.full(n_rounds, int(num_clients * BODY_PARAMS))
    elif algo == "fedproto":
        per_round = np.full(n_rounds, int(num_clients * PROTOS_PARAMS))
    elif algo == "pearfl":
        per_round = np.full(n_rounds, int(num_clients * (MODEL_PARAMS + PROTOS_PARAMS)))
    else:
        per_round = np.full(n_rounds, int(num_clients * join_ratio * MODEL_PARAMS))
    return np.cumsum(per_round)
