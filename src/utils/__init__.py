from .load_data import load_data
from .fed_utils import BaseClient, BaseServer, ClientInfo

def is_pfl(algo_name: str) -> bool:
    # --- 算法类型硬编码区分 ---
    if algo_name in ["fedavg", "moon"]:
        pfl = False
    elif algo_name in ["pfedavg", "fedproto"]:
        pfl = True
    else:
        raise ValueError(f"Unsupported algorithm: {algo_name}")
    return pfl