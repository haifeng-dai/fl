import os

import torch

# ──────────────────────────────────────────────────────────────────────────
# 半监督标签掩码（ssl）—— 仅用于类别划分（维度二），与域/ SFD 完全无关
#   作为 prepare_label_data 划分后的独立后处理，本模块不感知任何域语义
# ──────────────────────────────────────────────────────────────────────────
def apply_label_ratio_sample(output_dir, num_clients, label_ratio, seed=42):
    """场景B：每客户端内按 label_ratio 随机掩码样本为无标签（标签/未标签同分布）。"""
    rng = torch.Generator().manual_seed(seed)
    for i in range(num_clients):
        path = os.path.join(output_dir, f"client_{i}.pt")
        data = torch.load(path, weights_only=False)
        train = data["train"]
        n = len(train["y"])
        perm = torch.randperm(n, generator=rng)
        num_labeled = max(1, int(n * label_ratio))
        is_labeled = torch.zeros(n, dtype=torch.bool)
        is_labeled[perm[:num_labeled]] = True
        train["is_labeled"] = is_labeled
        torch.save(data, path)


def apply_label_ratio_client(output_dir, num_clients, label_ratio, seed=42):
    """场景A：按 label_ratio 选取客户端为全标签，其余全无标签（同分布）。"""
    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(num_clients, generator=rng)
    n_labeled = max(1, int(num_clients * label_ratio))
    labeled_clients = set(perm[:n_labeled].tolist())
    for i in range(num_clients):
        path = os.path.join(output_dir, f"client_{i}.pt")
        data = torch.load(path, weights_only=False)
        train = data["train"]
        n = len(train["y"])
        flag = i in labeled_clients
        train["is_labeled"] = torch.full((n,), flag, dtype=torch.bool)
        torch.save(data, path)
