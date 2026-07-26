import os

import numpy as np
import torch

# ──────────────────────────────────────────────────────────────────────────
# SFD 标签掩码（维度三）—— 仅用于 SFD 场景，与 ssl / 类别划分完全无关
#   作为 prepare_domain_data(heterogeneous=True) 划分后的独立后处理，
#   本模块不导入也不依赖 ssl，与 ssl.py 之间零耦合
# ──────────────────────────────────────────────────────────────────────────
def apply_sfd(
    output_dir, num_clients, label_domain, unlabel_domain, label_rate, seed=42
):
    """SFD：一个域有标签、一个域无标签，且标签域样本按 label_rate 缩减。

    - unlabel_domain：该域训练样本整体标为无标签。
    - label_domain ：仅保留 label_rate 比例的样本作为有标签，其余丢弃
      （有标签数据应少于无标签数据，故标签域数据乘比例减少）。
    其余域（如存在）保持全有标签。
    """
    rng = torch.Generator().manual_seed(seed)
    for i in range(num_clients):
        path = os.path.join(output_dir, f"client_{i}.pt")
        data = torch.load(path, weights_only=False)
        train = data["train"]
        domains = train.get("domains")
        if domains is None:
            continue
        n = len(train["y"])
        arr = np.array(domains)
        is_labeled = torch.ones(n, dtype=torch.bool)

        # 无标签域：全部标为无标签
        is_labeled[arr == unlabel_domain] = False

        # 标签域：仅保留 label_rate 比例样本，丢弃其余
        lbl_idx = np.where(arr == label_domain)[0]
        if len(lbl_idx) > 0:
            perm = torch.randperm(len(lbl_idx), generator=rng).numpy()
            keep = lbl_idx[perm[: int(len(lbl_idx) * label_rate)]]
            # drop = lbl_idx[perm[int(len(lbl_idx) * label_rate) :]]
            keep_set = set(keep.tolist())
            mask_keep = np.array([j in keep_set for j in range(n)])
            train["x"] = train["x"][mask_keep]
            train["y"] = train["y"][mask_keep]
            train["domains"] = [domains[j] for j in range(n) if mask_keep[j]]
            is_labeled = is_labeled[mask_keep]

        train["is_labeled"] = is_labeled
        torch.save(data, path)
