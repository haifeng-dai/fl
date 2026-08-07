from copy import deepcopy

import numpy as np
import torch

from .common import (
    get_output_dir,
    hetero_split,
    is_fresh,
    per_domain_train_test_split,
    save_client_data,
)


def prepare_sfd_data(args, dataset_name, raw_data):
    """SFD：双域半监督场景（一个域有标签、一个域无标签）。

    流程概览：
      1. 参数校验 + 数据过滤（仅保留两域样本）
      2. 两域各自按 test_ratio 切分 train/test
      3. 每个域独立做异质切分，分配给所有客户端（每个客户端同时拿到两个域的数据）
      4. 内存掩码：label_domain 按 label_rate 保留有标签样本，其余丢弃；
         unlabel_domain 全部保留但标为无标签（is_labeled）
      5. 保存 client_*.pt（含 domains/is_labeled 供评估按域切分）
    """
    num_clients = args.num_clients
    test_ratio = args.test_ratio

    # ════════════════════════════════════════════════════════════════
    # 第一阶段：参数校验与数据过滤
    # ════════════════════════════════════════════════════════════════
    if not args.label_domain or not args.unlabel_domain:
        raise ValueError("SFD 场景必须指定 label_domain 与 unlabel_domain")
    if args.label_domain == args.unlabel_domain:
        raise ValueError(
            f"SFD 要求 label_domain != unlabel_domain，当前均为 '{args.label_domain}'"
        )
    selected = [args.label_domain, args.unlabel_domain]

    X, Y = raw_data["x"], raw_data["y"]
    all_domains = deepcopy(raw_data["domains"])
    num_classes = raw_data["num_classes"]

    # 校验域存在性：确保配置的域在原始数据集中真实存在
    raw_domains = set(all_domains)
    for d in selected:
        if d not in raw_domains:
            raise ValueError(
                f"SFD 域 '{d}' 不存在于数据集 domains={sorted(raw_domains)}"
            )

    # 从原始数据中过滤出仅属于两域的样本（raw 可能含其他域如 color_jitter 等）
    mask = np.isin(all_domains, selected)
    X, Y = X[mask], Y[mask]
    all_domains = [all_domains[i] for i in np.where(mask)[0]]
    if len(X) == 0:
        raise ValueError(f"SFD 域集合 {selected} 过滤后无剩余样本")

    # ════════════════════════════════════════════════════════════════
    # 第二阶段：缓存判定（已划分则跳过）
    # ════════════════════════════════════════════════════════════════
    output_dir = get_output_dir(args, dataset_name)

    if not is_fresh(output_dir, num_clients):
        print(
            f"-> SFD partition for {dataset_name} already exists at {output_dir}. Skipping."
        )
        return output_dir

    print(
        f"-> Partitioning SFD data (n={num_clients}, heterogeneous=True), "
        f"domains={sorted(set(all_domains))}..."
    )

    # ════════════════════════════════════════════════════════════════
    # 第三阶段：域切分 + 异质分配
    #   - per_domain_train_test_split：每个域内部按 test_ratio 切分 train/test
    #   - hetero_split：每个域独立按类异质分布（dirichlet/iid/pathological）
    #     分配给所有客户端，确保每个客户端同时拿到两个域的数据
    # ════════════════════════════════════════════════════════════════
    tr_idx_by_domain, te_idx_by_domain = per_domain_train_test_split(
        all_domains, test_ratio
    )

    # label_domain：有标签域，其训练样本后续会被 label_rate 掩码
    lbl_tr = hetero_split(
        np.array(tr_idx_by_domain[args.label_domain]),
        num_clients,
        args.partition,
        args.alpha,
        Y,
        args.n_class,
    )
    lbl_te = hetero_split(
        np.array(te_idx_by_domain[args.label_domain]),
        num_clients,
        args.partition,
        args.alpha,
        Y,
        args.n_class,
    )
    # unlabel_domain：无标签域，全部保留但标记为无标签
    unlbl_tr = hetero_split(
        np.array(tr_idx_by_domain[args.unlabel_domain]),
        num_clients,
        args.partition,
        args.alpha,
        Y,
        args.n_class,
    )
    unlbl_te = hetero_split(
        np.array(te_idx_by_domain[args.unlabel_domain]),
        num_clients,
        args.partition,
        args.alpha,
        Y,
        args.n_class,
    )

    # 拼接：每个客户端的 train/test = label 域切片 + unlabel 域切片
    cli_tr = [
        np.concatenate([lbl_tr[i], unlbl_tr[i]]).tolist() for i in range(num_clients)
    ]
    cli_te = [
        np.concatenate([lbl_te[i], unlbl_te[i]]).tolist() for i in range(num_clients)
    ]

    # ════════════════════════════════════════════════════════════════
    # 第四阶段：内存掩码（不二次读写文件）
    #   - label_domain 样本：按 label_rate 随机保留有标签样本，其余丢弃
    #   - unlabel_domain 样本：全部保留，标记为无标签
    # ════════════════════════════════════════════════════════════════
    train_is_labeled = []
    rng = torch.Generator().manual_seed(42)
    for i in range(num_clients):
        tr = np.array(cli_tr[i], dtype=int)
        arr = np.array([all_domains[idx] for idx in tr])  # 该客户端训练样本的域数组
        keep = np.zeros(len(tr), dtype=bool)
        is_labeled = torch.zeros(len(tr), dtype=torch.bool)

        # label_domain：按 label_rate 采样保留
        lbl = np.where(arr == args.label_domain)[0]
        n_keep = int(len(lbl) * args.label_rate)
        if len(lbl) > 0 and n_keep > 0:
            perm = torch.randperm(len(lbl), generator=rng).numpy()
            keep_idx = lbl[perm[:n_keep]]
            keep[keep_idx] = True
            is_labeled[keep_idx] = True
        # unlabel_domain：全部保留（不掩码），但 is_labeled 保持 False
        keep[arr == args.unlabel_domain] = True

        cli_tr[i] = tr[keep].tolist()
        train_is_labeled.append(is_labeled[keep])

    # ════════════════════════════════════════════════════════════════
    # 第五阶段：组装 extra_fields 并保存
    #   - test.domains：样本级域标签，供 load_data 合并测试集后按域切分评估
    #   - train.is_labeled：样本级标签掩码，供算法层区分有标签/无标签样本
    #   - train.domains：冗余（SFD 下 is_labeled=True 等价于 label_domain），
    #     不再保存以保持精简
    # ════════════════════════════════════════════════════════════════
    test_domains = [[all_domains[idx] for idx in cli_te[i]] for i in range(num_clients)]
    extra_fields = {
        "train": {"is_labeled": train_is_labeled},
        "test": {"domains": test_domains},
    }

    save_client_data(output_dir, X, Y, cli_tr, cli_te, num_classes, extra_fields)

    print(
        f"-> 成功为 {num_clients} 个客户端准备了 {dataset_name} (SFD partition @ {output_dir})。"
    )
    return output_dir
