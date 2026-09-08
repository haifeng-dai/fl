from copy import deepcopy

import numpy as np

from .common import (
    domain_as_client_partition,
    ensure_nonempty_client_indices,
    get_output_dir,
    is_fresh,
    per_domain_train_test_split,
    save_client_data,
)


def prepare_fdg_data(args, dataset_name, raw_data):
    """FDG（Federated Domain Generalization）：源域全部作训练，目标域整体作测试。

    - selected_domains：参与划分的域集合（源域），训练 100% 使用（test_ratio=0）。
    - target_domain：留一域整体作测试，不参与训练；样本均分打散到各客户端 test，
      评估时由算法合并全局测试集（全目标域样本）得到 acc_target。
    - 客户端数据不保存 domains 字段（FDG 训练侧无按域消费方，保持最精简）。
    """
    num_clients = args.num_clients
    rng = np.random.default_rng(args.seed)

    # ── 必要参数检查与处理 ──
    if not args.selected_domains:
        raise ValueError("FDG 场景必须指定 selected_domains（逗号分隔字符串）")
    # 字符串 -> 列表（兼容 "clean,rotate_cutout" 写法；用于避免被配置框架误判为参数扫描）
    selected = [d.strip() for d in args.selected_domains.split(",") if d.strip()]
    if not selected:
        raise ValueError("FDG 场景 selected_domains 解析后为空")
    td = args.target_domain
    if not td:
        raise ValueError("FDG 场景必须指定 target_domain（留一域作测试）")

    X, Y = raw_data["x"], raw_data["y"]
    all_domains = deepcopy(raw_data["domains"])  # 每个样本的域
    num_classes = raw_data["num_classes"]

    # 参与划分的域必须真实存在于数据集
    raw_domains = set(all_domains)
    for d in selected + [td]:
        if d not in raw_domains:
            raise ValueError(
                f"FDG 域 '{d}' 不存在于数据集 domains={sorted(raw_domains)}"
            )

    # 过滤出 selected_domains 的样本
    mask = np.isin(all_domains, selected)
    X, Y = X[mask], Y[mask]
    all_domains = [all_domains[i] for i in np.where(mask)[0]]
    if len(X) == 0:
        raise ValueError(f"selected_domains={selected} 过滤后无剩余样本")

    # 留一域作测试：目标域整体作测试（不进训练），源域全部作训练
    target_mask = np.array(all_domains) == td
    target_idx = np.where(target_mask)[0]
    src_mask = ~target_mask
    X_full, Y_full = X, Y  # 完整索引空间（selected 过滤后）
    X, Y = X[src_mask], Y[src_mask]
    src_domains = [all_domains[i] for i in np.where(src_mask)[0]]
    if len(X) == 0:
        raise ValueError(f"target_domain={td} 移除了所有训练样本")
    if len(target_idx) == 0:
        raise ValueError(f"target_domain={td} 无测试样本")

    output_dir = get_output_dir(args, dataset_name)

    if not is_fresh(output_dir, num_clients):
        print(
            f"-> FDG partition for {dataset_name} already exists at {output_dir}. Skipping."
        )
        return output_dir

    print(
        f"-> Partitioning FDG data (n={num_clients}), "
        f"source_domains={sorted(set(src_domains))}, target_domain={td}..."
    )

    # 源域全部作为训练集，不留源域测试集（test_ratio=0）
    tr_idx_by_domain, te_idx_by_domain = per_domain_train_test_split(
        src_domains, 0.0, rng
    )

    cli_tr, cli_te = domain_as_client_partition(
        tr_idx_by_domain,
        te_idx_by_domain,
        num_clients,
        Y=Y,
        heterogeneous=True,
        partition=args.partition,
        alpha=args.alpha,
        n_classes_per_client=args.n_class,
        rng=rng,
    )

    # 源域索引为源域数组空间，映射回完整索引空间（与目标域索引对齐）
    src_pos = np.where(src_mask)[0]
    for i in range(num_clients):
        cli_tr[i] = [int(src_pos[j]) for j in cli_tr[i]]

    # 目标域整体作测试：样本均分打散到各客户端 test（不参与训练）
    td_splits = np.array_split(target_idx, num_clients)
    for i in range(num_clients):
        cli_te[i] = cli_te[i] + td_splits[i].tolist()

    cli_tr = ensure_nonempty_client_indices(cli_tr, rng, "train")
    cli_te = ensure_nonempty_client_indices(cli_te, rng, "test")

    # FDG 客户端不保存 domains（train/test 仅含 x/y）
    save_client_data(
        output_dir, X_full, Y_full, cli_tr, cli_te, num_classes
    )

    print(
        f"-> 成功为 {num_clients} 个客户端准备了 {dataset_name} (FDG partition @ {output_dir})。"
    )
    return output_dir
