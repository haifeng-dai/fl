import os
from copy import deepcopy

import numpy as np
import torch

from .common import (
    domain_as_client_partition,
    get_output_dir,
    per_domain_train_test_split,
    save_client_data,
)


def prepare_domain_data(args, dataset_name, raw_data, heterogeneous=False):
    """域划分：不同客户端分配不同域（DG / SFD 共用的底层函数，自身不感知场景语义）。

    - heterogeneous=False：仅按域顺序切分（纯 DG，不考虑类分布异质）。
    - heterogeneous=True ：在每个域内部再按类别做异质切分（SFD 的类异质部分）。

    selected_domains：参与划分的域集合（由调用方设置；SFD 下为 label/unlabel 两域）。
    target_domain：将该域整体留作 target_test，不参与训练划分（仅 DG 使用；SFD 置 None）。
    """
    num_clients = args.num_clients
    test_ratio = args.test_ratio

    X, Y = raw_data["x"], raw_data["y"]
    all_domains = deepcopy(raw_data["domains"])  # 每个样本的域
    num_classes = raw_data["num_classes"]

    # 参与划分的域集合与目标域由调用方决定：
    #   - DG：selected_domains（可空=全部域）/ target_domain（留一域作测试）
    #   - SFD：调用方已把 label/unlabel 两域写入 selected_domains
    # 本函数不感知 sfd/dg 语义，只做“按域隔离”的划分。
    selected = args.selected_domains
    target_domain = args.target_domain

    # 字符串 -> 列表（兼容 "clean,rotate_cutout" 写法；用于避免被配置框架误判为参数扫描）
    if isinstance(selected, str):
        selected = [d.strip() for d in selected.split(",") if d.strip()] or None

    # 过滤出 selected_domains 的样本
    if selected is not None:
        mask = np.isin(all_domains, selected)
        X, Y = X[mask], Y[mask]
        all_domains = [all_domains[i] for i in np.where(mask)[0]]
        if len(X) == 0:
            raise ValueError(f"selected_domains={selected} 过滤后无剩余样本")

    # 仅域泛化场景，选出测试域
    target_data = None
    if target_domain is not None:
        target_mask = np.array(all_domains) == target_domain  # 测试域
        source_mask = ~target_mask  # 训练域
        target_X, target_Y = X[target_mask], Y[target_mask]  # 测试样本
        X, Y = X[source_mask], Y[source_mask]  # 训练样本
        all_domains = [all_domains[i] for i in np.where(source_mask)[0]]  # 训练域
        target_data = {"x": target_X, "y": target_Y}  # 测试集
        if len(X) == 0:
            raise ValueError(f"target_domain={target_domain} 移除了所有训练样本")

    output_dir = get_output_dir(args, dataset_name)

    if os.path.exists(output_dir) and len(os.listdir(output_dir)) >= num_clients:
        print(
            f"-> Domain partition for {dataset_name} already exists at {output_dir}. Skipping."
        )
        return output_dir

    print(
        f"-> Partitioning domain data (n={num_clients}, heterogeneous={heterogeneous}), "
        f"domains={sorted(set(all_domains))}..."
    )

    tr_idx_by_domain, te_idx_by_domain = per_domain_train_test_split(
        all_domains, test_ratio
    )

    cli_tr, cli_te = domain_as_client_partition(
        tr_idx_by_domain,
        te_idx_by_domain,
        num_clients,
        Y=Y,
        heterogeneous=heterogeneous,
        partition=args.partition,
        alpha=args.alpha,
        n_classes_per_client=args.n_class,
    )

    # 样本级 domain 字段（供 load_data 与 SFD 掩码使用）
    train_domains = [
        [all_domains[idx] for idx in cli_tr[i]] for i in range(num_clients)
    ]
    test_domains = [[all_domains[idx] for idx in cli_te[i]] for i in range(num_clients)]
    extra_fields = {
        "train": {"domains": train_domains},
        "test": {"domains": test_domains},
    }

    save_client_data(output_dir, X, Y, cli_tr, cli_te, num_classes, extra_fields)

    # 把所有客户端的测试集合并起来，包括域信息
    # 在域泛化场景中，测试集包含源域的部分测试集
    all_test_x, all_test_y, all_test_domains = [], [], []
    for i in range(num_clients):
        all_test_x.append(X[cli_te[i]])
        all_test_y.append(Y[cli_te[i]])
        all_test_domains.extend([all_domains[idx] for idx in cli_te[i]])
    source_test = {
        "x": torch.cat(all_test_x, dim=0),
        "y": torch.cat(all_test_y, dim=0),
        "domains": all_test_domains,
    }
    torch.save(source_test, os.path.join(output_dir, "source_test.pt"))

    # 域泛化中目标域测试集，SFD 场景没有目标域（target_domain = None, target_data = None）
    if target_data is not None:
        torch.save(target_data, os.path.join(output_dir, "target_test.pt"))
        print(f"-> Target domain '{target_domain}' held out as target_test.pt")

    print(
        f"-> 成功为 {num_clients} 个客户端准备了 {dataset_name} (domain partition @ {output_dir})。"
    )
    return output_dir
