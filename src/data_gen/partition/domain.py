import os
from copy import deepcopy

import numpy as np
import torch


def get_domain_partition_dir(domain_partition, num_clients, alpha=0.5, domain_aware=True):
    if domain_partition == "domain_as_client":
        return f"domain_as_client_n{num_clients}"
    elif domain_partition == "domain_mixed":
        aware_str = "aware" if domain_aware else "blind"
        return f"domain_mixed_{aware_str}_n{num_clients}_a{alpha}"
    elif domain_partition == "domain_mixed_aware":
        return f"domain_mixed_aware_n{num_clients}_a{alpha}"
    elif domain_partition == "domain_mixed_blind":
        return f"domain_mixed_blind_n{num_clients}_a{alpha}"
    raise ValueError(f"Unknown domain partition method: {domain_partition}")


def per_domain_train_test_split(all_domains, test_ratio):
    unique_domains = sorted(set(all_domains))
    domain_train_indices = {}
    domain_test_indices = {}

    for domain in unique_domains:
        idx = np.where(np.array(all_domains) == domain)[0]
        np.random.shuffle(idx)
        split = int(len(idx) * (1 - test_ratio))
        domain_train_indices[domain] = idx[:split].tolist()
        domain_test_indices[domain] = idx[split:].tolist()

    return domain_train_indices, domain_test_indices


def domain_as_client_partition(domain_train_indices, domain_test_indices, num_clients):
    domains = list(domain_train_indices.keys())
    num_domains = len(domains)

    if num_domains == 0:
        return [[] for _ in range(num_clients)], [[] for _ in range(num_clients)]

    base = num_clients // num_domains
    remainder = num_clients % num_domains

    client_train = [[] for _ in range(num_clients)]
    client_test = [[] for _ in range(num_clients)]

    client_cursor = 0
    for di, domain in enumerate(domains):
        n_clients_for_this_domain = base + (1 if di < remainder else 0)
        tr_idx = np.array(domain_train_indices[domain])
        te_idx = np.array(domain_test_indices[domain])

        tr_splits = np.array_split(tr_idx, n_clients_for_this_domain)
        te_splits = np.array_split(te_idx, n_clients_for_this_domain)

        for j in range(n_clients_for_this_domain):
            cid = client_cursor + j
            client_train[cid] = tr_splits[j].tolist()
            client_test[cid] = te_splits[j].tolist()

        client_cursor += n_clients_for_this_domain

    return client_train, client_test


def domain_dirichlet_partition(
    domain_train_indices, domain_test_indices, num_clients, alpha=0.5, domain_aware=True
):
    domains = list(domain_train_indices.keys())
    num_domains = len(domains)
    client_train = [[] for _ in range(num_clients)]
    client_test = [[] for _ in range(num_clients)]

    proportions_list = (
        [np.random.dirichlet([alpha] * num_clients) for _ in range(num_domains)]
        if domain_aware
        else [np.random.dirichlet([alpha] * num_clients)] * num_domains
    )

    for di, domain in enumerate(domains):
        tr_idx = np.array(domain_train_indices[domain])
        te_idx = np.array(domain_test_indices[domain])
        proportions = proportions_list[di]

        tr_counts = (np.cumsum(proportions) * len(tr_idx)).astype(int)[:-1]
        te_counts = (np.cumsum(proportions) * len(te_idx)).astype(int)[:-1]

        tr_splits = np.split(tr_idx, tr_counts)
        te_splits = np.split(te_idx, te_counts)

        for i in range(num_clients):
            client_train[i].extend(tr_splits[i].tolist())
            client_test[i].extend(te_splits[i].tolist())

    return client_train, client_test


def prepare_domain_data(args, dataset_name, raw_data):
    partition_method = args.domain_partition
    domain_aware = getattr(args, "domain_aware", True)
    selected_domains = getattr(args, "selected_domains", None)
    target_domain = getattr(args, "target_domain", None)
    num_clients = args.num_clients
    test_ratio = args.test_ratio

    X, Y = raw_data["x"], raw_data["y"]
    all_domains = deepcopy(raw_data["domains"])
    num_classes = raw_data["num_classes"]

    if selected_domains is not None:
        mask = np.isin(all_domains, selected_domains)
        X, Y = X[mask], Y[mask]
        all_domains = [all_domains[i] for i in np.where(mask)[0]]
        if len(X) == 0:
            raise ValueError(f"selected_domains={selected_domains} 过滤后无剩余样本")

    target_data = None
    if target_domain is not None:
        target_mask = np.array(all_domains) == target_domain
        source_mask = ~target_mask
        target_X, target_Y = X[target_mask], Y[target_mask]
        X, Y = X[source_mask], Y[source_mask]
        all_domains = [all_domains[i] for i in np.where(source_mask)[0]]
        target_data = {"x": target_X, "y": target_Y}
        if len(X) == 0:
            raise ValueError(f"target_domain={target_domain} 移除了所有训练样本")

    tr_idx_by_domain, te_idx_by_domain = per_domain_train_test_split(
        all_domains, test_ratio
    )

    if partition_method == "domain_as_client":
        part_str = f"domain_as_client_n{num_clients}"
    elif partition_method == "domain_mixed":
        aware_str = "aware" if domain_aware else "blind"
        part_str = f"domain_mixed_{aware_str}_n{num_clients}_a{args.alpha}"
    else:
        raise ValueError(f"未知领域分区方法: {partition_method}")

    output_dir = os.path.join("./datasets", dataset_name, part_str)

    if os.path.exists(output_dir) and len(os.listdir(output_dir)) >= num_clients:
        print(
            f"-> Domain partition {part_str} for {dataset_name} already exists. Skipping."
        )
        return

    print(
        f"-> Partitioning domain data ({part_str}), domains={list(tr_idx_by_domain.keys())}..."
    )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cli_tr, cli_te = [], []
    if partition_method == "domain_as_client":
        cli_tr, cli_te = domain_as_client_partition(
            tr_idx_by_domain, te_idx_by_domain, num_clients
        )
    elif partition_method == "domain_mixed":
        cli_tr, cli_te = domain_dirichlet_partition(
            tr_idx_by_domain, te_idx_by_domain, num_clients, args.alpha, domain_aware
        )

    for i in range(num_clients):
        train_idx = cli_tr[i]
        test_idx = cli_te[i]
        client_data = {
            "train": {
                "x": X[train_idx],
                "y": Y[train_idx],
                "domains": [all_domains[idx] for idx in train_idx],
            },
            "test": {
                "x": X[test_idx],
                "y": Y[test_idx],
                "domains": [all_domains[idx] for idx in test_idx],
            },
            "num_classes": num_classes,
        }
        torch.save(client_data, os.path.join(output_dir, f"client_{i}.pt"))

    all_test_x, all_test_y, all_test_domains = [], [], []
    for i in range(num_clients):
        te_idx = cli_te[i]
        all_test_x.append(X[te_idx])
        all_test_y.append(Y[te_idx])
        all_test_domains.extend([all_domains[idx] for idx in te_idx])
    source_test = {
        "x": torch.cat(all_test_x, dim=0),
        "y": torch.cat(all_test_y, dim=0),
        "domains": all_test_domains,
    }
    torch.save(source_test, os.path.join(output_dir, "source_test.pt"))

    if target_data is not None:
        torch.save(target_data, os.path.join(output_dir, "target_test.pt"))

    print(f"-> 成功为 {num_clients} 个客户端准备了 {dataset_name} ({part_str})。")
    if target_domain:
        print(f"   Target domain '{target_domain}' held out for DG evaluation.")
