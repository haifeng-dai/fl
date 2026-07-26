import os

import numpy as np
import torch


# ──────────────────────────────────────────────────────────────────────────
# 客户端数据保存（统一接口）
# ──────────────────────────────────────────────────────────────────────────
def save_client_data(
    output_dir,
    X,
    Y,
    cli_tr_idx,
    cli_te_idx,
    num_classes,
    extra_fields=None,
):
    """统一保存每个客户端的 .pt 文件。

    extra_fields 形如 {"train": {"domains": [...]}, "test": {"domains": [...]}}，
    其中每个值是一个「按客户端排列」的列表；列表元素为该客户端对应样本级别的字段。
    """
    os.makedirs(output_dir, exist_ok=True)
    num_clients = len(cli_tr_idx)
    for i in range(num_clients):
        client_data = {
            "train": {"x": X[cli_tr_idx[i]], "y": Y[cli_tr_idx[i]]},
            "test": {"x": X[cli_te_idx[i]], "y": Y[cli_te_idx[i]]},
            "num_classes": num_classes,
        }
        if extra_fields:
            for phase in ("train", "test"):
                flds = extra_fields.get(phase)
                if flds:
                    for k, v in flds.items():
                        client_data[phase][k] = v[i]
        torch.save(client_data, os.path.join(output_dir, f"client_{i}.pt"))


# ──────────────────────────────────────────────────────────────────────────
# 目录命名
# ──────────────────────────────────────────────────────────────────────────
def partition_basename(partition, num_clients, alpha=0.5, n_classes=2):
    """仅返回划分短名（不含 dataset_name 前缀），供路径拼接复用。"""
    if partition == "iid":
        return f"iid_n{num_clients}"
    elif partition == "dirichlet":
        return f"dirichlet_n{num_clients}_a{alpha}"
    elif partition == "pathological":
        return f"pathological_n{num_clients}_c{n_classes}"
    else:
        raise ValueError(f"Unknown partition method: {partition}")


def sanitize(s):
    return s.replace(",", "_").replace(" ", "").replace("/", "_")


def get_output_dir(args, dataset_name):
    """数据划分输出目录（唯一真源），编码完整场景以避免缓存串味。"""
    n = args.num_clients
    if args.sfd:
        ld = args.label_domain
        ud = args.unlabel_domain
        lr = args.label_rate
        part_str = f"sfd_n{n}_a{args.alpha}_l{ld}_u{ud}_r{lr}"
    elif args.dg:
        dom = sanitize(args.selected_domains) if args.selected_domains else "all"
        td = sanitize(args.target_domain) if args.target_domain else "none"
        part_str = f"dg_n{n}_{dom}_t{td}"
    else:
        part_str = partition_basename(args.partition, n, args.alpha, args.n_class)
    return os.path.join("./datasets", dataset_name, part_str)


# ──────────────────────────────────────────────────────────────────────────
# 域划分基础工具
# ──────────────────────────────────────────────────────────────────────────
def per_domain_train_test_split(all_domains, test_ratio):
    """每个域内部按 test_ratio 切分训练/测试索引。"""
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


def hetero_split(idx, n_clients, partition, alpha, Y, n_classes_per_client=2):
    """域内核质：在 idx 所代表的单个域内，按类别做 dirichlet/iid/pathological 异质切分。"""
    if len(idx) == 0:
        return [np.array([], dtype=int) for _ in range(n_clients)]
    labels = Y[idx].numpy()
    uniq = np.unique(labels)  # 所有类
    client_idx = [[] for _ in range(n_clients)]

    if partition == "pathological":
        total_slots = n_clients * n_classes_per_client  # 总类槽位数
        shards_per_class = max(1, total_slots // len(uniq))  # 每个类切成几份碎片
        shards = []  # 收集所有碎片
        for c in uniq:
            c_idx = np.random.permutation(idx[labels == c])  # 特定类随机打乱
            for s in np.array_split(c_idx, shards_per_class):  # 某类样本切分
                shards.append(s)
        np.random.shuffle(shards)  # 打乱碎片
        for i in range(n_clients):
            for j in range(n_classes_per_client):
                shard = shards[i * n_classes_per_client + j]
                client_idx[i].append(shard)
    else:  # dirichlet / iid
        for c in uniq:
            c_idx = idx[labels == c]
            c_idx = np.random.permutation(c_idx)
            if partition == "iid":
                props = np.ones(n_clients) / n_clients  # 均匀比例
            else:
                props = np.random.dirichlet([alpha] * n_clients)  # 采样一个比例
            counts = np.cumsum(props) * len(c_idx)  # 每个客户端分配的样本数
            counts = counts.astype(int)[:-1]  # 转化成整数
            splits = np.split(c_idx, counts)  # 分割数据
            for i in range(n_clients):
                client_idx[i].append(splits[i])
    return [
        np.concatenate(c) if len(c) else np.array([], dtype=int) for c in client_idx
    ]


def domain_as_client_partition(
    domain_train_indices,  # 域训练集 idx
    domain_test_indices,  # 域测试集 idx
    num_clients,
    Y=None,  # 样本标签
    heterogeneous=False,
    partition="dirichlet",
    alpha=0.5,
    n_classes_per_client=2,
):
    """域泛化划分：不同客户端分配不同域。

    heterogeneous=True 时，在每个域内部再按类别做异质切分（SFD 场景：
    域偏移 + 类分布异质兼顾）；否则仅顺序切分（纯 DG，不考虑数据划分）。
    """
    domains = list(domain_train_indices.keys())  # 域名列表
    num_domains = len(domains)  # 域数量
    if num_domains == 0:
        return [[] for _ in range(num_clients)], [[] for _ in range(num_clients)]

    base = num_clients // num_domains  # 每个域客户端数量
    remainder = num_clients % num_domains  # 剩余域
    client_train = [[] for _ in range(num_clients)]
    client_test = [[] for _ in range(num_clients)]

    cursor = 0  # 当前正在填充数据的客户端索引起点
    for di, domain in enumerate(domains):
        n_for = base + (1 if di < remainder else 0)  # 每个域分配到的客户端数量
        tr_idx = np.array(domain_train_indices[domain])
        te_idx = np.array(domain_test_indices[domain])

        if heterogeneous and Y is not None and len(tr_idx) > 0:
            tr_splits = hetero_split(
                tr_idx, n_for, partition, alpha, Y, n_classes_per_client
            )
            te_splits = hetero_split(
                te_idx, n_for, partition, alpha, Y, n_classes_per_client
            )
        else:
            tr_splits = np.array_split(tr_idx, n_for)
            te_splits = np.array_split(te_idx, n_for)

        for j in range(n_for):  # 遍历分配数据
            cid = cursor + j
            client_train[cid] = tr_splits[j].tolist()
            client_test[cid] = te_splits[j].tolist()
        cursor += n_for

    return client_train, client_test  # 各客户端标签
