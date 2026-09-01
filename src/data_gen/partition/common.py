import os
import pickle

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
def sanitize(s):
    return s.replace(",", "_").replace(" ", "").replace("/", "_")


def resolve_n_class(args, num_classes):
    """n_class 自适应解析（唯一实现，幂等）。

    args.n_class == 0 表示自动根据数据集类别数与客户端数计算；
    用户显式指定 (>0) 时保持不变。解析结果写回 args，确保下游
    （划分目录 / is_fresh 检查 / get_pre_name 路径命名）口径一致。
    """
    if args.n_class == 0:
        args.n_class = max(2, -(-num_classes // args.num_clients))
    return args.n_class


def get_output_dir(args, dataset_name):
    """数据划分输出目录（唯一真源），编码完整场景以避免缓存串味。"""
    n = args.num_clients
    split_suffix = f"_{args.seed}_{args.test_ratio}"
    if args.partition == "iid":
        part_seg = f"iid_{n}"
    elif args.partition == "dirichlet":
        part_seg = f"dirichlet_{n}_{args.alpha}"
    elif args.partition == "pathological":
        part_seg = f"pathological_{n}_{args.n_class}"
    else:
        raise ValueError(f"Unknown partition method: {args.partition}")
    if args.ssl == "sfd":
        ld = sanitize(args.label_domain)
        ud = sanitize(args.unlabel_domain)
        part_str = f"sfd_{part_seg}_{ld}_{ud}_{args.label_ratio}{split_suffix}"
    elif args.fdg:
        dom = sanitize(args.selected_domains)
        td = sanitize(args.target_domain)
        part_str = f"fdg_{part_seg}_{dom}_{td}{split_suffix}"
    elif args.ssl in ("sample", "double", "client"):
        part_str = f"{args.ssl}_{part_seg}_{args.label_ratio}{split_suffix}"
    else:
        part_str = f"{part_seg}{split_suffix}"
    return os.path.join("./datasets", dataset_name, part_str)


def is_fresh(output_dir, num_clients):
    """划分是否已固化（即「非首次生成」）。

    目录已存在且客户端文件数充足，则视为划分已完成，应跳过重复处理；
    否则视为首次，需要执行划分与后处理。prepare_fdg_data / prepare_sfd_data
    与 prepare_data 总入口共用同一判定，保证「重复处理」逻辑单一来源、
    各场景后处理步骤口径完全一致，避免此前掩码被反复叠加执行、
    把数据逐级啃空的 bug。
    """
    if not os.path.isdir(output_dir):
        return True
    for i in range(num_clients):
        path = os.path.join(output_dir, f"client_{i}.pt")
        if not os.path.isfile(path):
            return True
        data = torch.load(path, weights_only=False)
        train = data["train"]
        test = data["test"]
        if len(train["x"]) == 0:
            return True
        if len(test["x"]) == 0:
            return True
    return False


def has_mixed_ssl_clients(output_dir, num_clients):
    """检查 output_dir 中的所有客户端数据是否都具有混合 SSL 所需的 L/U 掩码。

    仅当所有 client_*.pt 均满足以下条件时返回 True：
    1. 包含 train.is_labeled 字段；
    2. is_labeled 长度等于 train.y 长度；
    3. 至少包含一个 True（至少 1 个有标签样本）；
    4. 至少包含一个 False（至少 1 个无标签样本）。
    """
    if not os.path.isdir(output_dir):
        return False
    for i in range(num_clients):
        path = os.path.join(output_dir, f"client_{i}.pt")
        data = load_pt_or_none(path)
        if data is None:
            return False
        train = data.get("train", {})
        if "is_labeled" not in train:
            return False
        is_labeled = train["is_labeled"]
        y = train.get("y", [])
        if len(is_labeled) != len(y) or len(y) == 0:
            return False
        if not bool(is_labeled.any()) or not bool((~is_labeled).any()):
            return False
    return True


def load_pt_or_none(path):
    """读取缓存文件；缺失、空文件或损坏时返回 None（视为缓存无效）。

    只捕获预期的缓存损坏异常类型，避免吞掉编程错误。
    """
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return None
    try:
        return torch.load(path, weights_only=False)
    except (OSError, EOFError, RuntimeError, pickle.UnpicklingError):
        return None


# ──────────────────────────────────────────────────────────────────────────
# 客户端非空保底（sample / double 之外仍被 client 沿用）
# ──────────────────────────────────────────────────────────────────────────
def ensure_nonempty_client_indices(client_indices, rng, split_name):
    """仅在出现空客户端时转移一个既有索引。"""
    clients = [list(indices) for indices in client_indices]
    if not clients or not any(not indices for indices in clients):
        return clients
    total = sum(len(indices) for indices in clients)
    if total < len(clients):
        raise ValueError(
            f"{split_name} 样本总量不足，无法保证 {len(clients)} 个客户端均非空。"
        )
    while True:
        receivers = [i for i, indices in enumerate(clients) if not indices]
        if not receivers:
            return clients
        donor = max(range(len(clients)), key=lambda i: len(clients[i]))
        if len(clients[donor]) <= 1:
            raise RuntimeError(f"{split_name} 样本保底转移失败")
        receiver = receivers[0]
        position = int(rng.integers(len(clients[donor])))
        clients[receiver].append(clients[donor].pop(position))


# ──────────────────────────────────────────────────────────────────────────
# 域划分基础工具
# ──────────────────────────────────────────────────────────────────────────
def per_domain_train_test_split(all_domains, test_ratio, rng):
    """每个域内部按 test_ratio 切分训练/测试索引。"""
    unique_domains = sorted(set(all_domains))
    domain_train_indices = {}
    domain_test_indices = {}
    for domain in unique_domains:
        idx = np.where(np.array(all_domains) == domain)[0]
        rng.shuffle(idx)
        split = int(len(idx) * (1 - test_ratio))
        domain_train_indices[domain] = idx[:split].tolist()
        domain_test_indices[domain] = idx[split:].tolist()
    return domain_train_indices, domain_test_indices


def distribute_by_class(
    indices_by_class,
    num_clients,
    partition,
    rng,
    alpha=0.5,
    n_classes_per_client=2,
):
    """按类分组的索引列表，以 partition 策略分布到 num_clients 个客户端。

    这是 iid / dirichlet / pathological 三类异质分布的【唯一实现】，供类别划分
    （label.py 的 prepare_label_data）、域内核质（hetero_split）与混合 SSL
    （sample / double）共用，消除这些场景间的逻辑重复。

    indices_by_class: list[np.ndarray]，按类索引（顺序无关）。
    返回 client_idx: list[list[np.ndarray]]，client_idx[i] 为客户端 i 拿到的各分片，
        调用方负责用 np.concatenate 拼成最终索引。
    """
    if partition not in ("iid", "dirichlet", "pathological"):
        raise ValueError(f"未知分区方法: {partition}")

    pools = [np.asarray(class_indices, dtype=int) for class_indices in indices_by_class]
    client_idx = [[] for _ in range(num_clients)]
    num_classes = len(pools)

    if partition == "pathological":
        total_slots = num_clients * n_classes_per_client
        if total_slots < num_classes:
            raise ValueError(
                f"[Pathological Partition Error] 总需求分片数 ({total_slots}) "
                f"小于类别总数 ({num_classes})。"
            )
        shards_per_class = [total_slots // num_classes] * num_classes
        for i in range(total_slots % num_classes):
            shards_per_class[i] += 1

        shards = []
        for class_id, class_indices in enumerate(pools):
            required = shards_per_class[class_id]
            if len(class_indices) == 0:
                shards.extend([np.array([], dtype=int)] * required)
                continue
            if len(class_indices) < required:
                raise ValueError(
                    f"[Pathological Partition Error] 类别 {class_id} 仅有 "
                    f"{len(class_indices)} 个样本，不足以切成 {required} 个分片 "
                    f"（num_clients={num_clients}, n_classes_per_client="
                    f"{n_classes_per_client}）。"
                )
            shards.extend(np.array_split(class_indices, required))

        rng.shuffle(shards)
        for client_id in range(num_clients):
            start = client_id * n_classes_per_client
            end = start + n_classes_per_client
            client_idx[client_id].extend(shards[start:end])
        return client_idx

    for c, pool in enumerate(pools):
        if partition == "iid":
            splits = np.array_split(pool, num_clients)
        elif partition == "dirichlet":
            proportions = rng.dirichlet([alpha] * num_clients)
            counts = (np.cumsum(proportions) * len(pool)).astype(int)[:-1]
            splits = np.split(pool, counts)
        else:
            raise ValueError(f"未知分区方法: {partition}")
        for i in range(num_clients):
            client_idx[i].append(splits[i])
    return client_idx


def hetero_split(idx, n_clients, partition, alpha, Y, rng, n_classes_per_client=2):
    """域内核质：在 idx 所代表的单个域内，按类别做 dirichlet/iid/pathological 异质切分。

    仅负责把域子集 idx 按 Y 重新按类分组，实际分布逻辑全部委托给
    distribute_by_class（与类别划分共用同一份实现）。
    """
    if len(idx) == 0:
        return [np.array([], dtype=int) for _ in range(n_clients)]
    labels = Y[idx].numpy()
    uniq = np.unique(labels)  # 所有类
    indices_by_class = [idx[labels == c] for c in uniq]
    client_idx = distribute_by_class(
        indices_by_class, n_clients, partition, rng, alpha, n_classes_per_client
    )
    return [
        np.concatenate(c) if len(c) else np.array([], dtype=int) for c in client_idx
    ]


def domain_as_client_partition(
    domain_train_indices,  # 域训练集 idx
    domain_test_indices,  # 域测试集 idx
    num_clients,
    rng,
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
                tr_idx, n_for, partition, alpha, Y, rng, n_classes_per_client
            )
            te_splits = hetero_split(
                te_idx, n_for, partition, alpha, Y, rng, n_classes_per_client
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
