"""sample / double 两种类别型半监督场景的共享划分流程。

两者共用同一份基础 L/U/Test 索引（见 split_labeled_unlabeled_test_by_class），
唯一区别是客户端分配阶段：

- sample：L 与 U 使用相同分配随机流（同分布）；
- double：L 与 U 使用独立分配随机流（双异质）。

客户端各自保存自己的 ``L ∪ U`` 训练集与 ``is_labeled``，以及从基础 Test 池中
确定性均分得到、与其他客户端互斥的本地 ``test``。基础 Test 池被完整切分、不复制、
不遗漏。划分缓存是可再生产物：目录命名已编码全部决定参数，格式改变或文件损坏时
由使用者删除对应划分目录后重新生成，代码不维护历史格式兼容。
"""

import os

import numpy as np
import torch

from .common import (
    distribute_by_class,
    get_output_dir,
)

# 分配重试上限（原 MAX_TEMPLATE_ATTEMPTS，因不再有 template 概念而改名）。
MAX_DISTRIBUTION_ATTEMPTS = 32


def derive_rng(seed, attempt, stream):
    """由 seed / attempt / stream 确定性派生独立随机流。

    stream 固定为：
    - 0：L 分配结构；
    - 1：double 的 U 分配结构。

    sample 的 U 必须重新创建 ``derive_rng(seed, attempt, 0)``，不能复用已被 L
    消耗的同一个 RNG 对象；double 的 U 使用 stream 1。
    """
    return np.random.default_rng([int(seed), int(attempt), int(stream)])


def concat_parts(client_parts):
    return [
        np.concatenate(parts) if parts else np.array([], dtype=int)
        for parts in client_parts
    ]


# ──────────────────────────────────────────────────────────────────────────
# 基础 L / U / Test 三池（sample / double 共享的唯一划分接口）
# ──────────────────────────────────────────────────────────────────────────
def validate_three_pools(
    labeled_by_class,
    unlabeled_by_class,
    test_by_class,
    n_total,
):
    """校验三池索引严格互斥、完整且合法。

    要求：
    1. 三个外层列表非空且类别数量一致；
    2. 每个类别对应的 L/U/Test 数组是一维整数索引；
    3. 合并全部索引后 ``np.sort(merged)`` 严格等于 ``np.arange(n_total)``。

    该条件同时保证总数正确、无重复、无遗漏、无负索引、无越界索引。
    """
    pools = (labeled_by_class, unlabeled_by_class, test_by_class)
    if any(len(p) == 0 for p in pools):
        raise ValueError("[Mixed SSL Partition Error] 三池外层列表必须非空。")
    num_classes = len(labeled_by_class)
    if not all(len(p) == num_classes for p in pools):
        raise ValueError("[Mixed SSL Partition Error] 三池类别数量必须一致。")

    flat = []
    for pool in pools:
        for arr in pool:
            arr = np.asarray(arr)
            if arr.ndim != 1:
                raise ValueError("[Mixed SSL Partition Error] 三池索引必须是一维数组。")
            if not np.issubdtype(arr.dtype, np.integer):
                raise ValueError("[Mixed SSL Partition Error] 三池索引必须是整数类型。")
            flat.append(arr)
    merged = np.concatenate(flat)
    expected = np.arange(n_total, dtype=int)
    if len(merged) != n_total:
        raise ValueError(
            f"[Mixed SSL Partition Error] L/U/Test 合并索引数 ({len(merged)}) "
            f"与原始样本总数 ({n_total}) 不一致。"
        )
    if not np.array_equal(np.sort(merged), expected):
        raise ValueError(
            f"[Mixed SSL Partition Error] L/U/Test 索引必须严格等于 "
            f"np.arange({n_total})（无重复、无遗漏、无负索引、无越界）。"
        )


def split_labeled_unlabeled_test_by_class(targets, test_ratio, label_ratio, seed):
    """按类别生成互斥的 L / U / Test 三个数据池（唯一纯划分接口）。

    每个类别独立执行：

        n_test     = int(n_class * test_ratio)
        n_train    = n_class - n_test
        n_labeled  = int(n_train * label_ratio)
        n_unlabeled= n_train - n_labeled

    同一类别内的索引先由 ``seed`` 派生的 RNG 打乱，随后依次切出
    Test、L、U。三个池互斥、无放回、不复制样本，最后调用严格校验。
    """
    if not 0 < test_ratio < 1:
        raise ValueError(f"test_ratio 必须位于 (0, 1) 开区间，当前为 {test_ratio}")
    if not 0 < label_ratio < 1:
        raise ValueError(f"label_ratio 必须位于 (0, 1) 开区间，当前为 {label_ratio}")

    targets_np = targets.numpy()
    classes = np.unique(targets_np)
    num_classes = len(classes)
    rng = np.random.default_rng(seed)

    labeled_by_class = []
    unlabeled_by_class = []
    test_by_class = []
    for c in classes:
        idx = np.where(targets_np == c)[0]
        rng.shuffle(idx)
        n_test = int(len(idx) * test_ratio)
        n_train = len(idx) - n_test
        n_labeled = int(n_train * label_ratio)
        n_unlabeled = n_train - n_labeled
        if n_test <= 0 or n_labeled <= 0 or n_unlabeled <= 0:
            raise ValueError(
                f"[Mixed SSL Partition Error] 类别 {c} 无法生成非空的 L/U/Test："
                f"总样本 {len(idx)}，Test {n_test}，L {n_labeled}，U {n_unlabeled}。"
            )
        test_by_class.append(idx[:n_test])
        labeled_by_class.append(idx[n_test : n_test + n_labeled])
        unlabeled_by_class.append(idx[n_test + n_labeled :])

    validate_three_pools(
        labeled_by_class, unlabeled_by_class, test_by_class, len(targets_np)
    )
    return labeled_by_class, unlabeled_by_class, test_by_class, num_classes


def build_mixed_client_indices(
    labeled_by_class,
    unlabeled_by_class,
    num_clients,
    partition,
    alpha,
    n_class,
    seed,
    shared_distribution,
):
    """生成每个客户端的 L / U 索引。

    ``shared_distribution=True``（sample）L 与 U 使用由同一分配随机流（stream 0）
    派生的同一份分配结构（同分布）；``False``（double）两者使用不同 stream（0 与 1）
    派生的独立分配结构（双异质）。
    L/U 总量不足客户端数时立即报错；IID 下首次分配即失败也立即报错（更换 attempt 无效）。
    """
    n_labeled = sum(len(pool) for pool in labeled_by_class)
    n_unlabeled = sum(len(pool) for pool in unlabeled_by_class)
    if n_labeled < num_clients:
        raise ValueError(
            f"[Mixed SSL Partition Error] L 总量 ({n_labeled}) 少于客户端数 "
            f"({num_clients})，无法保证每客户端均有 L。"
        )
    if n_unlabeled < num_clients:
        raise ValueError(
            f"[Mixed SSL Partition Error] U 总量 ({n_unlabeled}) 少于客户端数 "
            f"({num_clients})，无法保证每客户端均有 U。"
        )

    last_missing = []
    attempt = 0
    for attempt in range(MAX_DISTRIBUTION_ATTEMPTS):
        # L 分配始终使用 stream 0
        rng_labeled = derive_rng(seed, attempt, 0)
        # sample 的 U 重新派生 stream 0（同分布）；double 的 U 使用 stream 1（双异质）
        rng_unlabeled = derive_rng(seed, attempt, 0 if shared_distribution else 1)

        client_labeled = concat_parts(
            distribute_by_class(
                labeled_by_class, num_clients, partition, rng_labeled, alpha, n_class
            )
        )
        client_unlabeled = concat_parts(
            distribute_by_class(
                unlabeled_by_class,
                num_clients,
                partition,
                rng_unlabeled,
                alpha,
                n_class,
            )
        )

        last_missing = [
            (i, len(labeled), len(unlabeled))
            for i, (labeled, unlabeled) in enumerate(
                zip(client_labeled, client_unlabeled)
            )
            if len(labeled) == 0 or len(unlabeled) == 0
        ]
        if not last_missing:
            return client_labeled, client_unlabeled

        # IID 没有随机分配结构，更换 attempt 不会改变结果，立即报错。
        if partition == "iid":
            break

    raise ValueError(
        f"[Mixed SSL Partition Error] 无法为 {num_clients} 个客户端生成同时包含 "
        f"L 与 U 的分配（partition={partition}, alpha={alpha}, n_class={n_class}，"
        f"已尝试 {attempt + 1} 次）。L 总量 {n_labeled}，U 总量 {n_unlabeled}，"
        f"最后一次缺少 L 或 U 的客户端 (客户端, L 数, U 数): {last_missing}"
    )


def assemble_client_train(client_labeled, client_unlabeled, X, Y):
    """按客户端合并 ``L ∪ U``，并严格同步生成 ``is_labeled``。

    训练 DataLoader 已负责随机采样，因此此处不做客户端内二次打乱：
    直接拼接 ``L + U``，同步生成 ``[True]*len(L) + [False]*len(U)``。
    """
    clients = []
    for labeled, unlabeled in zip(client_labeled, client_unlabeled):
        labeled = np.asarray(labeled, dtype=int)
        unlabeled = np.asarray(unlabeled, dtype=int)
        indices = np.concatenate((labeled, unlabeled))
        is_labeled = torch.cat(
            (
                torch.ones(len(labeled), dtype=torch.bool),
                torch.zeros(len(unlabeled), dtype=torch.bool),
            )
        )
        clients.append(
            {
                "x": X[indices],
                "y": Y[indices],
                "is_labeled": is_labeled,
            }
        )
    return clients


def check_conservation(client_indices, global_by_class, split_name):
    """客户端索引并集必须严格等于全局池（无重复、无遗漏）。"""
    expected = np.sort(np.concatenate(global_by_class))
    merged = np.sort(np.concatenate(client_indices))
    if len(merged) != len(expected) or not np.array_equal(merged, expected):
        raise ValueError(
            f"[Mixed SSL Partition Error] {split_name} 客户端分配不守恒："
            f"客户端合计 {len(merged)}，全局 {split_name} 合计 {len(expected)}。"
        )


def save_mixed_ssl_data(
    output_dir, clients, client_test_indices, X, Y, num_classes
):
    """保存每个客户端文件：train（L ∪ U + is_labeled）+ 互斥均分的 test。"""
    os.makedirs(output_dir, exist_ok=True)
    for i, (train, test_idx) in enumerate(zip(clients, client_test_indices)):
        torch.save(
            {
                "num_classes": num_classes,
                "train": train,
                "test": {
                    "x": X[test_idx],
                    "y": Y[test_idx],
                },
            },
            os.path.join(output_dir, f"client_{i}.pt"),
        )


def mixed_partition_exists(output_dir, num_clients):
    """仅按所需文件是否存在判断是否跳过生成。

    不 torch.load、不检查字段/长度/mask/类别数、不修改或删除文件。
    文件存在但人为损坏时，后续 torch.load 或字段访问自然报错，由使用者删目录重生成。
    """
    required_names = [f"client_{client_id}.pt" for client_id in range(num_clients)]
    return all(
        os.path.isfile(os.path.join(output_dir, name)) for name in required_names
    )


def prepare_mixed_ssl_data(args, dataset_name, raw_data):
    """sample / double 共用的划分与保存流程。模式由 args.ssl 唯一表达。

    sample：L 与 U 使用相同分配随机流（同分布）；
    double：L 与 U 使用不同分配随机流（双异质）。
    拒绝除 sample / double 之外的模式。
    """
    if args.ssl not in ("sample", "double"):
        raise ValueError(
            f"prepare_mixed_ssl_data 仅支持 sample / double，当前 ssl={args.ssl}"
        )
    shared_distribution = args.ssl == "sample"

    X, Y = raw_data["x"], raw_data["y"]
    (
        labeled_by_class,
        unlabeled_by_class,
        test_by_class,
        num_classes,
    ) = split_labeled_unlabeled_test_by_class(
        Y, args.test_ratio, args.label_ratio, args.seed
    )

    output_dir = get_output_dir(args, dataset_name)
    if mixed_partition_exists(output_dir, args.num_clients):
        print(
            f"-> Mixed SSL ({args.ssl}) partition already exists at {output_dir}. Skipping."
        )
        return output_dir

    print(f"-> Partitioning mixed SSL data ({os.path.basename(output_dir)})...")
    client_labeled, client_unlabeled = build_mixed_client_indices(
        labeled_by_class,
        unlabeled_by_class,
        args.num_clients,
        args.partition,
        args.alpha,
        args.n_class,
        args.seed,
        shared_distribution,
    )
    check_conservation(client_labeled, labeled_by_class, "L")
    check_conservation(client_unlabeled, unlabeled_by_class, "U")

    clients = assemble_client_train(client_labeled, client_unlabeled, X, Y)

    # 全局 Test 池在客户端间互斥均分（采用 IID 分配方式打散）
    test_rng = derive_rng(args.seed, 0, 2)
    client_test_by_class = distribute_by_class(
        test_by_class, args.num_clients, "iid", test_rng
    )
    client_test_indices = [
        np.concatenate(parts) if parts else np.array([], dtype=int)
        for parts in client_test_by_class
    ]
    check_conservation(client_test_indices, test_by_class, "Test")
    save_mixed_ssl_data(
        output_dir, clients, client_test_indices, X, Y, num_classes
    )

    print(
        f"-> 成功为 {args.num_clients} 个客户端准备了 "
        f"{dataset_name} ({args.ssl} SSL, {args.partition})。"
    )
    return output_dir
