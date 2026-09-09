"""训练结果命名的唯一纯函数实现。"""

import os


def fmt_value(value):
    """格式化命名值：整数浮点数不保留 ``.0``。"""
    if isinstance(value, float) and value == int(value):
        return str(int(value))
    return str(value)


def _sanitize(value):
    return str(value).replace(",", "_").replace(" ", "").replace("/", "_")


def build_part_seg(args):
    part_seg = f"{args.partition}_{fmt_value(args.num_clients)}"
    if args.partition == "dirichlet":
        part_seg += f"_{fmt_value(args.alpha)}"
    elif args.partition == "pathological":
        part_seg += f"_{fmt_value(args.n_class)}"
    return part_seg


def build_result_folder(args):
    return os.path.join(
        args.algo,
        f"{args.dataset}_{args.model}_{build_part_seg(args)}",
    )


def build_common_name(args):
    parts = [
        fmt_value(args.epochs),
        fmt_value(args.batch_size),
        fmt_value(args.lr),
        fmt_value(args.momentum),
        fmt_value(args.weight_decay),
    ]
    if args.ssl != "none":
        parts.extend(
            [
                args.ssl,
                fmt_value(args.unlabeled_ratio),
                fmt_value(args.label_ratio),
            ]
        )
        if args.ssl == "sfd":
            parts.extend([_sanitize(args.label_domain), _sanitize(args.unlabel_domain)])
    elif args.fdg:
        parts.extend([_sanitize(args.selected_domains), _sanitize(args.target_domain)])
    return "_".join(parts)
