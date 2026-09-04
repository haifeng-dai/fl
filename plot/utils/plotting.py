"""通用绘图与结果分析编排：读取实验结果并输出 figures/ 图片。"""

import os

import matplotlib.pyplot as plt
import numpy as np

PARAM_MAP = {
    "lamda_": r"$\lambda$",
    "lambda_": r"$\lambda$",
    "lamda": r"$\lambda$",
    "mu": r"$\mu$",
    "eta": r"$\eta$",
    "alpha": r"$\alpha$",
    "tau": r"$\tau$",
    "rho": r"$\rho$",
    "lr": "LR",
    "lr_alpha": r"$\alpha_{meta}$",
    "threshold": "Thres",
    "dense_ratio": "Density",
    "anneal_factor": "Anneal",
    "erk_power_scale": r"ERK Power",
    "lr_v": r"$LR_{head}$",
    "lambda_p": r"$\lambda_p$",
    "lambda_acl": r"$\lambda_{acl}$",
    "lambda_cos": r"$\lambda_{cos}$",
    "lambda_sa": r"$\lambda_{sa}$",
    "lambda_so": r"$\lambda_{so}$",
}


def beautify_label(name):
    """Converts code-style parameter names to LaTeX symbols or cleaner names."""
    if not name:
        return name

    # Check for exact matches in map
    if name in PARAM_MAP:
        return PARAM_MAP[name]

    # Check for param=value pattern
    for k, v in PARAM_MAP.items():
        if name.startswith(f"{k}="):
            return name.replace(f"{k}=", f"{v}=")

    return name


def plot_results(
    results_dict, x_lim, metric="acc", title=None, xlabel="Rounds", ylabel="Accuracy"
):
    plt.figure(figsize=(12, 7))
    summary = []
    optional = (("acc_proto", "Proto", "--"), ("acc_global", "Global", ":"))
    for label, data in results_dict.items():
        if not data:
            continue
        specs = [(metric, "", "-")]
        if metric == "acc":
            specs.extend(optional)
        for key, suffix, linestyle in specs:
            series = data.get(key)
            if not series:
                continue
            y = series[:x_lim]
            if not y:
                continue
            max_val = max(y)
            last_10_avg = (
                float(np.mean(y[-10:]))
                if len(y) >= 10
                else float(np.mean(y))
            )
            display = beautify_label(label) + (f"-{suffix}" if suffix else "")
            plt.plot(
                y,
                linestyle=linestyle,
                alpha=0.8,
                label=(
                    f"{display} (Max: {max_val:.4f}, "
                    f"Last10: {last_10_avg:.4f})"
                ),
            )
            summary.append(
                {"Algorithm": display, "Max": max_val, "Last10": last_10_avg}
            )

    plt.title(title or f"Comparison of {metric.upper()}")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc="lower right", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, x_lim)
    plt.tight_layout()

    save_name = (title or "comparison").lower().replace(" ", "_").replace(
        "(", ""
    ).replace(")", "") + ".png"
    os.makedirs("figures", exist_ok=True)
    plt.savefig(os.path.join("figures", save_name), dpi=300)
    print(f"Figure saved to figures/{save_name}")

    # Print summary table
    print(f"Summary of {metric.upper()}:")
    print("-" * 65)
    print(f"{'Algorithm':<20} | {'Max Acc':>15} | {'Last 10 Avg':>15}")
    print("-" * 65)
    for entry in summary:
        print(
            f"{entry['Algorithm']:<20} | {entry['Max']:>15.4f} | {entry['Last10']:>15.4f}"
        )
    print("-" * 65)

    plt.show()


def print_summary_table(results_dict, x_lim, metric="acc", label_name="Algorithm"):
    """Prints a consolidated summary table for model and prototype metrics."""
    print(f"Summary of {metric.upper()}:")

    display_label_name = beautify_label(label_name)
    header = (
        f"{display_label_name:<20} | {'Model Max':>12} | "
        f"{'Model Last10':>12} | {'Proto Max':>12} | {'Proto Last10':>12}"
    )
    divider = "-" * len(header)
    print(divider)
    print(header)
    print(divider)

    for label, data in results_dict.items():
        if data is None or metric not in data:
            continue

        series = data.get(metric)
        model_stats = _series_stats(series, x_lim) if series else None
        m_max, m_last10 = model_stats or (0.0, 0.0)
        proto_key = "acc_proto" if data.get("acc_proto") else "acc_global"
        proto_stats = (
            _series_stats(data.get(proto_key), x_lim)
            if metric == "acc" and data.get(proto_key)
            else None
        )
        p_max, p_last10 = proto_stats or (0.0, 0.0)

        # Clean up row label: if it's "param=value" and param matches label_name, just show "value"
        clean_row_label = label
        if "=" in label:
            param, val = label.split("=", 1)
            if param == label_name:
                clean_row_label = val

        print(
            f"{clean_row_label:<20} | {m_max:>12.4f} | {m_last10:>12.4f} | {p_max:>12.4f} | {p_last10:>12.4f}"
        )

    print(divider)


def plot_results_split(
    results_dict,
    x_lim,
    metric="acc",
    title=None,
    xlabel="Rounds",
    ylabel="Accuracy",
    label_name="Algorithm",
):
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    for label, data in results_dict.items():
        if data is None:
            continue
        metric_data = data.get(metric)
        if metric_data is None:
            continue

        display_label = beautify_label(label)
        y = metric_data[0:x_lim]
        if y:
            max_val = max(y)
            ax1.plot(y, label=f"{display_label} (Max: {max_val:.4f})")
        for key in ("acc_proto", "acc_global"):
            series = data.get(key)
            if series:
                y = series[0:x_lim]
                if y:
                    ax2.plot(
                        y,
                        label=(
                            f"{display_label}-{key[4:].title()} "
                            f"(Max: {max(y):.4f})"
                        ),
                    )

    ax1.set_title(f"Model {metric.upper()}: {title or ''}")
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.legend(loc="lower right", fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, x_lim)

    ax2.set_title(f"Prototype {metric.upper()}: {title or ''}")
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(ylabel)
    ax2.legend(loc="lower right", fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, x_lim)

    plt.tight_layout()

    save_base = (
        (title or "comparison")
        .lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("__", "_")
    )
    save_name = f"split_{save_base}.png"
    os.makedirs("figures", exist_ok=True)
    plt.savefig(os.path.join("figures", save_name), dpi=300)
    print(f"Figure saved to figures/{save_name}")

    # Use the extracted summary function
    print_summary_table(results_dict, x_lim, metric, label_name)

    plt.show()


def plot_loss(results_dict, title=None, xlabel="Rounds", ylabel="Loss", x_lim=None):
    plt.figure(figsize=(12, 7))
    for label, data in results_dict.items():
        if data is None:
            continue
        loss_data = data.get("loss")
        if loss_data is None:
            continue

        display_label = beautify_label(label)
        y = loss_data[:x_lim] if x_lim is not None else loss_data
        if y:
            plt.plot(y, label=f"{display_label} Loss")
        for key, suffix, linestyle, alpha in (
            ("loss_proto", "-Proto Loss", "--", 0.8),
            ("loss_global", "-Global Loss", ":", 0.8),
            ("loss_pln", "-PLN Loss", "--", 0.9),
            ("loss_pln_mse", "-PLN MSE", ":", 0.7),
            ("loss_pln_ortho", "-PLN Ortho", ":", 0.7),
        ):
            series = data.get(key)
            if series:
                optional_y = series[:x_lim] if x_lim is not None else series
                if optional_y:
                    plt.plot(
                        optional_y,
                        linestyle=linestyle,
                        alpha=alpha,
                        label=f"{display_label}{suffix}",
                    )

    plt.title(title or "Comparison of Loss")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc="lower right", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.yscale("log")
    if x_lim is not None:
        plt.xlim(0, x_lim)
    plt.tight_layout()

    save_name = (title or "loss_comparison").lower().replace(" ", "_").replace(
        "(", ""
    ).replace(")", "") + ".png"
    os.makedirs("figures", exist_ok=True)
    plt.savefig(os.path.join("figures", save_name), dpi=300)
    print(f"Figure saved to figures/{save_name}")
    plt.show()


def _load_selected(selected_group, experiments, common_args, loader):
    results = {}
    for label in selected_group:
        if label in experiments:
            algo_name, kwargs = experiments[label]
            results[label] = loader.load(
                algo_name, **{**common_args, **kwargs}, specific_run=0
            )
    if not results:
        raise ValueError("没有任何实验被加载：selected_group 与 experiments 均为空")
    return results


def load_plot(
    selected_group,
    experiments,
    common_args,
    loader,
    x_lim=1000,
    do_plot=True,
):
    results = _load_selected(selected_group, experiments, common_args, loader)
    title = f"Test Accuracy on {common_args['dataset']} ({common_args['partition']})"
    if do_plot:
        has_optional = any(
            data.get("acc_proto") or data.get("acc_global")
            for data in results.values()
            if data
        )
        plotter = plot_results_split if has_optional else plot_results
        plotter(results, x_lim, metric="acc", title=title)
    else:
        print_summary_table(results, x_lim, metric="acc")


def _series_stats(series, x_lim):
    y = series[:x_lim]
    if not y:
        return None
    return float(np.max(y)), float(np.mean(y[-10:]))


def load_plot_all_runs(
    selected_group, experiments, common_args, loader, x_lim=1000, *, runs
):
    print("Summary of ALL RUNS (Mean ± Std):")
    print(f"{'Algorithm':<25} | {'Max Acc':>20} | {'Last 10 Avg':>20}")
    print("-" * 75)
    for label in selected_group:
        if label not in experiments:
            continue
        algo_name, kwargs = experiments[label]
        run_results = loader.load_runs(
            algo_name, runs=runs, **{**common_args, **kwargs}
        )
        for key, suffix in (
            ("acc", ""),
            ("acc_proto", "-Proto"),
            ("acc_global", "-Global"),
        ):
            stats = [
                _series_stats(item[key], x_lim)
                for item in run_results
                if item.get(key)
            ]
            stats = [item for item in stats if item is not None]
            if stats:
                values = np.asarray(stats)
                print(
                    f"{label + suffix:<25} | "
                    f"{values[:, 0].mean():>8.4f} ± "
                    f"{values[:, 0].std():<7.4f} | "
                    f"{values[:, 1].mean():>8.4f} ± "
                    f"{values[:, 1].std():<7.4f} ({len(stats)} runs)"
                )


def print_stats(
    selected_group, experiments, common_args, loader, x_lim=1000, *, runs
):
    print("Summary of Performance (Mean over runs):")
    print(
        f"{'Algorithm':<25} | {'M-Max':>10} | {'M-Last':>10} | "
        f"{'Secondary-Max':>14} | {'Secondary-Last':>14}"
    )
    print("-" * 85)
    for label in selected_group:
        if label not in experiments:
            continue
        algo_name, kwargs = experiments[label]
        run_results = loader.load_runs(
            algo_name, runs=runs, **{**common_args, **kwargs}
        )
        model = [
            _series_stats(item["acc"], x_lim)
            for item in run_results
            if item.get("acc")
        ]
        secondary_key = (
            "acc_proto"
            if any(item.get("acc_proto") for item in run_results)
            else "acc_global"
        )
        secondary = [
            _series_stats(item[secondary_key], x_lim)
            for item in run_results
            if item.get(secondary_key)
        ]
        m = np.asarray([x for x in model if x is not None])
        p = np.asarray([x for x in secondary if x is not None])
        mvals = (m[:, 0].mean(), m[:, 1].mean()) if len(m) else ("-", "-")
        pvals = (p[:, 0].mean(), p[:, 1].mean()) if len(p) else ("-", "-")
        def format_value(value):
            if isinstance(value, (int, float, np.number)):
                return f"{value:>14.4f}"
            return f"{value:>14}"

        print(
            f"{label:<25} | {format_value(mvals[0])} | {format_value(mvals[1])} | "
            f"{format_value(pvals[0])} | {format_value(pvals[1])}"
        )
