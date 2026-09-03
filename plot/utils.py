import numbers
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch

import src
from src.naming import build_common_name, build_result_folder

os.makedirs("figures", exist_ok=True)


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


class ResultLoader:
    def __init__(self, base_dir="results"):
        self.base_dir = base_dir

    def average_recursive(self, data_list):
        if not data_list:
            return None
        first = data_list[0]
        if all(isinstance(data, numbers.Number) for data in data_list):
            return float(np.mean(data_list))
        if isinstance(first, list):
            min_len = min(len(d) for d in data_list)
            return [
                self.average_recursive([data[i] for data in data_list])
                for i in range(min_len)
            ]
        elif isinstance(first, dict):
            res = {}
            keys = set().union(
                *(data.keys() for data in data_list if isinstance(data, dict))
            )
            for key in keys:
                sub_list = [
                    d[key] for d in data_list if isinstance(d, dict) and key in d
                ]
                if sub_list:
                    res[key] = self.average_recursive(sub_list)
            return res
        return first

    def build_naming_args(self, algo, kwargs):
        values = dict(kwargs)
        values["algo"] = algo
        values.setdefault("model", "cnn")
        values.setdefault("ssl", "none")
        values.setdefault("fdg", False)
        required = [
            "dataset",
            "model",
            "partition",
            "num_clients",
            "epochs",
            "batch_size",
            "lr",
        ]
        if values["partition"] == "dirichlet":
            required.append("alpha")
        elif values["partition"] == "pathological":
            required.append("n_class")
        if values["ssl"] != "none":
            required.extend(["unlabeled_ratio", "label_ratio", "lam", "confidence"])
            if values["ssl"] == "sfd":
                required.extend(["label_domain", "unlabel_domain"])
        elif values["fdg"]:
            required.extend(["selected_domains", "target_domain"])
        missing = [key for key in required if key not in values]
        if missing:
            raise ValueError(f"缺少结果命名参数: {', '.join(missing)}")
        args = SimpleNamespace(**values)
        args.common_name = build_common_name(args)
        return args

    def resolve_folder_path(self, args, ablate_name):
        folder_path = os.path.join(self.base_dir, build_result_folder(args))
        if ablate_name:
            folder_path = os.path.join(folder_path, ablate_name)
        return folder_path

    def resolve_file_name(self, args):
        try:
            _, get_path = src.load_algorithm(args.algo)
            args.log_path = ""
            args.cur_time = 0
            get_path(args)
        except (AttributeError, KeyError) as exc:
            raise ValueError(f"算法 {args.algo} 无法生成结果文件名，缺少参数") from exc
        file_name = getattr(args, "file_name", "")
        if not file_name:
            raise ValueError(f"算法 {args.algo} 未生成有效结果文件名")
        return file_name

    def resolve_result_path(self, algo, dataset, partition, num_clients, ablate_name=None, **kwargs):
        """解析某算法结果的目录与文件名前缀（复用算法自身的 get_path 命名逻辑）。"""
        args = self.build_naming_args(
            algo,
            {
                **kwargs,
                "dataset": dataset,
                "partition": partition,
                "num_clients": num_clients,
            },
        )
        folder_path = self.resolve_folder_path(args, ablate_name)
        file_name = self.resolve_file_name(args)
        return folder_path, file_name

    def run_file_path(self, folder_path, file_name, run):
        """第 run 次 run 的结果文件完整路径。"""
        return os.path.join(folder_path, f"{file_name}_{run}.pt")

    def load_metrics_file(self, file_path, keys=None):
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"结果文件不存在: {file_path}")
        try:
            data = torch.load(file_path, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise ValueError(f"结果文件损坏: {file_path}") from exc
        if keys is None:
            return data
        return {key: data[key] for key in keys if key in data}

    def load_runs(
        self,
        algo,
        dataset,
        partition,
        num_clients,
        runs,
        ablate_name=None,
        keys=None,
        **kwargs,
    ):
        """严格直读第 0..runs-1 次 run 的结果文件；任一缺失立即抛 FileNotFoundError。"""
        folder_path, file_name = self.resolve_result_path(
            algo, dataset, partition, num_clients, ablate_name, **kwargs
        )
        return [
            self.load_metrics_file(self.run_file_path(folder_path, file_name, run), keys)
            for run in range(runs)
        ]

    def load(
        self,
        algo,
        dataset,
        partition,
        num_clients,
        specific_run=None,
        runs=None,
        ablate_name=None,
        keys=None,
        **kwargs,
    ):
        """读取结果：specific_run=k 读单次；runs=N 读第 0..N-1 次并求均值。

        两者必须显式二选一，文件缺失或损坏直接抛错，不做任何静默兜底。
        """
        if (specific_run is None) == (runs is None):
            raise ValueError(
                "必须显式指定 specific_run=k（单次 run）或 runs=N（0..N-1 求均值）之一"
            )
        folder_path, file_name = self.resolve_result_path(
            algo, dataset, partition, num_clients, ablate_name, **kwargs
        )
        if specific_run is not None:
            return self.load_metrics_file(
                self.run_file_path(folder_path, file_name, specific_run), keys
            )
        data_list = [
            self.load_metrics_file(self.run_file_path(folder_path, file_name, run), keys)
            for run in range(runs)
        ]
        keys_to_average = set().union(*(data.keys() for data in data_list))
        return {
            key: self.average_recursive(
                [data[key] for data in data_list if key in data]
            )
            for key in keys_to_average
        }

    def load_file(self, metrics_path, keys=None):
        if metrics_path.endswith("_params.pt"):
            raise ValueError("load_file 只接受指标文件，不能加载 _params.pt")
        if not metrics_path.endswith(".pt"):
            raise ValueError("load_file 只接受 .pt 指标文件")
        if not os.path.isfile(metrics_path):
            raise FileNotFoundError(metrics_path)
        return self.load_metrics_file(metrics_path, keys)


def plot_results(
    results_dict, x_lim, metric="acc", title=None, xlabel="Rounds", ylabel="Accuracy"
):
    plt.figure(figsize=(12, 7))
    summary = []

    for label, data in results_dict.items():
        if data is None:
            continue
        metric_data = data.get(metric)
        if metric_data is None:
            continue

        if isinstance(metric_data, list):
            y = metric_data[0:x_lim]
            max_val = max(y)
            last_10_avg = np.mean(y[-10:]) if len(y) >= 10 else np.mean(y)
            plt.plot(
                y, label=f"{label} (Max: {max_val:.4f}, Last10: {last_10_avg:.4f})"
            )
            summary.append({"Algorithm": label, "Max": max_val, "Last10": last_10_avg})
        elif isinstance(metric_data, dict):
            if "model" in metric_data or "local" in metric_data:
                k = "model" if "model" in metric_data else "local"
                y = metric_data[k][0:x_lim]
                max_val = max(y)
                last_10_avg = np.mean(y[-10:]) if len(y) >= 10 else np.mean(y)
                display_label = beautify_label(label)
                plt.plot(
                    y,
                    label=f"{display_label}-Model (Max: {max_val:.4f}, Last10: {last_10_avg:.4f})",
                )
                summary.append(
                    {
                        "Algorithm": f"{display_label}-Model",
                        "Max": max_val,
                        "Last10": last_10_avg,
                    }
                )

            pk = (
                "proto"
                if "proto" in metric_data
                else "global"
                if "global" in metric_data
                else None
            )
            if pk:
                y = metric_data[pk][0:x_lim]
                max_val = max(y)
                last_10_avg = np.mean(y[-10:]) if len(y) >= 10 else np.mean(y)
                display_label = beautify_label(label)
                suffix = "Proto" if pk == "proto" else "Global"
                plt.plot(
                    y,
                    linestyle="--",
                    alpha=0.8,
                    label=f"{display_label}-{suffix} (Max: {max_val:.4f}, Last10: {last_10_avg:.4f})",
                )
                summary.append(
                    {
                        "Algorithm": f"{display_label}-{suffix}",
                        "Max": max_val,
                        "Last10": last_10_avg,
                    }
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
    header = f"{display_label_name:<20} | {'Model Max':>12} | {'Model Last10':>12} | {'Proto Max':>12} | {'Proto Last10':>12}"
    divider = "-" * len(header)
    print(divider)
    print(header)
    print(divider)

    for label, data in results_dict.items():
        if data is None or metric not in data:
            continue
        metric_data = data[metric]

        m_max, m_last10 = 0.0, 0.0
        p_max, p_last10 = 0.0, 0.0

        if isinstance(metric_data, list):
            y = np.array(metric_data[0:x_lim])
            if len(y) > 0:
                m_max = np.max(y)
                m_last10 = np.mean(y[-10:]) if len(y) >= 10 else np.mean(y)
        elif isinstance(metric_data, dict):
            if "model" in metric_data or "local" in metric_data:
                mk = "model" if "model" in metric_data else "local"
                y = np.array(metric_data[mk][0:x_lim])
                if len(y) > 0:
                    m_max = np.max(y)
                    m_last10 = np.mean(y[-10:]) if len(y) >= 10 else np.mean(y)

            pk = (
                "proto"
                if "proto" in metric_data
                else "prototype"
                if "prototype" in metric_data
                else "global"
                if "global" in metric_data
                else None
            )
            if pk:
                y = np.array(metric_data[pk][0:x_lim])
                if len(y) > 0:
                    p_max = np.max(y)
                    p_last10 = np.mean(y[-10:]) if len(y) >= 10 else np.mean(y)

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
        if isinstance(metric_data, list):
            # List data is treated as Model Accuracy
            y = metric_data[0:x_lim]
            max_val = max(y)
            ax1.plot(y, label=f"{display_label} (Max: {max_val:.4f})")
        elif isinstance(metric_data, dict):
            # Model Acc on Ax1 (Left)
            mk = (
                "model"
                if "model" in metric_data
                else "local"
                if "local" in metric_data
                else None
            )
            if mk:
                y = metric_data[mk][0:x_lim]
                max_val = max(y)
                ax1.plot(y, label=f"{display_label} (Max: {max_val:.4f})")

            # Proto Acc on Ax2 (Right)
            pk = (
                "proto"
                if "proto" in metric_data
                else "global"
                if "global" in metric_data
                else None
            )
            if pk:
                y = metric_data[pk][0:x_lim]
                max_val = max(y)
                ax2.plot(y, label=f"{display_label} (Max: {max_val:.4f})")

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
        if isinstance(loss_data, list):
            y = loss_data
            plt.plot(y, label=f"{display_label} Loss")
        elif isinstance(loss_data, dict):
            if "model" in loss_data:
                y = loss_data["model"]
                plt.plot(y, label=f"{display_label}-Model Loss")
            if "proto" in loss_data:
                y = loss_data["proto"]
                plt.plot(
                    y, linestyle="--", alpha=0.8, label=f"{display_label}-Proto Loss"
                )
            if "aux" in loss_data:
                aux = loss_data["aux"]
                if "model_m" in aux:
                    y = aux["model_m"]
                    plt.plot(
                        y, linestyle=":", alpha=0.6, label=f"{display_label}-CE Loss"
                    )
                if "model_p" in aux:
                    y = aux["model_p"]
                    plt.plot(
                        y, linestyle=":", alpha=0.6, label=f"{display_label}-Align Loss"
                    )
            if "server_tgp" in loss_data:
                y = loss_data["server_tgp"]
                plt.plot(
                    y, linestyle="--", alpha=0.9, label=f"{display_label}-TGP Total"
                )
            if "server_tgp_ce" in loss_data:
                y = loss_data["server_tgp_ce"]
                plt.plot(y, linestyle=":", alpha=0.7, label=f"{display_label}-TGP CE")
            if "server_tgp_mse" in loss_data:
                y = loss_data["server_tgp_mse"]
                plt.plot(y, linestyle=":", alpha=0.7, label=f"{display_label}-TGP MSE")

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
    plt.savefig(os.path.join("figures", save_name), dpi=300)
    print(f"Figure saved to figures/{save_name}")
    plt.show()


def plot_sfd_results(results_dict, x_lim, title=None, show_proto=True):
    """
    SFD 场景三面板对比图：Overall / Source (Labeled) / Target (Unlabeled)。

    show_proto=True 时：Overall 面板叠加 acc_proto 虚线，Target 面板叠加
    source_proto_on_target 虚线。某算法缺少对应 key 时该线不画。
    三子图 xlim 在 savefig 之前设置，保证落盘图片与屏幕显示一致。
    """
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    ax_overall, ax_source, ax_target = axes
    panels = [
        (ax_overall, "Overall", "acc"),
        (ax_source, "Source (Labeled)", "acc_source"),
        (ax_target, "Target (Unlabeled)", "acc_target"),
    ]
    summary = []

    for label, data in results_dict.items():
        if data is None:
            continue
        display_label = beautify_label(label)
        row = {"Algorithm": display_label}
        for ax, panel_name, key in panels:
            series = data.get(key)
            if series is None:
                continue
            y = series[0:x_lim]
            if len(y) == 0:
                continue
            (line,) = ax.plot(y, label=display_label)
            row[panel_name] = (
                max(y),
                float(np.mean(y[-10:])) if len(y) >= 10 else float(np.mean(y)),
            )
            # Target 面板叠加源原型在目标域上的表现
            if ax is ax_target and show_proto and data.get("source_proto_on_target"):
                py = data["source_proto_on_target"][0:x_lim]
                ax.plot(
                    py,
                    linestyle="--",
                    alpha=0.8,
                    color=line.get_color(),
                    label=f"{display_label} SrcProto",
                )
        # Overall 面板叠加全局原型准确率
        if show_proto and data.get("acc_proto"):
            py = data["acc_proto"][0:x_lim]
            ax_overall.plot(
                py,
                linestyle="--",
                alpha=0.8,
                label=f"{display_label} Proto",
            )
        summary.append(row)

    for ax, panel_name, _ in panels:
        ax.set_title(panel_name, fontsize=13, fontweight="bold")
        ax.set_xlabel("Rounds")
        ax.set_ylabel("Accuracy (%)")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, x_lim)
        ax.legend(loc="lower right", fontsize=11)

    fig.suptitle(title or "SFD Comparison", fontsize=15)
    plt.tight_layout()

    save_base = (
        (title or "sfd_comparison")
        .lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("->", "to")
        .replace("__", "_")
    )
    save_name = f"sfd_{save_base}.png"
    plt.savefig(os.path.join("figures", save_name), dpi=300)
    print(f"Figure saved to figures/{save_name}")

    print("Summary of SFD Accuracy:")
    header = f"{'Algorithm':<22} | {'O-Max':>8} | {'O-Last10':>8} | {'S-Max':>8} | {'S-Last10':>8} | {'T-Max':>8} | {'T-Last10':>8}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for row in summary:
        cells = [f"{row['Algorithm']:<22}"]
        for panel in ("Overall", "Source (Labeled)", "Target (Unlabeled)"):
            if panel in row:
                cells.append(f"{row[panel][0]:>8.4f} | {row[panel][1]:>8.4f}")
            else:
                cells.append(f"{'-':>8} | {'-':>8}")
        print(" | ".join(cells))
    print("-" * len(header))

    plt.show()


def load_plot(
    selected_group,
    experiments,
    common_args,
    loader,
    x_lim=1000,
    do_plot=True,
):
    """
    加载并绘制对比图，根据数据格式自动识别是单线还是 Model/Proto 分离。
    加载失败（文件缺失/命名参数缺失）直接抛错，不做静默跳过。
    """
    results = {}
    for label in selected_group:
        if label not in experiments:
            continue
        algo_name, kwargs = experiments[label]
        results[label] = loader.load(
            algo_name, **{**common_args, **kwargs}, specific_run=0
        )

    if not results:
        raise ValueError("没有任何实验被加载：selected_group 与 experiments 均为空")

    if not do_plot:
        print_summary_table(results, x_lim, metric="acc")
        return

    # 尝试检查第一个结果的 'acc' 类型来决定绘图函数
    first_res = next(iter(results.values()))
    if isinstance(first_res.get("acc"), dict):
        plot_results_split(
            results,
            x_lim,
            metric="acc",
            title=f"Test Accuracy on {common_args['dataset']} ({common_args['partition']})",
        )
    else:
        plot_results(
            results,
            x_lim,
            metric="acc",
            title=f"Test Accuracy on {common_args['dataset']} ({common_args['partition']})",
        )


def load_plot_all_runs(selected_group, experiments, common_args, loader, x_lim=1000, *, runs):
    """
    加载并统计多轮实验的均值和标准差。runs 必填：直读第 0..runs-1 次结果。
    """
    print("Summary of ALL RUNS (Mean ± Std):")
    header = f"{'Algorithm':<25} | {'Max Acc':>20} | {'Last 10 Avg':>20}"
    print("-" * 75)
    print(header)
    print("-" * 75)

    for label in selected_group:
        if label not in experiments:
            continue
        algo_name, kwargs = experiments[label]
        merged_args = {**common_args, **kwargs}

        run_results = loader.load_runs(algo_name, runs=runs, **merged_args)

        if not run_results:
            continue

        sample_data = run_results[0]["acc"]

        if isinstance(sample_data, list):
            vals = []
            last_vals = []
            for run_data in run_results:
                y = run_data["acc"][0:x_lim]
                if not y:
                    continue
                vals.append(max(y))
                last_vals.append(np.mean(y[-10:]) if len(y) >= 10 else np.mean(y))

            if vals:
                mean_max, std_max = np.mean(vals), np.std(vals)
                mean_last, std_last = np.mean(last_vals), np.std(last_vals)
                print(
                    f"{label:<25} | {mean_max:>8.4f} ± {std_max:<7.4f} | {mean_last:>8.4f} ± {std_last:<7.4f} ({len(run_results)} runs)"
                )

        elif isinstance(sample_data, dict):
            for key in ["model", "local", "proto", "global"]:
                if key not in sample_data:
                    continue
                vals = []
                last_vals = []
                for run_data in run_results:
                    y = run_data["acc"][key][0:x_lim]
                    if not y:
                        continue
                    vals.append(max(y))
                    last_vals.append(np.mean(y[-10:]) if len(y) >= 10 else np.mean(y))

                if vals:
                    mean_max, std_max = np.mean(vals), np.std(vals)
                    mean_last, std_last = np.mean(last_vals), np.std(last_vals)
                    suffix = (
                        "Model"
                        if key in ["model", "local"]
                        else "Proto"
                        if key == "proto"
                        else "Global"
                    )
                    print(
                        f"{label + '-' + suffix:<25} | {mean_max:>8.4f} ± {std_max:<7.4f} | {mean_last:>8.4f} ± {std_last:<7.4f} ({len(run_results)} runs)"
                    )


def print_stats(selected_group, experiments, common_args, loader, x_lim=1000, *, runs):
    """
    加载并打印多轮实验的均值汇总（Model 和 Proto 分列显示）。
    runs 必填：直读第 0..runs-1 次结果，缺失即抛错。
    """
    print("Summary of Performance (Mean over runs):")
    # 表头：Algorithm | Model Max | Model Last | Proto Max | Proto Last
    header = f"{'Algorithm':<25} | {'M-Max':>10} | {'M-Last':>10} | {'P-Max':>10} | {'P-Last':>10}"
    print("-" * 85)
    print(header)
    print("-" * 85)

    for label in selected_group:
        if label not in experiments:
            continue
        algo_name, kwargs = experiments[label]
        merged_args = {**common_args, **kwargs}

        run_results = loader.load_runs(algo_name, runs=runs, **merged_args)

        if not run_results:
            continue

        # 初始化统计变量
        m_max, m_last, p_max, p_last = "-", "-", "-", "-"
        sample_data = run_results[0]["acc"]

        if isinstance(sample_data, list):
            vals, last_vals = [], []
            for run_data in run_results:
                y = run_data["acc"][0:x_lim]
                if not y:
                    continue
                vals.append(max(y))
                last_vals.append(np.mean(y[-10:]) if len(y) >= 10 else np.mean(y))
            if vals:
                m_max = f"{np.mean(vals):10.4f}"
                m_last = f"{np.mean(last_vals):10.4f}"

        elif isinstance(sample_data, dict):
            # Model / Local
            m_key = (
                "model"
                if "model" in sample_data
                else "local"
                if "local" in sample_data
                else None
            )
            if m_key:
                vals, last_vals = [], []
                for run_data in run_results:
                    y = run_data["acc"][m_key][0:x_lim]
                    if not y:
                        continue
                    vals.append(max(y))
                    last_vals.append(np.mean(y[-10:]) if len(y) >= 10 else np.mean(y))
                if vals:
                    m_max = f"{np.mean(vals):10.4f}"
                    m_last = f"{np.mean(last_vals):10.4f}"

            # Proto / Global
            p_key = (
                "proto"
                if "proto" in sample_data
                else "global"
                if "global" in sample_data
                else None
            )
            if p_key:
                vals, last_vals = [], []
                for run_data in run_results:
                    y = run_data["acc"][p_key][0:x_lim]
                    if not y:
                        continue
                    vals.append(max(y))
                    last_vals.append(np.mean(y[-10:]) if len(y) >= 10 else np.mean(y))
                if vals:
                    p_max = f"{np.mean(vals):10.4f}"
                    p_last = f"{np.mean(last_vals):10.4f}"

        print(
            f"{label:<25} | {m_max:>10} | {m_last:>10} | {p_max:>10} | {p_last:>10} ({len(run_results)} runs)"
        )


def freeze_cache_value(value):
    if isinstance(value, dict):
        return tuple(
            sorted((key, freeze_cache_value(item)) for key, item in value.items())
        )
    if isinstance(value, (list, tuple)):
        return tuple(freeze_cache_value(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted(freeze_cache_value(item) for item in value))
    return value


def batch_cached_load(loader, algo, specs, data_cache=None, keys=None, max_workers=8):
    """并行批量加载结果文件，兼容现有 data_cache。

    Args:
        loader: ResultLoader 实例
        algo: 算法名
        specs: list of kwargs dict（每个 dict 传给 loader.load）
        data_cache: 外部缓存 dict（可选），有则写入
        keys: 要提取的字段
        max_workers: 线程数
    Returns:
        list of (data or None)，顺序与 specs 一致
    """
    if data_cache is None:
        data_cache = {}

    cache_keys = []
    cached = [None] * len(specs)
    missing_idx = []
    missing_specs = []

    for i, kw in enumerate(specs):
        kt = freeze_cache_value(kw)
        kk = tuple(keys) if keys else None
        ck = (algo, kk, kt)
        cache_keys.append(ck)
        if ck in data_cache:
            cached[i] = data_cache[ck]
        else:
            missing_idx.append(i)
            missing_specs.append(kw)

    if not missing_specs:
        return cached

    def load_one(kw):
        return loader.load(algo, keys=keys, **kw)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        fut_map = {
            pool.submit(load_one, kw): i for i, kw in zip(missing_idx, missing_specs)
        }
        for fut in as_completed(fut_map):
            i = fut_map[fut]
            data = fut.result()
            cached[i] = data
            if data is not None:
                data_cache[cache_keys[i]] = data

    return cached
