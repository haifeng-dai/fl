import os

import matplotlib.pyplot as plt
import numpy as np
import torch

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
    "lr_v": r"$LR_{head}$",
    "lambda_p": r"$\lambda_p$",
    "lambda_acl": r"$\lambda_{acl}$",
    "lambda_cos": r"$\lambda_{cos}$",
    "lambda_sa": r"$\lambda_{sa}$",
    "lambda_so": r"$\lambda_{so}$",
}


def get_adj_suffix(args):
    adj_type = args["adj_type"]
    suffix = f"{adj_type}"
    if adj_type == "random":
        suffix += f"_{args['edge_p']}"
    elif adj_type == "small_world":
        suffix += f"_{args['k_small_world']}_{args['edge_p']}"
    elif adj_type == "scale_free":
        suffix += f"_{args['m_scale_free']}"
    return suffix


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
        self.algo_patterns = {
            "fedala": lambda args: (
                f"_{args['eta']}_{args['rand_percent']}_{args['layer_idx']}_{args['ala_threshold']}_{args['num_pre_loss']}"
            ),
            "fedavg": lambda args: "",
            "feddyn": lambda args: f"_alpha{args['alpha_coef']}",
            "fedfm": lambda args: f"_{args['mu']}",
            "fedkd": lambda args: f"_{args['lr_g']}_{args['energy']}",
            "fedlsa": lambda args: (
                f"_{args['lambda_com']}_{args['alpha_sep']}_{args['server_epochs']}_{args['server_lr']}_{args['tau']}"
            ),
            "fedper": lambda args: "",
            "fedpln": lambda args: (
                f"_{args['lambda_']}_{args['epoch_pln']}_{args['lr_pln']}_{args['batch_size_pln']}_{args['depth_pln']}_{args['width_pln']}_{args['mode']}_{args['fixed_proto']}_{args['init_emb']}_{args['har']}"
            ),
            "fedproc": lambda args: "",
            "fedproto": lambda args: f"_{args['mu']}",
            "fedprox": lambda args: f"_{args['mu']}",
            "fedrep": lambda args: f"_{args['epochs_head']}",
            "fedsa": lambda args: (
                f"_{args['alpha_sa']}_{args['lambda_r']}_{args['lambda_mcl']}_{args['lambda_cc']}"
            ),
            "scaffold": lambda args: f"_glr{args['global_lr']}",
            "fedtgp": lambda args: (
                f"_{args['lamda_']}_{args['server_epochs']}_{args['server_lr']}_{args['margin_threshold']}"
            ),
            "feddpc": lambda args: (
                f"_{args['lamda_']}_{args['head_epochs']}_{args['body_epochs']}_{args['lr_head']}_{args['lr_body']}_{args['server_epochs']}_{args['server_lr']}_{args['lambda_p']}_{args['lambda_acl']}"
            ),
            "fml": lambda args: f"_{args['alpha_fml']}_{args['beta_fml']}",
            "lgfedavg": lambda args: "",
            "moon": lambda args: f"_{args['mu']}_{args['tau']}",
            "fedtest": lambda args: f"_ray_{args['use_ray']}",
            "local": lambda args: "",
            # Decentralized Algorithms
            "l2c": lambda args: (
                f"_{get_adj_suffix(args)}_{args['val_ratio']}_{args['lr_alpha']}_{args['prune_round']}_{args['prune_num']}"
            ),
            "dispfl": lambda args: (
                f"_{get_adj_suffix(args)}_{args['dense_ratio']}_{args['anneal_factor']}"
            ),
            "pearfl": lambda args: f"_{get_adj_suffix(args)}_{args['lamda']}",
            "dfedavgm": lambda args: f"_{get_adj_suffix(args)}",
            "dfedpgp": lambda args: (
                f"_{get_adj_suffix(args)}_{args['local_v_epochs']}_{args['lr_v']}_{args['momentum_v']}_{args['weight_decay_v']}"
            ),
            "proxyfl": lambda args: f"_{get_adj_suffix(args)}_{args['mu']}",
            "dfedset": lambda args: (
                f"_{get_adj_suffix(args)}_{args.get('lambda_sa', args.get('mu'))}_{args['eta']}_{args.get('lambda_so', args.get('lambda_cos'))}"
            ),
            "efhc": lambda args: (
                f"_{get_adj_suffix(args)}_r{args['event_r']}_bw{args['bandwidth_mean']}"
            ),
        }

    def _average_recursive(self, data_list):
        if not data_list:
            return None
        first = data_list[0]
        if isinstance(first, list):
            min_len = min(len(d) for d in data_list)
            return np.mean(np.array([d[:min_len] for d in data_list]), axis=0).tolist()
        elif isinstance(first, dict):
            res = {}
            for key in first.keys():
                sub_list = [
                    d[key] for d in data_list if isinstance(d, dict) and key in d
                ]
                if sub_list:
                    res[key] = self._average_recursive(sub_list)
            return res
        return first

    def load(self, algo, dataset, partition, num_clients, specific_run=None, ablate_name=None, **kwargs):
        folder_name = f"{dataset}_{partition}_{num_clients}"
        if partition == "dirichlet":
            folder_name += f"_{kwargs['alpha']}"
        elif partition == "pathological":
            folder_name += f"_{kwargs['n_class']}"

        # 结果目录路径：算法 / 数据集 /
        folder_path = os.path.join(self.base_dir, algo, folder_name)
        if ablate_name:
            folder_path = os.path.join(folder_path, ablate_name)

        if not os.path.exists(folder_path):
            if specific_run is None or specific_run == 0:
                print(f"  [提示] 实验目录不存在: {folder_path}")
            return None

        # 构造基础文件名 (包含公共参数前缀)
        common_name = f"{kwargs['epochs']}_{kwargs['batch_size']}_{kwargs['lr']}"
        base_name = common_name

        # 添加算法特定的后缀以实现严格参数匹配
        suffix_gen = self.algo_patterns.get(algo)

        # 动态匹配逻辑：对于 dfedset 的各种消融实验版本，自动匹配基础模式
        # 增加 .lower() 确保匹配鲁棒性
        if not suffix_gen and algo.lower().startswith("dfedset"):
            suffix_gen = self.algo_patterns.get("dfedset")

        if suffix_gen:
            base_name += suffix_gen(kwargs)

        def get_indices(b_name):
            if specific_run is not None:
                if os.path.exists(
                    os.path.join(folder_path, f"{b_name}_{specific_run}.pt")
                ):
                    return [specific_run]
                return []
            else:
                indices = []
                idx = 0
                while os.path.exists(os.path.join(folder_path, f"{b_name}_{idx}.pt")):
                    indices.append(idx)
                    idx += 1
                return indices

        # 严格匹配
        run_indices = get_indices(base_name)

        if not run_indices:
            if specific_run is None or specific_run == 0:
                print(
                    f"  [提示] 结果文件不存在: {os.path.join(folder_path, base_name)}_X.pt"
                )
            return None

        loaded_data = []
        for idx in run_indices:
            file_path = os.path.join(folder_path, f"{base_name}_{idx}.pt")
            try:
                loaded_data.append(torch.load(file_path, map_location="cpu"))
            except Exception as e:
                # 真正的加载错误（如文件损坏）才报错
                print(f"  [Error] Failed to load {file_path}: {e}")

        if not loaded_data:
            return None
        if len(loaded_data) == 1:
            return loaded_data[0]

        avg_result = {}
        for key in ["acc", "loss"]:
            if key in loaded_data[0]:
                avg_result[key] = self._average_recursive(
                    [d[key] for d in loaded_data if key in d]
                )
        return avg_result


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
    plt.tight_layout()

    if not os.path.exists("figures"):
        os.makedirs("figures")
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
    plt.xlim(0, x_lim)

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

    if not os.path.exists("figures"):
        os.makedirs("figures")
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


def plot_loss(results_dict, title=None, xlabel="Rounds", ylabel="Loss"):
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
    plt.tight_layout()

    if not os.path.exists("figures"):
        os.makedirs("figures")
    save_name = (title or "loss_comparison").lower().replace(" ", "_").replace(
        "(", ""
    ).replace(")", "") + ".png"
    plt.savefig(os.path.join("figures", save_name), dpi=300)
    print(f"Figure saved to figures/{save_name}")
    plt.show()


def load_plot(
    selected_group, experiments, common_args, loader, x_lim=200, do_plot=True
):
    """
    加载并绘制对比图，根据数据格式自动识别是单线还是 Model/Proto 分离。
    """
    results = {}
    for label in selected_group:
        if label not in experiments:
            continue
        algo_name, kwargs = experiments[label]
        data = loader.load(algo_name, **{**common_args, **kwargs}, specific_run=0)
        if data:
            results[label] = data

    if results:
        if not do_plot:
            print_summary_table(results, x_lim, metric="acc")
            return

        # 尝试检查第一个结果的 'acc' 类型来决定绘图函数
        first_res = list(results.values())[0]
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
    else:
        print(
            "\n[Error] No results loaded. Check the warnings above for path/parameter mismatches."
        )


def load_plot_all_runs(selected_group, experiments, common_args, loader, x_lim=200):
    """
    加载并统计多轮实验的均值和标准差。
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

        run_results = []
        idx = 0
        while True:
            data = loader.load(algo_name, **merged_args, specific_run=idx)
            if data is None:
                break
            run_results.append(data)
            idx += 1

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


def print_stats(selected_group, experiments, common_args, loader, x_lim=200):
    """
    加载并打印多轮实验的均值汇总（Model 和 Proto 分列显示）。
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

        run_results = []
        idx = 0
        while True:
            data = loader.load(algo_name, **merged_args, specific_run=idx)
            if data is None:
                break
            run_results.append(data)
            idx += 1

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
