import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import re

PARAM_MAP = {
    "lamda_": r"$\lambda$",
    "lambda_": r"$\lambda$",
    "mu": r"$\mu$",
    "eta": r"$\eta$",
    "alpha": r"$\alpha$",
    "tau": r"$\tau$",
    "rho": r"$\rho$",
    "lr": "LR",
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
        self.algo_patterns = {
            "fedala": lambda args: f"_{args['eta']}_{args['rand_percent']}_{args['layer_idx']}_{args['ala_threshold']}_{args['num_pre_loss']}",
            "fedavg": lambda args: "",
            "feddyn": lambda args: f"_alpha{args['alpha_coef']}",
            "fedfm": lambda args: f"_{args['mu']}",
            "fedkd": lambda args: f"_{args['lr_g']}_{args['energy']}",
            "fedlsa": lambda args: f"_{args['lambda_com']}_{args['alpha_sep']}_{args['server_epochs']}_{args['server_lr']}_{args['tau']}",
            "fedper": lambda args: "",
            "fedpln": lambda args: f"_{args['lambda_']}_{args['epoch_pln']}_{args['lr_pln']}_{args['batch_size_pln']}_{args['depth_pln']}_{args['width_pln']}_{args['mode']}_{args['fixed_proto']}_{args['init_emb']}_{args['har']}",
            "fedproc": lambda args: "",
            "fedproto": lambda args: f"_{args['mu']}",
            "fedprox": lambda args: f"_{args['mu']}",
            "fedrep": lambda args: f"_{args['epochs_head']}",
            "fedsa": lambda args: f"_{args['alpha_sa']}_{args['lambda_r']}_{args['lambda_mcl']}_{args['lambda_cc']}",
            "scaffold": lambda args: f"_glr{args['global_lr']}",
            "fedtgp": lambda args: f"_{args['lamda_']}_{args['server_epochs']}_{args['server_lr']}_{args['margin_threshold']}",
            "fml": lambda args: f"_{args['alpha_fml']}_{args['beta_fml']}",
            "lgfedavg": lambda args: "",
            "moon": lambda args: f"_{args['mu']}_{args['tau']}",
            "proxyfl": lambda args: f"_{args['mu']}_{args['adj_type']}",
            "fedtest": lambda args: f"_{args['mu_test']}",
            "local": lambda args: "_local",
        }

    def _average_recursive(self, data_list):
        if not data_list: return None
        first = data_list[0]
        if isinstance(first, list):
            min_len = min(len(d) for d in data_list)
            return np.mean(np.array([d[:min_len] for d in data_list]), axis=0).tolist()
        elif isinstance(first, dict):
            res = {}
            for key in first.keys():
                sub_list = [d[key] for d in data_list if isinstance(d, dict) and key in d]
                if sub_list: res[key] = self._average_recursive(sub_list)
            return res
        return first

    def load(self, algo, dataset, partition, num_clients, specific_run=None, **kwargs):
        folder_name = f"{dataset}_{partition}_{num_clients}"
        if partition == "dirichlet": folder_name += f"_{kwargs.get('alpha', 0.1)}"
        elif partition == "pathological": folder_name += f"_{kwargs.get('n_class', 2)}"

        folder_path = os.path.join(self.base_dir, algo, folder_name)
        if not os.path.exists(folder_path):
            print(f"  [Warning] Folder not found: {folder_path}")
            return None

        base_name = f"{kwargs.get('epochs', 10)}_{kwargs.get('batch_size', 64)}_{kwargs.get('lr', 0.01)}"
        suffix_gen = self.algo_patterns.get(algo)
        if suffix_gen:
            try: base_name += suffix_gen(kwargs)
            except KeyError as e:
                print(f"  [Error] Missing parameter {e} for algo {algo}")
                return None

        if specific_run is not None: run_indices = [specific_run]
        else:
            run_indices = []
            idx = 0
            while os.path.exists(os.path.join(folder_path, f"{base_name}_{idx}.pt")):
                run_indices.append(idx)
                idx += 1

        if not run_indices:
            print(f"  [Warning] No files found for: {base_name}_*.pt in {folder_path}")
            return None

        loaded_data = []
        for idx in run_indices:
            file_path = os.path.join(folder_path, f"{base_name}_{idx}.pt")
            try:
                loaded_data.append(torch.load(file_path, map_location="cpu"))
            except Exception as e:
                print(f"  [Error] Failed to load {file_path}: {e}")

        if not loaded_data: return None
        if len(loaded_data) == 1: return loaded_data[0]

        avg_result = {}
        for key in ["acc", "loss"]:
            if key in loaded_data[0]:
                avg_result[key] = self._average_recursive([d[key] for d in loaded_data if key in d])
        return avg_result

def plot_results(results_dict, x_lim, metric="acc", title=None, xlabel="Rounds", ylabel="Accuracy"):
    plt.figure(figsize=(12, 7))
    summary = []

    for label, data in results_dict.items():
        if data is None: continue
        metric_data = data.get(metric)
        if metric_data is None: continue

        if isinstance(metric_data, list):
            y = metric_data[0:x_lim]
            max_val = max(y)
            last_10_avg = np.mean(y[-10:]) if len(y) >= 10 else np.mean(y)
            plt.plot(y, label=f"{label} (Max: {max_val:.4f}, Last10: {last_10_avg:.4f})")
            summary.append({"Algorithm": label, "Max": max_val, "Last10": last_10_avg})
        elif isinstance(metric_data, dict):
            if "model" in metric_data:
                y = metric_data["model"][0:x_lim]
                max_val = max(y)
                last_10_avg = np.mean(y[-10:]) if len(y) >= 10 else np.mean(y)
                display_label = beautify_label(label)
                plt.plot(y, label=f"{display_label}-Model (Max: {max_val:.4f}, Last10: {last_10_avg:.4f})")
                summary.append({"Algorithm": f"{display_label}-Model", "Max": max_val, "Last10": last_10_avg})
            if "proto" in metric_data:
                y = metric_data["proto"][0:x_lim]
                max_val = max(y)
                last_10_avg = np.mean(y[-10:]) if len(y) >= 10 else np.mean(y)
                display_label = beautify_label(label)
                plt.plot(y, linestyle="--", alpha=0.8, label=f"{display_label}-Proto (Max: {max_val:.4f}, Last10: {last_10_avg:.4f})")
                summary.append({"Algorithm": f"{display_label}-Proto", "Max": max_val, "Last10": last_10_avg})

    plt.title(title or f"Comparison of {metric.upper()}")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if not os.path.exists("figures"): os.makedirs("figures")
    save_name = (title or "comparison").lower().replace(" ", "_").replace("(", "").replace(")", "") + ".png"
    plt.savefig(os.path.join("figures", save_name), dpi=300)
    print(f"Figure saved to figures/{save_name}")

    # Print summary table
    print(f"\nSummary of {metric.upper()}:")
    print("-" * 65)
    print(f"{'Algorithm':<20} | {'Max Acc':>15} | {'Last 10 Avg':>15}")
    print("-" * 65)
    for entry in summary:
        print(f"{entry['Algorithm']:<20} | {entry['Max']:>15.4f} | {entry['Last10']:>15.4f}")
    print("-" * 65)
    plt.xlim(0, x_lim)

    plt.show()


def print_summary_table(results_dict, x_lim, metric="acc", label_name="Algorithm"):
    """Prints a consolidated summary table for model and prototype metrics."""
    print(f"\nSummary of {metric.upper()}:")

    display_label_name = beautify_label(label_name)
    header = f"{display_label_name:<20} | {'Model Max':>12} | {'Model Last10':>12} | {'Proto Max':>12} | {'Proto Last10':>12}"
    divider = "-" * len(header)
    print(divider)
    print(header)
    print(divider)

    for label, data in results_dict.items():
        if data is None or metric not in data: continue
        metric_data = data[metric]
        if not isinstance(metric_data, dict): continue

        m_max, m_last10 = 0.0, 0.0
        p_max, p_last10 = 0.0, 0.0

        if "model" in metric_data:
            y = np.array(metric_data["model"][0:x_lim])
            m_max = np.max(y)
            m_last10 = np.mean(y[-10:]) if len(y) >= 10 else np.mean(y)

        pk = "proto" if "proto" in metric_data else "prototype" if "prototype" in metric_data else None
        if pk:
            y = np.array(metric_data[pk][0:x_lim])
            p_max = np.max(y)
            p_last10 = np.mean(y[-10:]) if len(y) >= 10 else np.mean(y)

        # Clean up row label: if it's "param=value" and param matches label_name, just show "value"
        clean_row_label = label
        if "=" in label:
            param, val = label.split("=", 1)
            if param == label_name:
                clean_row_label = val

        print(f"{clean_row_label:<20} | {m_max:>12.4f} | {m_last10:>12.4f} | {p_max:>12.4f} | {p_last10:>12.4f}")

    print(divider)

def plot_results_split(results_dict, x_lim, metric="acc", title=None, xlabel="Rounds", ylabel="Accuracy", label_name="Algorithm"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    for label, data in results_dict.items():
        if data is None: continue
        metric_data = data.get(metric)
        if metric_data is None: continue

        if isinstance(metric_data, dict):
            display_label = beautify_label(label)
            # Model Acc on Ax1 (Left)
            if "model" in metric_data:
                y = metric_data["model"][0:x_lim]
                max_val = max(y)
                ax1.plot(y, label=f"{display_label} (Max: {max_val:.4f})")

            # Proto Acc on Ax2 (Right)
            if "proto" in metric_data:
                y = metric_data["proto"][0:x_lim]
                max_val = max(y)
                ax2.plot(y, label=f"{display_label} (Max: {max_val:.4f})")

    ax1.set_title(f"Model {metric.upper()}: {title or ''}")
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, x_lim)

    ax2.set_title(f"Prototype {metric.upper()}: {title or ''}")
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(ylabel)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, x_lim)

    plt.tight_layout()

    if not os.path.exists("figures"): os.makedirs("figures")
    save_base = (title or "comparison").lower().replace(" ", "_").replace("(", "").replace(")", "").replace("__", "_")
    save_name = f"split_{save_base}.png"
    plt.savefig(os.path.join("figures", save_name), dpi=300)
    print(f"Figure saved to figures/{save_name}")

    # Use the extracted summary function
    print_summary_table(results_dict, x_lim, metric, label_name)

    plt.show()

def plot_loss(results_dict, title=None, xlabel="Rounds", ylabel="Loss"):
    plt.figure(figsize=(12, 7))
    for label, data in results_dict.items():
        if data is None: continue
        loss_data = data.get("loss")
        if loss_data is None: continue

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
                plt.plot(y, linestyle="--", alpha=0.8, label=f"{display_label}-Proto Loss")
            if "aux" in loss_data:
                aux = loss_data["aux"]
                if "model_m" in aux:
                    y = aux["model_m"]
                    plt.plot(y, linestyle=":", alpha=0.6, label=f"{display_label}-CE Loss")
                if "model_p" in aux:
                    y = aux["model_p"]
                    plt.plot(y, linestyle=":", alpha=0.6, label=f"{display_label}-Align Loss")
            if "server_tgp" in loss_data:
                y = loss_data["server_tgp"]
                plt.plot(y, linestyle="--", alpha=0.9, label=f"{display_label}-TGP Total")
            if "server_tgp_ce" in loss_data:
                y = loss_data["server_tgp_ce"]
                plt.plot(y, linestyle=":", alpha=0.7, label=f"{display_label}-TGP CE")
            if "server_tgp_mse" in loss_data:
                y = loss_data["server_tgp_mse"]
                plt.plot(y, linestyle=":", alpha=0.7, label=f"{display_label}-TGP MSE")

    plt.title(title or "Comparison of Loss")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale("log")
    plt.tight_layout()

    if not os.path.exists("figures"): os.makedirs("figures")
    save_name = (title or "loss_comparison").lower().replace(" ", "_").replace("(", "").replace(")", "") + ".png"
    plt.savefig(os.path.join("figures", save_name), dpi=300)
    print(f"Figure saved to figures/{save_name}")
    plt.show()
