import os
import torch
import matplotlib.pyplot as plt
import numpy as np

class ResultLoader:
    def __init__(self, base_dir="results"):
        self.base_dir = base_dir
        self.algo_patterns = {
            "fedala": lambda args: f"_{args['eta']}_{args['rand_percent']}_{args['layer_idx']}_{args['ala_threshold']}_{args['num_pre_loss']}",
            "fedavg": lambda args: "",
            "feddyn": lambda args: f"_alpha{args['alpha_coef']}",
            "feddpl": lambda args: f"_{args['lambda_']}_{args['epoch_pln']}_{args['lr_pln']}_{args['batch_size_pln']}_{args['depth_pln']}_{args['width_pln']}_{args['mode']}_{args['fixed_proto']}_{args['init_emb']}_{args['har']}",
            "feddpl1": lambda args: f"_{args['lambda_']}_{args['epoch_pln']}_{args['lr_pln']}_{args['batch_size_pln']}_{args['depth_pln']}_{args['width_pln']}_{args['mode']}_{args['fixed_proto']}_{args['init_emb']}_{args['har']}",
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
            "fedtgp1": lambda args: f"_{args['lamda_']}_{args['head_epochs']}_{args['body_epochs']}_{args['lr_head']}_{args['lr_body']}",
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

def plot_results(results_dict, metric="acc", title=None, xlabel="Rounds", ylabel="Accuracy"):
    plt.figure(figsize=(12, 7))
    for label, data in results_dict.items():
        if data is None: continue
        metric_data = data.get(metric)
        if metric_data is None: continue
        if isinstance(metric_data, list):
            y = metric_data
            plt.plot(y, label=f"{label} (Max: {max(y):.2f})")
        elif isinstance(metric_data, dict):
            if "model" in metric_data:
                y = metric_data["model"]
                plt.plot(y, label=f"{label}-Model (Max: {max(y):.2f})")
            if "proto" in metric_data:
                y = metric_data["proto"]
                plt.plot(y, linestyle="--", alpha=0.8, label=f"{label}-Proto (Max: {max(y):.2f})")

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
    plt.show()

def plot_loss(results_dict, title=None, xlabel="Rounds", ylabel="Loss"):
    plt.figure(figsize=(12, 7))
    for label, data in results_dict.items():
        if data is None: continue
        loss_data = data.get("loss")
        if loss_data is None: continue

        if isinstance(loss_data, list):
            y = loss_data
            plt.plot(y, label=f"{label} Loss")
        elif isinstance(loss_data, dict):
            if "model" in loss_data:
                y = loss_data["model"]
                plt.plot(y, label=f"{label}-Model Loss")
            if "proto" in loss_data:
                y = loss_data["proto"]
                plt.plot(y, linestyle="--", alpha=0.8, label=f"{label}-Proto Loss")
            if "aux" in loss_data:
                aux = loss_data["aux"]
                if "model_m" in aux:
                    y = aux["model_m"]
                    plt.plot(y, linestyle=":", alpha=0.6, label=f"{label}-CE Loss")
                if "model_p" in aux:
                    y = aux["model_p"]
                    plt.plot(y, linestyle=":", alpha=0.6, label=f"{label}-Align Loss")

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
