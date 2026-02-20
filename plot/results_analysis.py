# %% [markdown]
# Federated Learning Results Analysis
# Load, visualize, and compare results from different FL algorithms.

# %%
import os
import torch
import matplotlib.pyplot as plt
import numpy as np

# Set plot style
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except:
    plt.style.use("ggplot")


# %%
class ResultLoader:
    def __init__(self, base_dir="results"):
        self.base_dir = base_dir

        # Dictionary mapping algorithm names to their specific filename suffix generators
        # Base format is always: {epochs}_{batch_size}_{lr}
        self.algo_patterns = {
            "fedala": lambda args: f"_{args['eta']}_{args['rand_percent']}_{args['layer_idx']}_{args['ala_threshold']}_{args['num_pre_loss']}",
            "fedavg": lambda args: "",
            "feddyn": lambda args: f"_alpha{args['alpha_coef']}",
            "feddpl": lambda args: f"_{args['lambda_']}_{args['epoch_pln']}_{args['lr_pln']}_{args['batch_size_pln']}_{args['depth_pln']}_{args['width_pln']}_{args['mode']}_{args['fixed_proto']}_{args['init_emb']}_{args['har']}",
            "fedfm": lambda args: f"_{args['mu']}",
            "fedkd": lambda args: f"_{args['lr_g']}_{args['energy']}",
            "fedlsa": lambda args: f"_{args['lambda_com']}_{args['alpha_sep']}_{args['server_epochs']}_{args['server_lr']}_{args['tau']}",
            "fedper": lambda args: "",
            "fedpln": lambda args: f"_{args['lambda_']}_{args['epoch_pln']}_{args['lr_pln']}_{args['batch_size_pln']}_{args['depth_pln']}_{args['width_pln']}_{args['mode']}_{args['fixed_proto']}_{args['init_emb']}_{args['har']}",
            "fedproc": lambda args: f"_{args['mu']}_{args['temperature']}",
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
            "fedtest": lambda args: "",
        }

    def _average_recursive(self, data_list):
        """Recursively average lists and dictionaries."""
        if not data_list:
            return None

        first = data_list[0]
        if isinstance(first, list):
            try:
                # Average lists by taking the minimum length to avoid shape mismatch
                min_len = min(len(d) for d in data_list)
                data_np = np.array([d[:min_len] for d in data_list])
                return np.mean(data_np, axis=0).tolist()
            except Exception as e:
                print(f"Warning: Could not average lists: {e}")
                return first
        elif isinstance(first, dict):
            res = {}
            for key in first.keys():
                sub_list = [
                    d[key] for d in data_list if isinstance(d, dict) and key in d
                ]
                if sub_list:
                    res[key] = self._average_recursive(sub_list)
            return res
        else:
            # For other types (int, float), return the mean if possible
            try:
                return np.mean(data_list)
            except:
                return first

    def load(self, algo, dataset, partition, num_clients, specific_run=None, **kwargs):
        """
        Load results.
        :param specific_run: If None, loads all available runs and averages them.
                             If int (e.g., 0), loads only that specific run index.
        """
        # 1. Construct Folder Path
        folder_name = f"{dataset}_{partition}_{num_clients}"
        if partition == "dirichlet":
            folder_name += f"_{kwargs.get('alpha', 0.1)}"
        elif partition == "pathological":
            folder_name += f"_{kwargs.get('n_class', 2)}"

        folder_path = os.path.join(self.base_dir, algo, folder_name)

        # 2. Construct Base Filename (without index)
        epochs = kwargs.get("epochs", 10)
        batch_size = kwargs.get("batch_size", 64)
        lr = kwargs.get("lr", 0.01)
        base_name = f"{epochs}_{batch_size}_{lr}"

        suffix_gen = self.algo_patterns.get(algo)
        if suffix_gen:
            full_args = kwargs.copy()
            full_args.update({"dataset": dataset, "model": kwargs.get("model", "cnn")})
            try:
                base_name += suffix_gen(full_args)
            except KeyError as e:
                print(f"Error: Missing argument {e} required for algorithm {algo}")
                return None

        # 3. Determine runs to load
        if specific_run is not None:
            # Load only one specific run
            run_indices = [specific_run]
        else:
            # Scan for all available runs
            run_indices = []
            run_idx = 0
            while True:
                file_name = f"{base_name}_{run_idx}.pt"
                full_path = os.path.join(folder_path, file_name)
                if not os.path.exists(full_path):
                    break
                run_indices.append(run_idx)
                run_idx += 1

        if not run_indices:
            print(f"[{algo}] No results found in {folder_path}")
            return None

        # 4. Load Data
        loaded_data = []
        for idx in run_indices:
            file_name = f"{base_name}_{idx}.pt"
            full_path = os.path.join(folder_path, file_name)
            try:
                data = torch.load(full_path, map_location="cpu")
                loaded_data.append(data)
            except Exception as e:
                print(f"Failed to load {full_path}: {e}")

        if not loaded_data:
            return None

        # 5. Aggregate/Average Results
        if len(loaded_data) == 1:
            return loaded_data[0]

        avg_result = {}
        first_run = loaded_data[0]

        # Process all keys found in the result files
        for key in first_run.keys():
            if key in ["acc", "loss", "acc_p", "loss_p"]:
                all_runs_metric = [d[key] for d in loaded_data if key in d]
                avg_result[key] = self._average_recursive(all_runs_metric)
            else:
                # For non-metric data, just copy from first run
                avg_result[key] = first_run[key]

        return avg_result


# %%
def plot_results(
    results_dict, metric="acc", title=None, xlabel="Rounds", ylabel="Accuracy"
):
    plt.figure(figsize=(10, 6))
    for label, data in results_dict.items():
        if data is None:
            continue

        metric_data = data.get(metric)
        if metric_data is None:
            continue

        if isinstance(metric_data, list):
            max_val = max(metric_data) if metric_data else 0
            plt.plot(metric_data, label=f"{label} (Model) Max: {max_val:.2f}")
        elif isinstance(metric_data, dict):
            if "model" in metric_data:
                model_data = metric_data["model"]
                max_val = max(model_data) if model_data else 0
                plt.plot(model_data, label=f"{label} (Model) Max: {max_val:.2f}")

            # Check for various prototype naming conventions
            proto_key = None
            for k in ["prototype", "proto", "acc_p", "pln"]:
                if k in metric_data:
                    proto_key = k
                    break

            if proto_key:
                proto_data = metric_data[proto_key]
                max_val = max(proto_data) if proto_data else 0
                plt.plot(
                    proto_data,
                    linestyle="--",
                    alpha=0.7,
                    label=f"{label} (Proto) Max: {max_val:.2f}",
                )

    plt.title(title or f"Comparison of {metric.upper()}")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save figure
    if not os.path.exists("figures"):
        os.makedirs("figures")

    save_name = (title or "comparison").lower().replace(" ", "_") + ".png"
    save_path = os.path.join("figures", save_name)
    plt.savefig(save_path, dpi=300)
    print(f"Figure saved to {save_path}")

    plt.show()


# %%
# 1. Initialize Loader
loader = ResultLoader()

# 2. Common Settings (Adjust these)
common_args = {
    "dataset": "cifar10",
    "partition": "dirichlet",
    "num_clients": 10,
    "epochs": 10,
    "batch_size": 64,
    "lr": 0.01,
    "alpha": 0.1,
    "n_class": 2,
    "model": "cnn",
    "feature_dim": 512,
}

# 3. Define Experiments to Compare
experiments = {
    "FedALA": (
        "fedala",
        {
            "eta": 1.0,
            "rand_percent": 80,
            "layer_idx": 2,
            "ala_threshold": 0.1,
            "num_pre_loss": 10,
        },
    ),
    "FedAvg": ("fedavg", {}),
    "FedDyn": ("feddyn", {"alpha_coef": 0.01}),
    "FedDPL": (
        "feddpl",
        {
            "lambda_": 0.01,
            "epoch_pln": 10,
            "lr_pln": 0.01,
            "batch_size_pln": 64,
            "feature_dim": 512,
            "depth_pln": 1,
            "width_pln": 512,
            "mode": "normal",
            "fixed_proto": 0,
            "init_emb": 0,
            "har": 0,
        },
    ),
    "FedKD": ("fedkd", {"lr_g": 0.01, "energy": 0.9}),
    "FedFM": ("fedfm", {"mu": 1.0}),
    "FedLSA": (
        "fedlsa",
        {
            "lambda_com": 0.1,
            "alpha_sep": 0.1,
            "server_epochs": 10,
            "server_lr": 0.01,
            "tau": 0.1,
        },
    ),
    "FedPer": ("fedper", {}),
    "FedPLN": (
        "fedpln",
        {
            "lambda_": 10.0,
            "epoch_pln": 10,
            "lr_pln": 0.01,
            "batch_size_pln": 64,
            "feature_dim": 512,
            "depth_pln": 1,
            "width_pln": 512,
            "mode": "normal",
            "har": 0,
            "fixed_proto": 0,
            "init_emb": 0,
        },
    ),
    "FedProc": ("fedproc", {"mu": 1.0, "temperature": 0.5}),
    "FedProto": ("fedproto", {"mu": 0.1}),
    "SCAFFOLD": ("scaffold", {"global_lr": 1.0}),
    "FedProx": ("fedprox", {"mu": 0.01}),
    "FedRep": ("fedrep", {"epochs_head": 5}),
    "FedSA": (
        "fedsa",
        {"alpha_sa": 0.5, "lambda_r": 0.1, "lambda_mcl": 0.1, "lambda_cc": 0.1},
    ),
    "FedTGP": (
        "fedtgp",
        {
            "lamda_": 10.0,
            "server_epochs": 10,
            "server_lr": 0.01,
            "margin_threshold": 1.0,
            "feature_dim": 512,
        },
    ),
    "FML": ("fml", {"alpha_fml": 1.0, "beta_fml": 1.0}),
    "LGFedAvg": ("lgfedavg", {}),
    "MOON": ("moon", {"mu": 0.01, "tau": 0.5}),
    "ProxyFL": ("proxyfl", {"mu": 1.0, "adj_type": "ring"}),
    "Fedtest": ("fedtest", {"mu": 0.1}),
}


traditional_algos = [
    "FedAvg",
    "FedDyn",
    "FedFM",
    "FedLSA",
    "FedPLN",
    "FedProc",
    "FedProx",
    "MOON",
    "SCAFFOLD",
]

personalized_algos = [
    "FedALA",
    "FedDPL",
    "FedKD",
    "FedPer",
    "FedProto",
    "FedRep",
    "FedSA",
    "FedTGP",
    "FML",
    "LGFedAvg",
    "ProxyFL",
]

# Select algorithms to plot
algos = traditional_algos
# algos = personalized_algos
# algos = traditional_algos + personalized_algos

# 4. Load & Plot
results = {}
for label, (algo, kwargs) in experiments.items():
    if label not in algos:
        continue

    args = {**common_args, **kwargs}
    # specific_run=None will load all available runs and average them
    data = loader.load(algo, **args, specific_run=0)
    if data:
        results[label] = data

if results:
    title = f"Test Accuracy on {common_args['dataset']} ({common_args['partition']})"
    plot_results(results, metric="acc", title=title)
else:
    print("No results found. Please check your data directory and paths.")
