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
            "feddpl": lambda args: f"_{args['lambda_']}_{args['epoch_pln']}_{args['lr_pln']}_{args['batch_size_pln']}_{args['feature_dim']}_{args['depth_pln']}_{args['width_pln']}_{args['mode']}_{args['fixed_proto']}_{args['init_emb']}_{args['har']}",
            "fedkd": lambda args: f"_{args['lr_g']}_{args['energy']}",
            "fedlsa": lambda args: f"_{args['lambda_com']}_{args['alpha_sep']}_{args['server_epochs']}_{args['server_lr']}_{args['tau']}",
            "fedper": lambda args: "",
            "fedpln": lambda args: f"_{args['lambda_']}_{args['epoch_pln']}_{args['lr_pln']}_{args['batch_size_pln']}_{args['feature_dim']}_{args['depth_pln']}_{args['width_pln']}_{args['mode']}_{args['har']}_{args['fixed_proto']}_{args['init_emb']}",
            "fedproto": lambda args: f"_{args['mu']}",
            "fedprox": lambda args: f"_{args['mu']}",
            "fedrep": lambda args: f"_{args['dataset']}_{args['model']}_{args['epochs']}_{args['epochs_head']}",
            "fedsa": lambda args: f"_{args['alpha_sa']}_{args['lambda_r']}_{args['lambda_mcl']}_{args['lambda_cc']}",
            "fedtgp": lambda args: f"_{args['lamda']}_{args['server_epochs']}_{args['server_lr']}_{args['margin_threshold']}_{args['feature_dim']}",
            "fml": lambda args: f"_{args['alpha_fml']}_{args['beta_fml']}",
            "lgfedavg": lambda args: "",
            "moon": lambda args: f"_{args['mu']}_{args['tau']}",
            "proxyfl": lambda args: f"_{args['mu']}_{args['adj_type']}",
        }

    def get_path(
        self, algo, dataset, partition, num_clients, alpha=None, n_class=None, **kwargs
    ):
        # 1. Construct Folder Path
        folder_name = f"{dataset}_{partition}_{num_clients}"
        if partition == "dirichlet":
            folder_name += f"_{alpha}"
        elif partition == "pathological":
            folder_name += f"_{n_class}"

        folder_path = os.path.join(self.base_dir, algo, folder_name)

        # 2. Construct Filename
        epochs = kwargs.get("epochs", 10)
        batch_size = kwargs.get("batch_size", 64)
        lr = kwargs.get("lr", 0.01)

        base_name = f"{epochs}_{batch_size}_{lr}"

        suffix_gen = self.algo_patterns.get(algo)
        if suffix_gen is None:
            suffix = ""
        else:
            full_args = kwargs.copy()
            full_args.update({"dataset": dataset, "model": kwargs.get("model", "cnn")})
            try:
                suffix = suffix_gen(full_args)
            except KeyError as e:
                print(f"Error: Missing argument {e} required for algorithm {algo}")
                return None

        file_name = f"{base_name}{suffix}.pt"
        full_path = os.path.join(folder_path, file_name)
        return full_path

    def load(self, algo, dataset, partition, num_clients, **kwargs):
        path = self.get_path(algo, dataset, partition, num_clients, **kwargs)
        if not path or not os.path.exists(path):
            return None
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"Failed to load {path}: {e}")
            return None


# %%
def plot_results(
    results_dict, metric="acc", title=None, xlabel="Rounds", ylabel="Accuracy"
):
    plt.figure(figsize=(10, 6))
    for label, data in results_dict.items():
        if data is None:
            continue

        if isinstance(data.get(metric), list):
            plt.plot(data[metric], label=f"{label} (Model)")
        elif isinstance(data.get(metric), dict):
            if "model" in data[metric]:
                plt.plot(data[metric]["model"], label=f"{label} (Model)")
            if "prototype" in data[metric]:
                plt.plot(
                    data[metric]["prototype"],
                    linestyle="--",
                    alpha=0.7,
                    label=f"{label} (Proto)",
                )

    plt.title(title or f"Comparison of {metric.upper()}")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# %%
# 1. Initialize Loader
loader = ResultLoader()

# 2. Common Settings (Adjust these)
common_args = {
    "dataset": "mnist",
    "partition": "iid",
    "num_clients": 10,
    "epochs": 10,
    "batch_size": 64,
    "lr": 0.01,
    "alpha": 0.1,
    "n_class": 2,
    "model": "cnn",
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
    "FedDPL": (
        "feddpl",
        {
            "lambda_": 10.0,
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
    "FedProto": ("fedproto", {"mu": 1.0}),
    "FedProx": ("fedprox", {"mu": 0.01}),
    "FedRep": ("fedrep", {"epochs_head": 5}),
    "FedSA": (
        "fedsa",
        {"alpha_sa": 0.5, "lambda_r": 0.1, "lambda_mcl": 0.1, "lambda_cc": 0.1},
    ),
    "FedTGP": (
        "fedtgp",
        {
            "lamda": 10.0,
            "server_epochs": 10,
            "server_lr": 0.01,
            "margin_threshold": 1.0,
            "feature_dim": 512,
        },
    ),
    "FML": ("fml", {"alpha_fml": 1.0, "beta_fml": 1.0}),
    "LGFedAvg": ("lgfedavg", {}),
    "MOON": ("moon", {"mu": 1.0, "tau": 0.5}),
    "ProxyFL": ("proxyfl", {"mu": 1.0, "adj_type": "ring"}),
}

# 4. Load & Plot
results = {}
for label, (algo, kwargs) in experiments.items():
    args = {**common_args, **kwargs}
    data = loader.load(algo, **args)
    if data:
        results[label] = data

if results:
    title = f"Test Accuracy on {common_args['dataset']} ({common_args['partition']})"
    plot_results(results, metric="acc", title=title)
else:
    print("No results found. Please check your data directory and paths.")
