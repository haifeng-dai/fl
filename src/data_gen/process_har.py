import os
import zipfile

import numpy as np
import pandas as pd
import requests
import torch
from sklearn.preprocessing import StandardScaler

RAW_DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"


def download_and_extract(root):
    if not os.path.exists(root):
        os.makedirs(root)

    zip_path = os.path.join(root, "UCI_HAR_Dataset.zip")
    if not os.path.exists(zip_path):
        print(f"Downloading UCI HAR Dataset to {zip_path}...")
        try:
            r = requests.get(RAW_DATA_URL, timeout=60)
            with open(zip_path, "wb") as f:
                f.write(r.content)
            print("Download complete.")
        except Exception as e:
            print(f"Download failed: {e}")
            raise

    extract_path = os.path.join(root, "UCI HAR Dataset")
    if not os.path.exists(extract_path):
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(root)
        print("Extraction complete.")

    return os.path.join(root, "UCI HAR Dataset")


def load_raw_signals(data_root):
    """Load 9-channel raw inertial signals (128 time steps)"""
    print("Loading Raw Inertial Signals (9x128)...")
    SIGNAL_TYPES = [
        "body_acc_x_",
        "body_acc_y_",
        "body_acc_z_",
        "body_gyro_x_",
        "body_gyro_y_",
        "body_gyro_z_",
        "total_acc_x_",
        "total_acc_y_",
        "total_acc_z_",
    ]

    X_train_list, X_test_list = [], []

    for group in ["train", "test"]:
        folder = os.path.join(data_root, group, "Inertial Signals")
        signals = []
        for signal in SIGNAL_TYPES:
            filename = os.path.join(folder, f"{signal}{group}.txt")
            signals.append(np.loadtxt(filename, dtype=np.float32))
        # Stack to (N, 128, 9) then transpose to (N, 9, 128)
        stacked = np.stack(signals, axis=2).transpose(0, 2, 1)

        if group == "train":
            X_train_list = stacked
        else:
            X_test_list = stacked

    y_train = (
        pd.read_csv(
            os.path.join(data_root, "train", "y_train.txt"),
            sep=r"\s+",
            header=None,
        ).values.flatten()
        - 1
    )
    y_test = (
        pd.read_csv(
            os.path.join(data_root, "test", "y_test.txt"),
            sep=r"\s+",
            header=None,
        ).values.flatten()
        - 1
    )

    return np.concatenate([X_train_list, X_test_list]), np.concatenate(
        [y_train, y_test]
    )


def load_expert_features(data_root):
    """Load 561-dimensional expert features"""
    print("Loading Expert Engineered Features (561)...")
    X_train = pd.read_csv(
        os.path.join(data_root, "train", "X_train.txt"),
        sep=r"\s+",
        header=None,
    ).values
    X_test = pd.read_csv(
        os.path.join(data_root, "test", "X_test.txt"),
        sep=r"\s+",
        header=None,
    ).values

    y_train = (
        pd.read_csv(
            os.path.join(data_root, "train", "y_train.txt"),
            sep=r"\s+",
            header=None,
        ).values.flatten()
        - 1
    )
    y_test = (
        pd.read_csv(
            os.path.join(data_root, "test", "y_test.txt"),
            sep=r"\s+",
            header=None,
        ).values.flatten()
        - 1
    )

    return np.concatenate([X_train, X_test]), np.concatenate([y_train, y_test])


def process(save_dir):
    print("Processing UCI-HAR dataset...")

    # Download to save_dir directly to be consistent with other datasets
    # save_dir is expected to be "./datasets/raw" or similar
    dataset_path = download_and_extract(save_dir)

    # 1. Process Raw Signals
    X_raw, y_raw = load_raw_signals(dataset_path)
    # Standardize Raw Signals (per channel)
    # Shape: (N, 9, 128) -> reshape to (N*128, 9) for scaler -> reshape back
    N, C, L = X_raw.shape
    scaler = StandardScaler()
    X_raw_flat = X_raw.transpose(0, 2, 1).reshape(-1, C)
    X_raw_flat = scaler.fit_transform(X_raw_flat)
    X_raw = X_raw_flat.reshape(N, L, C).transpose(0, 2, 1)

    # Save Raw
    raw_save_path = os.path.join(save_dir, "har_raw.pt")
    torch.save(
        {
            "x": torch.tensor(X_raw, dtype=torch.float32),
            "y": torch.tensor(y_raw, dtype=torch.long),
        },
        raw_save_path,
    )
    print(f"Saved HAR Raw Signals to {raw_save_path} (Shape: {X_raw.shape})")

    # 2. Process Expert Features
    X_feat, y_feat = load_expert_features(dataset_path)
    # Standardize Features
    scaler_feat = StandardScaler()
    X_feat = scaler_feat.fit_transform(X_feat)

    # Save Features
    feat_save_path = os.path.join(save_dir, "har_feat_raw.pt")
    torch.save(
        {
            "x": torch.tensor(X_feat, dtype=torch.float32),
            "y": torch.tensor(y_feat, dtype=torch.long),
        },
        feat_save_path,
    )
    print(f"Saved HAR Expert Features to {feat_save_path} (Shape: {X_feat.shape})")
