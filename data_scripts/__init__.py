import os
import torch
import numpy as np
import importlib

# --- Partition Methods ---

def iid_partition(targets, num_clients):
    indices = np.arange(len(targets))
    np.random.shuffle(indices)
    return np.array_split(indices, num_clients)

def dirichlet_partition(targets, num_clients, alpha=0.5):
    num_classes = len(np.unique(targets))
    indices = [np.where(targets == i)[0] for i in range(num_classes)]
    client_indices = [[] for _ in range(num_clients)]
    
    for k in range(num_classes):
        np.random.shuffle(indices[k])
        proportions = np.random.dirichlet([alpha] * num_clients)
        proportions = (np.cumsum(proportions) * len(indices[k])).astype(int)[:-1]
        splits = np.split(indices[k], proportions)
        for i, split in enumerate(splits):
            client_indices[i].append(split)
            
    return [np.concatenate(idx) for idx in client_indices]

def pathological_partition(targets, num_clients, n_classes_per_client=2):
    num_classes = len(np.unique(targets))
    indices = [np.where(targets == i)[0] for i in range(num_classes)]
    for i in range(num_classes):
        np.random.shuffle(indices[i])
        
    client_indices = [[] for _ in range(num_clients)]
    shards_per_class = (num_clients * n_classes_per_client) // num_classes
    class_shards = []
    for i in range(num_classes):
        shards = np.array_split(indices[i], shards_per_class)
        class_shards.extend(shards)
    
    np.random.shuffle(class_shards)
    for i in range(num_clients):
        for j in range(n_classes_per_client):
            client_indices[i].append(class_shards[i * n_classes_per_client + j])
            
    return [np.concatenate(idx) for idx in client_indices]

# --- Main Entry Point ---

def prepare_data(dataset_name, partition_method, num_clients, **kwargs):
    """
    Unified entry point for data processing and partitioning.
    """
    raw_dir = './datasets/raw'
    raw_path = os.path.join(raw_dir, f'{dataset_name}_raw.pt')
    
    # 1. Ensure Raw Data exists
    if not os.path.exists(raw_path):
        print(f"-> Raw data for {dataset_name} not found. Processing...")
        try:
            module = importlib.import_module(f'data_scripts.process_{dataset_name}')
            module.process(raw_dir)
        except ImportError:
            raise ImportError(f"No processing script: data_scripts/process_{dataset_name}.py")

    # 2. Check if partition already exists
    output_dir = f'./datasets/{dataset_name}/{partition_method}'
    if os.path.exists(output_dir) and len(os.listdir(output_dir)) >= num_clients:
        print(f"-> Partition {partition_method} for {dataset_name} already exists. Skipping.")
        return

    print(f"-> Partitioning {dataset_name} using {partition_method}...")
    data = torch.load(raw_path)
    X, Y = data['x'], data['y']
    
    # 3. Split Global Test Set
    num_samples = len(Y)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    
    test_ratio = kwargs.get('test_ratio', 0.2)
    test_size = int(num_samples * test_ratio)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    
    test_data = {'x': X[test_indices], 'y': Y[test_indices]}
    X_train, Y_train = X[train_indices], Y[train_indices]
    
    # 4. Partition Logic
    if partition_method == 'iid':
        client_indices = iid_partition(Y_train, num_clients)
    elif partition_method == 'dirichlet':
        alpha = kwargs.get('alpha', 0.5)
        client_indices = dirichlet_partition(Y_train, num_clients, alpha)
    elif partition_method == 'pathological':
        n_classes = kwargs.get('n_classes', 2)
        client_indices = pathological_partition(Y_train, num_clients, n_classes)
    else:
        raise ValueError(f"Unknown partition: {partition_method}")
        
    # 5. Save
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    torch.save(test_data, os.path.join(output_dir, 'test_data.pt'))
    for i, idx in enumerate(client_indices):
        torch.save({'x': X_train[idx], 'y': Y_train[idx]}, os.path.join(output_dir, f'client_{i}.pt'))
        
    print(f"-> Successfully prepared {dataset_name} ({partition_method}) for {num_clients} clients.")
