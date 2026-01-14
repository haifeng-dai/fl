# Gemini Code Understanding Report

## Project Overview

This project is a multi-GPU parallel federated learning framework with a unified pipeline. It's designed for running federated learning experiments with different algorithms and data partitioning strategies.

**Key Features:**

*   **Unified Entry Point:** All experiments are run through `main.py`.
*   **Automatic Data Management:** The framework automatically downloads and preprocesses data if it's not found.
*   **Algorithm and Parameter Decoupling:** It supports dynamic loading of algorithm-specific parameters.
*   **High-Performance Simulation:** It supports multi-GPU parallelism for client training.

**Technologies:**

*   **Language:** Python
*   **Core Libraries:** PyTorch, NumPy
*   **Dependency Management:** `uv`

**Architecture:**

*   The project is structured into `src`, `data_scripts`, and `dataset` directories.
*   `main.py` is the central entry point that orchestrates the entire process.
*   `src` contains the implementations of the federated learning algorithms (e.g., FedAvg, MOON) and models.
*   `data_scripts` contains scripts for processing different datasets.
*   `dataset` stores the processed data.

## Building and Running

### Environment Setup

The project uses `uv` for dependency management. To install the dependencies, run:

```bash
uv sync
```

### Running Experiments

The project provides a convenience script `run_demo.sh` for a quick start:

```bash
./run_demo.sh
```

You can also run experiments directly using `main.py` with `uv run`. Here are some examples:

**Run FedAvg with Dirichlet partitioning:**

```bash
uv run main.py --algo fedavg --dataset mnist --partition dirichlet --alpha 0.5 --num_clients 10 --gpus 0,1
```

**Run MOON with pathological partitioning:**

```bash
uv run main.py --algo moon --dataset mnist --partition pathological --n_classes 2 --num_clients 10 --gpus 0
```

## Development Conventions

### Adding a New Dataset

1.  Create a new `process_xxx.py` file in the `data_scripts/` directory.
2.  Implement a `process()` function in this file that downloads, processes, and saves the dataset.

### Adding a New Algorithm

1.  Create a new Python file for your algorithm in the `src/` directory.
2.  Implement `Server` and `Client` classes for your algorithm.
3.  Implement an `add_args` function to add any algorithm-specific command-line arguments.
