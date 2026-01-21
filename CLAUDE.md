# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a multi-GPU parallel federated learning framework (统一流水线) designed for efficient and scalable FL research. The framework supports multiple algorithms (FedAvg, MOON, FedPLN, FedDPL, FedProto) with automatic data management and high-performance multi-GPU client simulation.

## Common Commands

### Environment Setup
```bash
# Install dependencies using uv
uv sync
```

### Running Experiments

**Main entry point**: All experiments run through `main.py` with `uv run`

```bash
# Basic FedAvg with Dirichlet partitioning
uv run main.py --algo fedavg --dataset mnist --partition dirichlet --alpha 0.5 --num_clients 10 --gpus 0,1

# MOON with pathological partitioning
uv run main.py --algo moon --dataset mnist --partition pathological --n_classes 2 --num_clients 10 --gpus 0

# Run all configured experiments via scripts
./run.sh
```

### Algorithm-Specific Scripts

Individual algorithm scripts are in `scripts/` directory:
- `scripts/fedavg.sh` - FedAvg experiments
- `scripts/moon.sh` - MOON experiments
- `scripts/fedpln.sh` - FedPLN experiments
- `scripts/feddpl.sh` - FedDPL experiments
- `scripts/fedproto.sh` - FedProto experiments

These scripts support batch experiments with multiple hyperparameter configurations through environment variables set in `run.sh`.

### Code Formatting
```bash
uv run black .
```

## Architecture

### Entry Point Flow

1. **`main.py`** - Single unified entry point:
   - First-pass argument parsing to determine algorithm (`--algo`)
   - Builds full parser with common args (data, training, GPU config)
   - Dynamically imports algorithm module from `src/{algo}.py`
   - Calls algorithm's `add_args()` to register algorithm-specific parameters
   - Automatically triggers data preparation via `src.data_gen.prepare_data()`
   - Instantiates `Server` class from algorithm module and runs `fit()`

### Core Architecture Pattern

All federated learning algorithms follow a consistent structure:

**Algorithm Module** (`src/{algorithm}.py`):
- `add_args(parser)` - Registers algorithm-specific CLI arguments
- `client_worker(client_id, params)` - Worker function for parallel client training
- `Server(BaseServer)` - Main server class inheriting from `BaseServer`

**Client Worker Function**:
- Receives parameters as a list (GPU device, model state, training data, hyperparameters)
- Instantiates model on assigned GPU using `get_model(model_name, dataset_name)`
- Performs local training epochs
- Returns `(client_id, [loss, model.state_dict()])` or algorithm-specific results

**Server Class**:
- `__init__(args)` - Initialize model, call `super().__init__(model, personalized_flag, args)`
- `fit()` - Main training loop coordinating rounds of client training and aggregation
- `aggregate()` - Aggregates client models (inherits from BaseServer or customizes)
- `evaluate()` - Evaluates global model performance
- `save(test)` - Saves results and model checkpoints

### Multi-GPU Parallelization

**GPU Assignment** (`src/utils/parallel.py`):
- `run_parallel_clients()` orchestrates parallel client execution
- Clients are dynamically assigned to GPUs based on `--gpus` argument
- `BaseServer.__init__()` creates `self.client_gpu` mapping and `self.gpu_pools`
- Supports three parallel modes (`--parallel_mode`):
  - `sequential`: One client at a time
  - `stream`: One client per GPU simultaneously
  - `multi_stream`: Multiple clients per GPU (controlled by `--max_workers_per_gpu`)

**Key implementation details**:
- Model states are moved to CPU (`v.cpu()`) before passing to workers to avoid GPU memory conflicts
- Workers load models onto their assigned GPU device
- Results are collected and processed on the server

### Data Management

**Automatic Data Pipeline** (`src/data_gen/`):
- `prepare_data()` in `__init__.py` checks if preprocessed data exists
- If missing, calls dataset-specific `process_{dataset}.py` to download and partition
- Saves partitioned data to `datasets/{dataset}/{partition}/` directory
- `BaseServer` loads data via `load_data()` from `src/utils/load_data.py`

**Partitioning Strategies**:
- `iid`: Uniform random distribution
- `dirichlet`: Non-IID with Dirichlet distribution (parameter: `--alpha`)
- `pathological`: Each client has data from limited classes (parameter: `--n_classes`)

### Personalized FL Support

The framework distinguishes between:
- **Global FL** (e.g., FedAvg, MOON, FedPLN): `BaseServer(model, False, args)`
  - Single `self.test_set` for global evaluation
  - Model aggregation updates global model

- **Personalized FL** (e.g., FedDPL): `BaseServer(model, True, args)`
  - `self.test_set` is a list of per-client test sets
  - Server maintains `self.client_model_states` for local models
  - Only personalization network/prototypes are aggregated

### Model Architecture

Models defined in `src/models/`:
- `CNN`: Configurable input channels (1 for MNIST, 3 for CIFAR-10)
- `ResNet18`: Standard ResNet-18 architecture

**Important**: Models return `(output, feature)` tuple for prototype-based algorithms.

`get_model(model_name, dataset)` factory function handles model instantiation with correct configuration.

## Key Implementation Patterns

### Algorithm Extension

To add a new algorithm `newalgo`:

1. Create `src/newalgo.py` with:
   ```python
   def add_args(parser):
       group = parser.add_argument_group("NewAlgo Specific Arguments")
       group.add_argument("--param", type=float, default=1.0)
       return parser

   def client_worker(client_id, params):
       # Unpack params and implement local training
       return client_id, [loss, model.state_dict()]

   class Server(BaseServer):
       def __init__(self, args):
           model = get_model(args.model, args.dataset)
           super().__init__(model, personalized_flag, args)

       def fit(self):
           # Implement federated training rounds
           pass
   ```

2. Add to `main.py` choices: `choices=["fedavg", "fedavg_stream", "moon", "fedpln", "feddpl", "fedproto", "newalgo"]`

3. Create `scripts/newalgo.sh` for batch experiments

### Dataset Extension

To add a new dataset:

1. Create `src/data_gen/process_{dataset}.py` with:
   ```python
   def process(save_dir, num_clients, partition_method, alpha=0.5, n_classes=2):
       # Download, preprocess, partition, and save data
       pass
   ```

2. Update `prepare_data()` in `src/data_gen/__init__.py` to handle the new dataset

3. Add dataset to `main.py` choices: `choices=["mnist", "cifar10", "newdataset"]`

### Avoiding Common Pitfalls

- **Import statements**: Only use `import copy` if explicitly calling `copy.deepcopy()` in the module
- **GPU memory**: Always move shared model states to CPU before passing to parallel workers
- **Loss calculation**: Use incremental summation pattern to avoid list memory overhead:
  ```python
  total_loss = 0.0
  for res in results:
      total_loss += res[0]
  avg_loss = total_loss / num_clients
  ```
- **Parameter passing**: Client workers receive parameters as lists, not kwargs. Maintain consistent ordering.

## Results and Checkpoints

- Results saved to `results/{dataset}/{partition}/{algo}/` directory
- Test mode (`--test 1`): Saves to `test/` subdirectory
- Checkpoint format varies by algorithm (see individual `save()` methods)