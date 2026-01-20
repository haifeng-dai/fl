# AGENTS.md

This file contains guidelines and commands for agentic coding assistants working in this federated learning framework repository.

## Build / Lint / Test Commands

### Running Experiments
- **Quick demo**: `./run.sh` (uses environment variables in the script)
- **Direct run**: `uv run main.py --algo fedavg --dataset mnist --partition dirichlet --alpha 0.5 --num_clients 10 --gpus 0,1`
- **Example algorithms**: fedavg, moon, fedpln, feddpl, fedproto

### Setup & Dependencies
- **Install dependencies**: `uv sync`
- **Update dependencies**: `uv sync --upgrade`

### Code Quality
- **Format code**: `uvx black .` (Black formatter with default 88 char line length)
- **Type checking**: BasedPyright configured with `typeCheckingMode: "standard"` in pyproject.toml

### Testing
- No automated test framework (pytest/unittest) configured
- Manual testing via: `uv run main.py [args]` with `--test` parameter (defaults to True)
- Custom test files exist in `test/` directory but are run manually

## Code Style Guidelines

### File Structure & Organization
- Entry point: `main.py` (parsing and orchestration only)
- Algorithms: `src/<algo_name>.py` (e.g., `fedavg.py`, `moon.py`)
- Each algorithm file exports:
  - `add_args(parser: argparse.ArgumentParser)`: Adds algorithm-specific CLI arguments
  - `Server` class: Inherits from `BaseServer`
  - `Client` class: Inherits from `BaseClient`
- Models: `src/models/` (CNN, ResNet18, etc.)
- Utilities: `src/utils/` (BaseClient, BaseServer, aggregation, evaluation, parallelization)
- Data processing: `src/data_gen/` (dataset-specific processing logic)

### Imports
Order: standard library → third-party → local modules (separated by blank lines)
```python
import argparse
import os

import torch
import torch.multiprocessing as mp

from .utils import BaseClient, BaseServer
from src.models import CNN
```

### Type Hints
- Required on function signatures
- Use `list[type]`, `dict[key_type, value_type]` syntax (Python 3.9+)
- Use `|` for unions (Python 3.10+): `float | None`
- Example: `def train(self) -> float:`

### Naming Conventions
- **Classes**: PascalCase (e.g., `BaseClient`, `FedAvgServer`)
- **Functions/Methods**: snake_case (e.g., `train()`, `aggregate()`)
- **Variables**: snake_case (e.g., `client_id`, `num_clients`)
- **Constants**: UPPER_SNAKE_CASE
- **Private methods**: single underscore prefix (e.g., `_start_pools()`)

### Code Formatting
- Black formatter with 88 character line limit
- 4 spaces for indentation
- 2 blank lines between top-level definitions
- 1 blank line between method definitions
- Comments in English or Chinese (both are acceptable)

### Error Handling
- Use explicit error messages with context
- Raise `ValueError` for invalid arguments
- Use try/finally for resource cleanup (e.g., closing GPU pools)

### Algorithm Implementation Pattern
```python
def add_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("<Algo> Specific Arguments")
    group.add_argument("--param", type=float, default=1.0, help="Description")
    return parser

class Client(BaseClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize algo-specific attributes

    def train(self):
        # Training loop
        return avg_loss

    def set_client(self, parameters):
        # Load parameters into model

class Server(BaseServer):
    def __init__(self, model, args):
        super().__init__(model, pfl_flag, args)
        # Initialize clients

    def fit(self):
        for r in range(self.rounds):
            # Communication round
            self.evaluate()

    def save(self, test):
        # Save results
```

### Multi-GPU Parallel Training
- `run_parallel_clients()` handles both parallel and sequential execution
- Use `--no_mp` flag to disable multiprocessing (useful for debugging)
- GPU allocation handled automatically by `BaseServer.__start_pools()`

### Data Partitioning
- Supported strategies: `iid`, `dirichlet`, `pathological`
- Automatically prepares data if missing via `prepare_data()`
- Data stored in `datasets/` directory

### Result Storage
- Saved to `results/<dataset>_<partition>_<num_clients>/` path
- Format: `results/<path>/<epochs>_<batch_size>_<lr>.pt`
- Contains: `acc`, `loss`, `state_dict`

### Important Notes
- Must use `torch.multiprocessing.set_start_method("spawn", force=True)` in main
- Always call `server.close()` in finally block to release GPU resources
- Models return tuple `(logits, features)` for algorithms requiring embeddings
- Use `.to(device)` for all tensors in training loops
