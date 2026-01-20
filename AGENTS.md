# PROJECT KNOWLEDGE BASE

**Generated:** 2026-01-20
**Type:** Federated Learning Research Framework
**Stack:** Python 3.14, PyTorch 2.6+, CUDA 13.0, uv

## OVERVIEW

Federated learning framework with multi-GPU parallel training support. Implements 5+ FL algorithms (FedAvg, MOON, FedProto, FedPLN, FedDPL) with IID/Dirichlet/Pathological data partitioning.

## STRUCTURE

```
./
├── main.py              # Entry: two-phase arg parsing → dynamic algo load
├── run.sh               # Shell orchestrator → scripts/*.sh dispatch
├── pyproject.toml       # uv + BasedPyright (standard mode)
├── src/                 # Core package (18 .py files)
│   ├── <algo>.py        # fedavg, moon, fedproto, fedpln, feddpl
│   ├── models/          # CNN, ResNet18 (return tuple: logits, features)
│   ├── utils/           # BaseClient, BaseServer, parallel, aggregate
│   └── data_gen/        # Dataset prep (MNIST, CIFAR-10)
├── scripts/             # Grid search shell scripts (12-13 nested loops)
├── test/                # Manual test scripts (no pytest)
└── datasets/            # Generated partition data
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Add algorithm | `src/<algo>.py` | Inherit BaseClient/BaseServer, export `add_args`, `Client`, `Server` |
| Add model | `src/models/` | Must return `(logits, features)` tuple |
| Multi-GPU training | `src/utils/fed_utils.py:94-116` | `__start_pools()` handles GPU allocation |
| Data partitioning | `src/data_gen/__init__.py` | **CRITICAL BUG: line 149 uses wrong import path** |
| Parallel execution | `src/utils/parallel.py` | `run_parallel_clients()` |

## CONVENTIONS

- **Imports**: stdlib → third-party → local (blank lines between)
- **Type hints**: Required, use `list[type]`, `float | None`
- **Naming**: PascalCase (classes), snake_case (funcs/vars), UPPER_SNAKE_CASE (const)
- **Formatting**: Black (88 char), 4 spaces, 2 blank lines (top-level)
- **Multi-GPU**: Must call `set_start_method("spawn", force=True)` in main
- **Cleanup**: Always call `server.close()` in finally block

## ANTI-PATTERNS (THIS PROJECT)

- **CRITICAL BUG**: `src/data_gen/__init__.py:149` → `data_scripts.process_X` should be `src.data_gen.process_X`
- **Documentation mismatch**: README.md references non-existent `data_scripts/`
- **Code duplication**: `fedpln.py` and `feddpl.py` are ~95% identical (copy-paste inheritance)
- **Commented debug code**: `moon.py` lines 97, 110-112, 120-122 contain debug code
- **Magic numbers**: CNN (7*7, 8*8, 128, 64), ResNet (32, 32, 128), PLN init values (-0.1, 0.1, 0.01)
- **Hardcoded paths**: `./datasets/`, `./results/` scattered across data_gen/__init__.py, load_data.py, fed_utils.py
- **Inconsistent device management**: Mixed `.to(device)` patterns, redundant deep copies in evaluate.py
- **Unused import**: `feddpl.py:2` imports `cli` from pydoc (never used)
- **Inconsistent return values**: Client.train() returns scalar (FedAvg) vs tuple (FedProto, FedPLN)
- **Resource leak risk**: `__del__()` cleanup in fed_utils.py is unreliable; use try/finally instead
- **No test framework**: Manual testing only via `uv run main.py --test`
- **Deep nested loops**: `scripts/fedpln.sh` has 12-13 nested for loops (extreme)

## COMMANDS

```bash
# Setup
uv sync

# Run experiment
uv run main.py --algo fedavg --dataset mnist --gpus 0,1

# Grid search
./run.sh

# Format
uvx black .

# Type check (via editor - BasedPyright configured in pyproject.toml)
```

## NOTES

- Models MUST return `(logits, features)` tuple for contrastive algorithms
- Use `.to(device)` for ALL tensors in training loops
- Data auto-prepares if missing (calls `process_<dataset>()`)
- Results saved to `results/<dataset>_<partition>_<num_clients>/<epochs>_<batch_size>_<lr>.pt`
