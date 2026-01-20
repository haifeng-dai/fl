# src/ KNOWLEDGE BASE

**Parent:** ./AGENTS.md

## OVERVIEW

Core package containing 18 Python files: 5 algorithm implementations, 2 models, 5 utilities, 3 data generators.

## STRUCTURE

```
src/
├── <algo>.py          # FedAvg, MOON, FedProto, FedPLN, FedDPL
├── models/
│   ├── cnn.py         # CNN → (logits, features)
│   └── resnet.py      # ResNet18 → (logits, features)
├── utils/
│   ├── fed_utils.py   # BaseClient, BaseServer
│   ├── parallel.py    # run_parallel_clients()
│   ├── aggregate.py   # param_aggregate()
│   ├── evaluate.py    # evaluate_model()
│   └── load_data.py   # load_data()
└── data_gen/
    ├── __init__.py    # prepare_data() → CRITICAL BUG line 149
    ├── process_mnist.py
    └── process_cifar10.py
```

## WHERE TO LOOK

| Task | File | Key |
|------|------|-----|
| Implement FL algorithm | `<algo>.py` | Inherit BaseClient/BaseServer |
| Multi-GPU pool setup | `utils/fed_utils.py:94` | `__start_pools()` |
| Data partitioning | `data_gen/__init__.py` | `iid_partition()`, `dirichlet_partition()`, `pathological_partition()` |
| Parallel training | `utils/parallel.py` | `run_parallel_clients()` |
| Model aggregation | `utils/aggregate.py` | `param_aggregate()` |

## ALGORITHM PATTERN

Each `<algo>.py` MUST export:

```python
def add_args(parser): ...
class Client(BaseClient): ...
class Server(BaseServer): ...
```

**Client required methods:**
- `train() -> float`: Returns avg loss
- `set_client(parameters)`: Loads model params

**Server required methods:**
- `fit()`: Communication round loop
- `save(test)`: Saves results

## MODEL SIGNATURE (CRITICAL)

All models MUST return tuple:

```python
def forward(self, x):
    h = self.features(x)
    z = self.proj(h)
    y = self.fc(z)
    return y, z  # (logits, features) - REQUIRED
```

## DATA GEN PATTERN

`process_<dataset>.py` must implement `process(raw_dir)`:
- Downloads raw data
- Saves to `datasets/raw/<dataset>_raw.pt` with keys: `{"x", "y", "num_classes"}`

## CONVENTIONS (OVERRIDES)

- **No override** - follow parent AGENTS.md conventions
- **Import order**: stdlib → torch → `.utils`, `src.models`
