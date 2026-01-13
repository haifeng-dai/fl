# Multi-GPU Parallel Federated Learning Framework (Unified Pipeline)

## 特性
- **一站式入口**: 所有的实验（包括数据下载、切分、训练）全部通过 `main.py` 统一触发。
- **自动数据管理**: `main.py` 会自动检查数据是否存在。如果不存在，它会自动调用 `data_scripts` 中的逻辑进行合并、预处理和划分。
- **算法与参数解耦**: 支持动态加载算法特有参数，并保持通用的联邦学习与数据参数。
- **高性能模拟**: 支持多 GPU 并行，每个客户端固定 GPU 分配。

## 目录结构
- `main.py`: 项目唯一的运行入口。
- `data_scripts/`
    - `__init__.py`: **核心数据准备接口**，包含统一的划分逻辑（IID, Dirichlet, Pathological）。
    - `process_mnist.py`: MNIST 数据集特有的处理脚本。
- `src/`: 核心算法与模型定义。
- `datasets/`: 存储生成的持久化数据。

## 使用方法

### 环境配置
项目使用 `uv` 管理依赖。首先安装依赖：
```bash
uv sync
```

### 快速运行脚本
项目提供了一个便捷的运行脚本 `run_demo.sh`（内部使用 `uv run`）：
```bash
./run_demo.sh
```

### 直接运行实验
使用 `uv run` 启动 `main.py`：

#### 运行 FedAvg (Dirichlet 划分)
```bash
uv run main.py --algo fedavg --dataset mnist --partition dirichlet --alpha 0.5 --num_clients 10 --gpus 0,1
```

#### 运行 MOON (病态分布)
```bash
uv run main.py --algo moon --dataset mnist --partition pathological --n_classes 2 --num_clients 10 --gpus 0
```

### 参数分组说明
- **Data & Partitioning**: `--dataset`, `--partition`, `--num_clients`, `--alpha`, `--n_classes`。
- **Training**: `--rounds`, `--epochs`, `--lr`, `--gpus`。
- **Algorithm Specific**: 根据 `--algo` 动态显示（如 MOON 的 `--mu`）。

## 扩展指南
- **新数据集**: 在 `data_scripts/` 下创建 `process_xxx.py` 并实现 `process()`。
- **新算法**: 在 `src/` 下创建新脚本，实现 `Server` 和 `Client` 类以及 `add_args` 函数。