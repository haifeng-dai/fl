# 多 GPU 并行联邦学习框架 (统一流水线)

本项目是一个高效、可扩展的联邦学习研究框架，旨在简化不同算法在各种数据分布下的实验流程。

## 核心特性

- **一站式入口**: 所有的实验流程（数据下载、划分、训练、评估）均通过 `main.py` 统一管理。
- **自动数据管理**: 系统会自动检查数据集是否存在。若缺失，将自动调用 `src/data_gen` 中的逻辑进行下载、预处理和按需划分。
- **高性能模拟**: 内置多 GPU 并行支持，能够显著加速大规模客户端模拟过程，支持为客户端动态分配 GPU 资源。
- **算法解耦设计**: 算法实现与核心框架解耦，通过动态参数加载机制，支持轻松扩展新算法，同时保持通用的联邦学习与数据配置参数。

## 目录结构

```text
├── main.py              # 项目唯一运行入口，负责解析参数与调度
├── src/                 # 核心代码
│   ├── data_gen/        # 数据处理脚本
│   │   ├── __init__.py      # 数据准备核心接口与统一划分逻辑
│   │   └── process_mnist.py # MNIST 数据集特定处理逻辑
│   ├── fedavg.py        # FedAvg 算法实现
│   ├── moon.py          # MOON 算法实现
│   └── utils/           # 通用工具函数 (聚合、评估、并行化等)
├── models/              # 模型架构定义 (如 CNN 等)
├── datasets/            # 存储生成的持久化数据集
└── run_demo.sh          # 快速演示脚本
```

## 快速开始

### 环境配置

本项目推荐使用 [uv](https://github.com/astral-sh/uv) 管理 Python 依赖：

```bash
uv sync
```

### 运行演示

执行以下脚本快速启动一个默认配置的实验：

```bash
./run_demo.sh
```

### 运行实验示例

通过 `uv run main.py` 启动实验，可以灵活配置各项参数：

#### 1. 运行 FedAvg (Dirichlet 划分)
```bash
uv run main.py --algo fedavg --dataset mnist --partition dirichlet --alpha 0.5 --num_clients 10 --gpus 0,1
```

#### 2. 运行 MOON (病态划分/Pathological)
```bash
uv run main.py --algo moon --dataset mnist --partition pathological --n_classes 2 --num_clients 10 --gpus 0
```

## 参数说明

实验参数分为以下几组：

- **数据与划分 (Data & Partitioning)**:
    - `--dataset`: 使用的数据集名称 (如 `mnist`)。
    - `--partition`: 数据划分策略 (`iid`, `dirichlet`, `pathological`)。
    - `--num_clients`: 客户端总数。
    - `--alpha`: Dirichlet 划分的浓度参数。
    - `--n_classes`: 每个客户端拥有的类别数 (用于 pathological 划分)。
- **训练配置 (Training)**:
    - `--rounds`: 联邦学习通信轮数。
    - `--epochs`: 客户端本地训练轮数。
    - `--lr`: 学习率。
    - `--gpus`: 指定使用的 GPU 索引 (如 `0,1,2`)。
- **算法特定参数 (Algorithm Specific)**:
    - 视具体算法而定 (如 MOON 的 `--mu`)。

## 扩展指南

### 添加新数据集
1. 在 `src/data_gen/` 目录下创建 `process_xxx.py`。
2. 实现 `process()` 函数，负责数据的下载与持久化存储。

### 添加新算法
1. 在 `src/` 目录下创建算法文件。
2. 实现 `Server` 和 `Client` 类。
3. 实现 `add_args` 函数以定义算法专属的命令行参数。

---
*本项目采用模块化设计，旨在为联邦学习研究提供一个高效且易于扩展的实验平台。*
