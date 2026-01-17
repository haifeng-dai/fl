# Gemini 代码理解报告

## 项目概览

本项目是一个具有统一流水线的多 GPU 并行联邦学习框架。它旨在通过不同的算法和数据划分策略进行联邦学习实验。

**核心特性：**

*   **统一入口点：** 所有实验都通过 `main.py` 运行，集成了数据准备、客户端分配和训练流程。
*   **自动数据管理：** 框架会自动检测数据。如果 `datasets/` 目录下缺失数据，会自动调用 `data_scripts` 中的逻辑进行下载、预处理和划分。
*   **算法与参数解耦：** 框架支持动态加载算法特定的参数。通过在算法脚本中定义 `add_args` 函数，实现通用参数与算法参数的隔离。
*   **高性能模拟：** 支持多 GPU 并行进行客户端训练。每个客户端被分配到一个固定的 GPU，以最大化计算效率。

**技术栈：**

*   **语言：** Python
*   **核心库：** PyTorch (深度学习), NumPy (数值计算)
*   **依赖管理：** `uv` (现代 Python 包管理器)

**架构设计：**

*   `main.py`：项目的中心枢纽，负责参数解析、数据加载触发、模型初始化以及联邦学习轮次的调度。
*   `src/`：存放联邦学习核心算法。
    *   `fedavg.py`：标准的联邦平均算法。
    *   `moon.py`：基于模型对比学习的联邦学习算法。
    *   `fedpln.py`：(待补充具体描述) 框架支持的另一种联邦学习算法。
    *   `utils/`：包含聚合逻辑 (`aggregate.py`)、评估指标 (`evaluate.py`)、通用联邦工具 (`fed_utils.py`)、数据加载 (`load_data.py`) 和并行执行驱动 (`parallel.py`)。
*   `models/`：定义神经网络架构，如 `cnn.py`。
*   `data_scripts/`：数据集特定的处理脚本，如 `process_mnist.py`。
*   `datasets/`：持久化存储原始数据及划分后的客户端数据。

## 构建与运行

### 环境配置

项目使用 `uv` 进行依赖管理。请确保已安装 `uv`，然后运行：

```bash
uv sync
```

### 运行实验

使用 `uv run main.py` 启动实验。

**示例 1：使用 Dirichlet 划分运行 FedAvg**
```bash
uv run main.py --algo fedavg --dataset mnist --partition dirichlet --alpha 0.5 --num_clients 10 --gpus 0,1
```

**示例 2：使用病态划分运行 MOON**
```bash
uv run main.py --algo moon --dataset mnist --partition pathological --n_classes 2 --num_clients 10 --gpus 0
```

## 开发规范

### 添加新数据集

1.  在 `data_scripts/` 目录中创建一个新的 `process_xxx.py` 文件。
2.  在该文件中实现一个 `process()` 函数。该函数应下载数据、执行预处理（如归一化），并根据 `data_scripts/__init__.py` 中的逻辑保存划分后的数据。

### 添加新算法

1.  在 `src/` 目录中创建一个新的 Python 文件。
2.  实现 `Server` 类（负责全局聚合和调度）和 `Client` 类（负责本地训练）。
3.  实现一个 `add_args(parser)` 函数，用于向全局解析器添加该算法特有的命令行参数。
4.  在 `main.py` 中确保该算法能被动态识别（通常通过文件名或显式注册）。

### 性能建议

*   **GPU 分配：** 通过 `--gpus` 参数传入可用的 GPU 索引列表。框架会自动在客户端之间负载均衡。
*   **多进程：** 客户端训练是在独立的进程中运行的，确保本地训练代码是线程/进程安全的。