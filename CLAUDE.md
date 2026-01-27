# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 提供项目操作指南。

## 项目概览

这是一个支持 **17 种算法**（包括 FedAvg, FedProto, FedALA, FedTGP 等）的多 GPU 并行联邦学习框架（Unified Pipeline）。该框架集成了自动数据管理和高性能多 GPU 客户端模拟。

## 常用命令

### 环境配置
```bash
# 使用 uv 安装依赖
uv sync
```

### 运行实验

**主要入口**：所有实验均通过 `main.py` 运行。

```bash
# 基础 FedAvg (Dirichlet 划分)
uv run main.py --algo fedavg --dataset mnist --partition dirichlet --alpha 0.5 --num_clients 10 --gpus 0,1

# 运行 FedALA (自定义参数)
uv run main.py --algo fedala --dataset cifar10 --eta 1.0 --rand_percent 80

# 运行所有配置好的实验 (推荐)
./run.sh
```

### 算法专属脚本

`scripts/` 目录下存放了各算法的专用脚本：
- `scripts/fedavg.sh`, `scripts/moon.sh`, `scripts/fedpln.sh`
- `scripts/fedala.sh` (新增), `scripts/fedtgp.sh` (新增)
- ... (以及其他)

这些脚本支持通过 `run.sh` 设置环境变量来进行**嵌套循环参数搜索**。

### 代码格式化
```bash
uv run black .
```

### 结果分析
```bash
# 使用 Jupyter Notebook 分析实验结果
uv run jupyter notebook results_analysis.ipynb
```

## 架构说明

### 入口流程

1.  **`main.py`** - 统一入口：
    - 第一阶段参数解析：确定算法 (`--algo`)。
    - 动态导入算法模块 `src/{algo}.py`。
    - 第二阶段参数解析：加载通用参数 + 算法专属参数 (`add_args`)。
    - 自动触发数据准备 `src.data_gen.prepare_data()`。
    - 实例化 `Server` 并运行 `fit()`。

### 核心架构模式

所有算法遵循一致的结构：

**算法模块** (`src/{algorithm}.py`):
- `add_args(parser)`: 注册参数。
- `client_worker(params)`: 并行客户端训练函数。
  - **必须** 接收参数列表 (List)。
  - **建议** 返回 `client_id, [loss, model_state...]` 以确保索引安全。
- `Server(BaseServer)`: 服务器类。

**客户端 Worker**:
- 接收参数列表（GPU 设备, 模型状态, 数据集, 超参）。
- 使用 `get_model(model_name, dataset_name)` 实例化模型。
  - **优化**: 使用 `get_model` + `load_state_dict` 替代 `copy.deepcopy` 以提升多进程性能。
- 执行本地训练。
- 返回结果。

**Server 类**:
- `fit()`: 主循环。
  - 使用 `run_parallel_clients` 执行并行训练。
  - **关键**: 更新 `self.clients_state` 时，必须使用 `selected_clients[i]` 作为索引，严禁直接使用 `i`。

### 多 GPU 并行

**GPU 分配** (`src/utils/parallel.py`):
- `run_parallel_clients()`: 编排并行执行。
- **顺序保证**: 返回结果列表的顺序严格对应输入参数列表（即 `selected_clients` 的顺序）。
- `BaseServer.__init__()` 负责创建 `self.client_gpu` 映射和进程池。

### 数据管理

- `src/data_gen/__init__.py`: 统一数据入口。
- `src/utils/load_data.py`: `BaseServer` 加载数据的接口。
- 支持策略: `iid`, `dirichlet` (`--alpha`), `pathological` (`--n_classes`)。

### 扩展指南

**添加新算法**:
1. 创建 `src/newalgo.py`。
2. 在 `main.py` 的 `choices` 中添加新算法名。
3. 创建 `scripts/newalgo.sh`。
4. 在 `run.sh` 中添加配置和执行分支。

### 性能最佳实践 (Performance Tips)

- **DataLoader 切片**: 严禁直接对 Dataset 进行切片 `dataset[0:100]`，这通常会返回 Tensor 元组导致 DataLoader 崩溃。**必须**使用 `torch.utils.data.Subset(dataset, indices)`。
- **向量化操作**: 避免在训练循环中对 batch 内的样本进行 Python 循环和 GPU 传输（如原型匹配）。应使用 Tensor 向量化索引。
- **模型复制**: 在 worker 进程中，优先使用 `get_model()` + `load_state_dict()`，避免 `copy.deepcopy()` 的序列化开销。

## 支持算法 (17种)

FedAvg, FedAvg Stream, MOON, FedPLN, FedDPL, FedProto, FedKD, FML, ProxyFL, FedPer, FedProx, FedSA, FedLSA, LG-FedAvg, FedRep, FedALA (New), FedTGP (New).
