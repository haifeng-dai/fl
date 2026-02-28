# Gemini 代码理解报告 & 项目开发指南

**生成日期:** 2026-02-28
**类型:** 联邦学习研究框架 (Unified Pipeline)
**语言:** 中文 (Chinese)
**技术栈:** Python 3.14+, PyTorch 2.6+, CUDA 13.0, uv

---

## 1. 项目概览

本项目是一个具有统一流水线的高效多 GPU 并行联邦学习框架。它旨在简化不同算法（目前支持 23 种）在各种数据分布策略下的实验流程。

### 核心特性
*   **统一入口 (One-Stop Entry)**：所有实验（数据准备、训练、评估）均由 `main.py` 统一调度。
*   **自动数据管理**: 自动检测、下载、预处理和划分数据集（MNIST, CIFAR-10/100, HAR, Flowers102 等）。
*   **算法解耦**: 通过 `src/<algo>.py` 中的 `add_args` 动态加载算法特定参数。
*   **高性能模拟**: 基于 `torch.multiprocessing` 的多 GPU 并行架构，支持为每个客户端动态分配 GPU 资源。

---

## 2. 常用命令

### 环境配置与规范检查
```bash
# 安装依赖
uv sync

# 代码格式化 (Black)
uv run black src/

# 类型检查 (BasedPyright)
uv run basedpyright src/
```

### 运行实验
```bash
# 运行批量实验
bash ./run.sh
```

### 结果分析
```bash
# 使用 Jupyter Notebook 分析实验结果
uv run jupyter notebook results_analysis.ipynb
```

---

## 3. 架构设计

### 目录结构
*   `main.py`: 中央调度器。负责解析参数、触发数据准备、初始化 Server 并运行。
*   `src/`: 核心代码库。
    *   `<algo>.py`: 算法实现 (如 `fedavg.py`, `fedproto.py` 等)。
    *   `utils/`: 包含 `fed_utils.py` (BaseServer), `parallel.py` (并行驱动), `load_data.py` 等。
    *   `models/`: 模型定义 (CNN, ResNet18/50, HAR 等)，**必须返回 `(logits, features)` 元组**。
    *   `data_gen/`: 数据集处理与分区逻辑。
*   `scripts/`: 存放各算法的嵌套循环参数搜索脚本。
*   `datasets/`: 持久化存储处理后的数据。

### 关键流程
1.  **启动**: `main.py` 确定算法 -> 动态导入 `src/{algo}.py` -> 加载通用及算法专属参数。
2.  **数据**: 检查 `datasets/` -> 调用 `src.data_gen` 生成分区数据。
3.  **并行执行**: `BaseServer` 建立 `client_gpu` 映射 -> `run_parallel_clients` 分发任务到进程池 -> 收集并保序返回结果。

---

## 4. 开发规范与模式

### 算法实现模式
每个算法文件 (`src/<algo>.py`) 必须导出以下组件：
1.  `add_args(parser)`: 注册算法超参。
2.  `client_worker(params)`: 客户端训练逻辑。必须接收列表参数，返回 `[avg_loss, model_state, ...]`。
3.  `Server(BaseServer)`: 服务器聚合与协调逻辑。

### 命名与类型规范
*   **命名**: 类使用 `PascalCase`，函数/变量使用 `snake_case`，常量使用 `UPPER_SNAKE_CASE`。
*   **类型注解**: 强烈推荐使用 Python 3.14+ 风格（如 `list[float] | None`）。
*   **导入**: 内部模块使用相对导入 (如 `from .utils import ...`)。

### 性能陷阱 (Critical Performance Tips)
*   **Dataset 切片**: **严禁直接切片** `dataset[0:10]`。必须使用 `torch.utils.data.Subset(dataset, indices)`。
*   **GPU 循环**: 避免在训练循环中对 batch 内样本进行 Python 循环和 `to(device)`，必须使用 Tensor 向量化操作。
*   **模型复制**: 在 worker 进程中，优先使用 `get_model()` + `load_state_dict()`，避免 `copy.deepcopy()` 的序列化开销。
*   **资源回收**: 显式调用 `server.close()`（已集成在 `BaseServer` 中）以释放 GPU 进程池。

---

## 5. 支持算法 (23种)

*   **基础 (Base / pfl=False)**: FedAvg, FedProx, Scaffold, FedDyn, MOON, FedSA, FedLSA, FedPLN, FedProc, FedFM
*   **个性化 (Personalized / pfl=True)**: FedPer, FedRep, FedProto, FedALA, FedTGP, FedDPL, FedDPL1, LG-FedAvg, FedKD, FML, ProxyFL, FedTest, Local (Baseline)

---

## 6. 常见问题 (Pitfalls)

*   不要直接对数据集进行索引。
*   不要在循环中进行 GPU 内存搬运。
*   不要使用 `as any` 或 `@ts-ignore` 压制类型错误，应正确修复。
*   在 pFL 算法中，不要聚合模型参数，应维护 `self.clients_state`。
