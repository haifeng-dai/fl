# Gemini 代码理解报告

**生成日期:** 2026-01-27
**类型:** 联邦学习研究框架 (Unified Pipeline)
**语言:** 中文 (Chinese)

## 项目概览

本项目是一个具有统一流水线的高效多 GPU 并行联邦学习框架。它旨在简化不同算法（目前支持 17 种）在各种数据分布策略下的实验流程。

**核心特性：**

*   **统一入口 (One-Stop Entry)**：所有实验（数据准备、训练、评估）均由 `main.py` 统一调度。
*   **自动数据管理**: 自动检测、下载、预处理和划分数据集（MNIST, CIFAR-10, HAR 等）。
*   **算法解耦**: 通过 `src/<algo>.py` 中的 `add_args` 动态加载算法特定参数，实现核心框架与算法逻辑的解耦。
*   **高性能模拟**: 基于 `torch.multiprocessing` 的多 GPU 并行架构，支持为每个客户端动态分配 GPU 资源。

## 技术栈

*   **语言**: Python 3.14
*   **深度学习**: PyTorch 2.6+
*   **依赖管理**: `uv`
*   **并行计算**: CUDA 13.0

## 架构设计

### 目录结构

*   `main.py`: 中央调度器。负责两阶段参数解析（通用+算法特定）、触发数据准备、初始化 Server 并运行。
*   `run.sh`: 批量实验编排脚本。通过环境变量控制参数，调用 `scripts/` 下的脚本。
*   `src/`: 核心代码库。
    *   `fedavg.py`, `fedproto.py`, `fedala.py`, `fedtgp.py`: 具体算法实现。
    *   `utils/`: 通用工具。
        *   `fed_utils.py`: `BaseServer` 基类，处理 GPU 进程池初始化。
        *   `parallel.py`: `run_parallel_clients` 函数，核心并行驱动，保证结果保序性。
        *   `load_data.py`: 数据加载接口。
    *   `models/`: 模型定义 (`cnn.py`, `resnet.py`)，**必须**返回 `(logits, features)` 元组。
    *   `data_gen/`: 数据集处理逻辑。
*   `datasets/`: 持久化存储处理后的数据。

### 关键流程

1.  **启动**: `main.py` 解析 `--algo` -> 导入 `src/{algo}.py` -> 解析完整参数。
2.  **数据**: 检查 `datasets/` -> 若缺失则调用 `src.data_gen` -> 生成分区数据。
3.  **初始化**: `Server` 初始化模型 -> `BaseServer` 建立 `client_gpu` 映射和进程池。
4.  **训练**: `Server.fit()` 循环 -> `run_parallel_clients` 分发任务到 Worker -> 收集结果 -> 聚合更新。

## 支持算法 (17种)

1.  **基础**: FedAvg, FedProx, LG-FedAvg, FedAvg Stream
2.  **个性化**: FedPer, FedRep, FedProto, FedALA (New), FedTGP (New), FedPLN, FedDPL
3.  **蒸馏**: FedKD, FedAMD, FML, ProxyFL
4.  **其他**: MOON (对比学习), FedSA/FedLSA (语义锚点)

## 开发规范

### 添加新算法

1.  在 `src/` 下新建 `.py` 文件。
2.  定义 `add_args(parser)`: 注册超参。
3.  定义 `client_worker(params)`: 客户端训练逻辑（**必须**接收列表参数，建议返回 client_id）。
4.  定义 `Server(BaseServer)`: 服务器聚合逻辑（**注意**: 更新状态时必须使用 `selected_clients[i]` 索引）。

### 性能陷阱 (Critical Performance Tips)

*   **Dataset 切片**: **严禁**使用 `dataset[start:end]` 直接切片（会导致 DataLoader 异常）。**必须**使用 `torch.utils.data.Subset(dataset, indices)`。
*   **GPU 循环**: 避免在训练循环中对 batch 内的每个样本进行 Python 循环和 `to(device)` 操作（例如原型匹配）。这会造成 GPU 流水线严重阻塞（"卡死"现象）。**必须**使用 Tensor 向量化操作。
*   **进程间通信**: 在 worker 中优先使用 `get_model()` + `load_state_dict()` 重新创建模型，这比 `copy.deepcopy()` 在多进程下更高效且兼容性更好。

## 运行示例

```bash
# 安装依赖
uv sync

# 运行单个实验
uv run main.py --algo fedavg --dataset mnist --gpus 0,1

# 运行批量实验 (FedALA & FedTGP)
./run.sh
```
