# 多 GPU 并行联邦学习框架 (Unified Pipeline)

本项目是一个高效、可扩展的联邦学习研究框架，旨在简化不同算法在各种数据分布下的实验流程。

## 核心特性

- **一站式入口**: 所有的实验流程（数据下载、划分、训练、评估）均通过 `main.py` 统一管理。
- **自动数据管理**: 系统会自动检查数据集是否存在。若缺失，将自动调用 `src/data_gen` 中的逻辑进行下载、预处理和按需划分。
- **高性能模拟**: 内置多 GPU 并行支持，能够显著加速大规模客户端模拟过程，支持为客户端动态分配 GPU 资源。
- **算法解耦设计**: 算法实现与核心框架解耦，通过动态参数加载机制，支持轻松扩展新算法。
- **批量实验支持**: 提供 `run.sh` 脚本，支持通过环境变量进行多参数网格搜索和批量实验。

## 支持列表

### 📚 算法 (23种)

本项目目前支持以下主流及前沿联邦学习算法：

| 类别 | 算法 | 简介 | 论文/来源 |
|------|------|------|----------|
| **基础** | **FedAvg** | 联邦平均算法 | AISTATS 2017 |
| | **FedProx** | 针对异构数据的近端项优化 | MLSys 2020 |
| | **LG-FedAvg** | 本地/全局表示解耦 | arXiv 2020 |
| | **Scaffold** | 使用控制变量缓解客端偏移 | ICML 2020 |
| | **FedDyn** | 动态正则化联邦学习 | ICLR 2021 |
| **个性化** | **FedPer** | 个性化层（Base + Personalized Head） | arXiv 2019 |
| | **FedRep** | 学习表示（Representation Learning） | ICML 2021 |
| | **FedProto** | 基于原型的联邦学习 | AAAI 2022 |
| | **FedALA** | 自适应本地聚合 (Adaptive Local Aggregation) | AAAI 2023 |
| | **FedTGP** | 可训练全局原型 (Trainable Global Prototypes) | -- |
| | **FedPLN** | 原型学习网络 | -- |
| | **FedDPL** | 双重原型学习 | -- |
| | **FedDPL1** | 双重原型学习变体 | -- |
| | **FedFM** | 特征匹配联邦学习 | -- |
| | **FedProc** | 原型对比联邦学习 | -- |
| **蒸馏** | **FedKD** | 知识蒸馏 | -- |
| | **FML** | 联邦互学习 (Federated Mutual Learning) | -- |
| | **ProxyFL** | 代理模型互学习 | -- |
| **对比学习** | **MOON** | 模型对比学习 | CVPR 2021 |
| **语义/锚点** | **FedSA** | 语义锚点 (Semantic Anchors) | -- |
| | **FedLSA** | 位置感知语义锚点 | -- |
| **其他** | **FedTest** | 测试/实验性算法 | -- |
| | **Local** | 本地训练基准 (Baseline) | -- |

### 💾 数据集

- **MNIST**: 手写数字识别 (28x28, 1通道)
- **CIFAR-10**: 通用物体识别 (32x32, 3通道)
- **CIFAR-100**: 细粒度物体识别 (32x32, 3通道)
- **HAR**: 人类活动识别 (UCI HAR Dataset)
- **HAR-Feat**: 预处理的 HAR 特征数据

### 🧬 数据划分策略

- **IID**: 独立同分布，随机均匀划分。
- **Dirichlet (Non-IID)**: 基于狄利克雷分布划分 (`--alpha` 参数控制异构程度)。
- **Pathological (Non-IID)**: 病态非独立同分布，每个客户端只拥有有限个类别 (`--n_class` 参数控制)。

## 目录结构

```text
├── main.py              # 项目唯一运行入口，负责解析参数与调度
├── run.sh               # 批量实验启动脚本 (推荐入口)
├── src/                 # 核心代码
│   ├── data_gen/        # 数据处理脚本 (MNIST, CIFAR, HAR等)
│   ├── models/          # 模型架构定义 (CNN, ResNet18/50, HARCNN等)
│   ├── utils/           # 通用工具 (BaseServer, 并行化, 聚合, 评估)
│   ├── fedavg.py        # 算法实现示例
│   ├── fedproto.py      # ...
│   ├── fedala.py        # ...
│   ├── fedtgp.py        # ...
│   └── ...              # 其他算法文件
├── scripts/             # 算法专属运行脚本 (由 run.sh 调用)
└── datasets/            # 存储生成的持久化数据集
```

## 快速开始

### 1. 环境配置

本项目推荐使用 [uv](https://github.com/astral-sh/uv) 管理 Python 依赖：

```bash
uv sync
```

### 2. 运行单个实验

通过 `uv run main.py` 启动实验，可以灵活配置各项参数：

```bash
# 运行 FedAvg (Dirichlet 划分, alpha=0.5)
uv run main.py --algo fedavg --dataset mnist --partition dirichlet --alpha 0.5 --num_clients 10 --gpus 0

# 运行 FedALA (自定义参数)
uv run main.py --algo fedala --dataset cifar10 --partition dirichlet --alpha 0.1 \
    --eta 1.0 --rand_percent 80 --layer_idx 2
```

### 3. 运行批量实验 (推荐)

使用 `run.sh` 可以方便地配置多组参数进行批量实验。

```bash
# 编辑 run.sh 配置参数
# export ALGO="fedavg,fedproto,fedala"
# export DATASETS="mnist,cifar10"

# 启动实验
./run.sh
```

## 参数说明

实验参数分为以下几组：

- **通用参数**:
    - `--algo`: 算法名称 (如 `fedavg`, `fedproto`, `fedala`)。
    - `--dataset`: 数据集名称。
    - `--model`: 模型架构 (`cnn`, `resnet18`, `resnet50`, `harcnn` 等)。
    - `--partition`: 数据划分策略。
    - `--num_clients`: 客户端总数。
    - `--rounds`: 通信轮数。
    - `--epochs`: 本地训练轮数。
    - `--batch_size`: 批次大小。
    - `--lr`: 学习率。
    - `--gpus`: 使用的 GPU (如 `0,1`)。
    - `--mp`: 是否启用多进程 (1=开启, 0=关闭)。

- **算法特定参数**:
    - **FedProto**: `--mu` (原型损失权重)
    - **FedALA**: `--eta`, `--rand_percent`, `--layer_idx`
    - **FedTGP**: `--lamda` (TGP损失权重), `--server_epochs`
    - (更多参数请参考各算法源码中的 `add_args` 函数)

## 扩展指南

### 添加新算法
1. 在 `src/` 目录下创建算法文件 (例如 `src/new_algo.py`)。
2. 实现 `add_args(parser)` 注册参数。
3. 实现 `client_worker(params)` 定义客户端行为。
4. 实现 `Server(BaseServer)` 定义服务器行为。
5. 在 `main.py` 的算法列表中注册新算法名。
6. (可选) 在 `scripts/` 下创建脚本并在 `run.sh` 中添加支持。

---
*本项目旨在为联邦学习研究提供一个高效且易于扩展的实验平台。*
