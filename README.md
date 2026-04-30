# Unified Federated Learning Framework (Ray Powered)

本项目是一个高性能、一站式的多 GPU 并行联邦学习研究框架。它集成了 23 种主流联邦学习算法，并采用 **Ray** 分布式计算后端，为大规模客户端模拟（尤其是 ResNet 等深度模型）提供工业级的稳定性与效率。

## 🚀 快速开始

### 1. 环境准备

推荐使用 `uv` 进行依赖管理：

```bash
git checkout ray  # 确保在 Ray 分支
uv sync
```

### 2. 运行实验

现在系统采用 **Config-First** 架构，所有参数优先从 YAML 加载，并支持命令行动态覆盖。

```bash
# 基本运行 (使用 configs/default.yaml 中的默认配置)
uv run main.py --algo fedavg

# 大规模并行运行 (指定 4 块 GPU, 每块卡跑 3 个并发, 使用 ResNet18)
uv run main.py --algo fedprox --model resnet18 --dataset cifar100 --gpus 0,1,2,3 --max_workers_per_gpu 3

# 快速测试模式 (不记录文件，直接输出到终端)
uv run main.py -a fedavg -t 1
```

## 🛠️ 核心参数说明

- `-a, --algo`: 算法名称（支持 FedAvg, FedProx, Scaffold, FedRep, FedALA 等 23 种）。
- `--model`: 模型架构（支持 cnn, resnet18, resnet50, harcnn 等）。
- `--dataset`: 数据集（支持 cifar10/100, mnist, tiny_imagenet 等）。
- `--max_workers_per_gpu`: **关键资源参数**。每块 GPU 上同时运行的 Worker 数量（如 2 表示单卡 2 并行）。
- `--gpus`: 指定物理 GPU 编号（如 `0,1,2,3`）。
- `-t, --test 1`: 开启测试模式，日志将直接输出到终端而非文件。
- `-r, --num_runs`: 覆盖实验重复次数（对应 YAML 中的 `times`）。

## 📂 配置与结果管理

- **全局默认配置**: `configs/default.yaml`
- **算法专属配置**: `configs/algorithms.yaml`（支持超参数搜索，只需将参数设为列表即可自动展开）。
- **实验结果存储**: `results_ray/`
- **运行日志存储**: `logs_ray/`

## 📚 支持算法 (部分)

| 类别           | 算法                                                |
| -------------- | --------------------------------------------------- |
| **基础算法**   | FedAvg, FedProx, Scaffold, FedDyn, MOON             |
| **个性化算法** | FedRep, FedProto, FedALA, FedPer, LG-FedAvg, FedDPC |
| **知识蒸馏**   | FedKD, FML, ProxyFL                                 |
| **语义锚点**   | FedSA, FedLSA                                       |
| **去中心化**   | L2C, PearFL, DispFL, DFedAvgM, DFedPGP              |

---

_更多详细开发规范与底层原理，请参阅 `GEMINI.md`。_
