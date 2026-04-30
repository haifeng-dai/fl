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
```bash
# 基本运行 (FedAvg, CIFAR-10)
uv run main.py --algo fedavg --dataset cifar10 --gpus 0

# 大规模并行运行 (4 GPU, 每卡 2 并行, ResNet18)
uv run python main.py --algo fedtest --model resnet18 --dataset cifar100 --gpus 0,1,2,3 --ray_gpu 0.5
```

## 🛠️ 核心参数说明

- `--algo`: 算法名称（支持 FedAvg, FedProx, Scaffold, FedRep, FedALA 等 23 种）。
- `--model`: 模型架构（支持 cnn, resnet18, resnet50, harcnn 等）。
- `--dataset`: 数据集（支持 cifar10/100, mnist, tiny_imagenet 等）。
- `--ray_gpu`: **关键资源参数**。每个 Worker 占用的 GPU 比例（如 0.5 表示单卡 2 并行）。
- `--gpus`: 指定物理 GPU 编号（如 `0,1,2,3`）。
- `--test 1`: 开启测试模式，日志将直接输出到终端而非文件。

## 📂 结果展示与隔离

为了保护历史实验数据，该分支的输出已重定向：
- **实验结果**: 存储在 `results_ray/`。
- **运行日志**: 存储在 `logs_ray/`。

## 📚 支持算法 (部分)

| 类别 | 算法 |
|------|------|
| **基础算法** | FedAvg, FedProx, Scaffold, FedDyn, MOON |
| **个性化算法** | FedRep, FedProto, FedALA, FedPer, LG-FedAvg |
| **知识蒸馏** | FedKD, FML, ProxyFL |
| **语义锚点** | FedSA, FedLSA |

---
*更多详细开发规范与底层原理，请参阅 `GEMINI.md`。*
