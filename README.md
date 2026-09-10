# Unified Federated Learning Framework (Ray Powered)

本项目是一个基于 **Ray** 的多 GPU 联邦学习研究框架，支持多种联邦学习算法、大规模客户端模拟和半监督/域泛化实验。客户端训练任务由 Ray 统一调度，项目仅支持 NVIDIA CUDA GPU。

## 🚀 快速开始

### 1. 环境准备

推荐使用 `uv` 进行依赖管理：

```bash
uv sync
```

### 2. 运行实验

现在系统采用 **Config-First** 架构，所有参数优先从 YAML 加载，并支持命令行动态覆盖。

```bash
# 基本运行 (使用 configs/default.yaml 中的默认配置)
uv run main.py --algo fedavg

# 大规模并行运行：先在 configs/default.yaml 中设置模型、数据集、GPU 和并发数
uv run main.py --algo fedprox

# 快速测试模式 (不记录文件，直接输出到终端，默认只测一次)
uv run main.py -a fedavg -t

# 定制通信轮数 (测试或正式训练均生效，覆盖 YAML 中的 rounds)
uv run main.py -a fedavg -t -r 10

# 多次实验 (配合 YAML times 指定运行次数，--run_time 选择试验索引)
uv run main.py -a fedavg -t --run_time "0,1,2"

# 从检查点恢复训练
uv run main.py -a fedavg --resume_from results/<实验目录>/checkpoints/checkpoint_50.pt
```

## 🛠️ 核心参数说明

- `-a, --algo`: 算法名称；具体可用算法以 `configs/algorithms.yaml` 和 `src/algorithms/` 中的实现为准。
- `model`（YAML）: 模型架构（支持 cnn, resnet18, resnet50, harcnn 等）。
- `dataset`（YAML）: 数据集（支持 cifar10/100, mnist, tiny_imagenet 等）。
- `max_workers_per_gpu`（YAML）: **关键资源参数**。每块 GPU 上同时运行的 Worker 数量，当前默认值为 `5`；较大的模型通常需要降低该值。
- `gpus`（YAML）: 指定至少一张 NVIDIA GPU（如 `0,1,2,3`）。本项目不提供 CPU 训练模式，CUDA 不可用时会直接报错。
- `-t, --test`: 开启测试模式（无需参数值），日志直接输出到终端而非文件；默认只测一次（`times=1`），并使用较小的轮数和 Epoch 进行快速验证。
- `--run_time`: 指定要运行的试验索引子集（0 起始，逗号分隔，如 `"0,1,2"`），与 YAML `times` 配合执行多次实验。
- `-r, --rounds`: 定制通信轮次，覆盖 YAML 中的 `rounds`；默认正式训练为 1000 轮，测试模式与正式训练均生效。
- `--resume_from`: 从指定 checkpoint 文件恢复训练；checkpoint 是否保存以及保存间隔由 `checkpoint_enabled` 和 `checkpoint_interval` 控制。

## 📂 配置与结果管理

- **全局默认配置**: `configs/default.yaml`
- **算法专属配置**: `configs/algorithms.yaml`（支持超参数搜索，只需将参数设为列表即可自动展开）。
- **实验结果存储**: `results/`
- **运行日志存储**: `logs/`
- **检查点存储**: 默认位于对应实验结果目录下的 `checkpoints/`

## 🧪 测试与开发

使用项目环境运行测试：

```bash
uv run pytest
```

快速验证 Ray、GPU 和基本训练流程：

```bash
uv run main.py -a fedavg -t
```

运行前请在 `configs/default.yaml` 中确认 `gpus`、`model`、`dataset` 和
`max_workers_per_gpu` 配置。项目不提供 CPU fallback；没有可用 CUDA GPU 时会直接报错。

## 🧩 域划分与半监督（测试集生成规则）

框架为 Config-First，域相关行为由 `configs/default.yaml` 的 `domain_partition` / `selected_domains` / `unlabeled_domain` / `target_domain` 控制。测试集文件按以下规则生成（与 `src/algorithms/utils/load_data.py` 行为一致）：

- **`source_test.pt`**：只要设置了 `domain_partition`（非空），划分生成时**恒会**产出，包含全部域测试样本的拼接。
- **`target_test.pt`**：仅当设置 `target_domain`（留一域评估）时生成——该域被移出训练集、单独留作测试；未设置时文件不存在，对应 `self.target_test` 为 `None`，切勿直接遍历。
- **`selected_domains`**：逗号分隔字符串，仅保留指定域参与训练/测试；与 `target_domain` 互斥（勿用列表，会被框架误判为参数扫描）。
- **`unlabeled_domain`**：在 `domain_partition` 非空时生效，将该域样本标签掩掉以构造半监督场景。

常见参数组合（传统 FL / 通用半监督 / DG / DG+域掩半监督 / 留一域评估等）见 `configs/default.yaml` 的"组合模式"注释块。

## 📦 数据集支持

### 数据集一览

| 类型         | 数据集                                                        | 分辨率          | 类别     | 推荐模型        |
| ------------ | ------------------------------------------------------------- | --------------- | -------- | --------------- |
| **通用**     | cifar10, cifar100, mnist, fashionmnist, svhn, emnist, femnist | 28×28 ~ 32×32   | 10 ~ 100 | cnn             |
| **高分辨率** | tiny_imagenet, cars, flowers102, gtsrb, cinic10               | 32×32 ~ 224×224 | 10 ~ 200 | resnet18        |
| **传感器**   | har (UCI-HAR)                                                 | 9ch 时序        | 6        | harcnn / harmlp |
| **域泛化**   | cifar10_dg, pacs, officehome, vlcs, domainnet                 | 32×32 ~ 224×224 | 5 ~ 345  | resnet18        |

### 域泛化数据集详情

#### CIFAR-10 DG（合成域，自动下载）

由 CIFAR-10 通过 4 种增广策略生成，无需手动下载。

| 领域            | 增广策略                                    |
| --------------- | ------------------------------------------- |
| `clean`         | 仅 ToTensor + Normalize                     |
| `color_jitter`  | ColorJitter(亮度/对比度/饱和度/色相)        |
| `blur_noise`    | GaussianBlur(3×3)                           |
| `rotate_cutout` | RandomRotation(30°) + RandomResizedCrop(32) |

```yaml
# 配置示例：以 rotate_cutout 为无标签域做半监督
dataset: cifar10_dg
model: cnn
domain_partition: domain_mixed_aware
unlabeled_domain: rotate_cutout
```

#### PACS

下载 `pacs.zip` 放到 `./datasets/raw/pacs.zip`：

- 官网：https://sketchx.eecs.qmul.ac.uk/downloads/
- 4 个域，7 类（dog / elephant / giraffe / guitar / horse / house / person），分辨率 224×224

| 领域           | 说明     |
| -------------- | -------- |
| `photo`        | 照片     |
| `art_painting` | 艺术绘画 |
| `cartoon`      | 卡通     |
| `sketch`       | 素描     |

```yaml
# 配置示例：mask sketch 域做半监督自训练
dataset: pacs
model: resnet18
domain_partition: domain_as_client
unlabeled_domain: sketch
```

#### OfficeHome

下载 `officehome.zip` 放到 `./datasets/raw/officehome.zip`：

- 官网：https://www.hemanthdv.org/officeHomeDataset.html
- 4 个域，65 类，分辨率 224×224

| 领域        | 说明     |
| ----------- | -------- |
| `Art`       | 艺术品   |
| `Clipart`   | 剪贴画   |
| `Product`   | 商品照片 |
| `RealWorld` | 真实世界 |

#### VLCS

4 个域，5 个共享类（bird / car / chair / dog / person），分辨率 224×224

| 领域         | 来源            |
| ------------ | --------------- |
| `VOC2007`    | PASCAL VOC 2007 |
| `LabelMe`    | LabelMe         |
| `Caltech101` | Caltech-101     |
| `SUN09`      | SUN09           |

#### DomainNet

自动下载，6 个域，345 类

| 领域        | 说明                       |
| ----------- | -------------------------- |
| `clipart`   | 剪贴画                     |
| `infograph` | 信息图                     |
| `painting`  | 绘画                       |
| `quickdraw` | 涂鸦（灰度，自动转伪 RGB） |
| `real`      | 真实照片                   |
| `sketch`    | 素描                       |

### 配置切换要点

切换数据集时需调整以下参数：

| 参数               | 通用 → DG                              | DG → 通用                        |
| ------------------ | -------------------------------------- | -------------------------------- |
| `dataset`          | `cifar10` → `pacs`                     | `domainnet` → `mnist`            |
| `model`            | `cnn` → `resnet18`（224px 数据集）     | `resnet18` → `cnn`               |
| `domain_partition` | `~` → `domain_as_client`（启用域划分） | `domain_as_client` → `~`（关闭） |
| `partition`        | 域划分下无效，可保留                   | 生效（`iid` / `dirichlet`）      |
| `unlabeled_domain` | 指定要掩标签的域名                     | 不适用，置 `~`                   |
| `selected_domains` | 过滤仅保留部分域                       | 不适用，置 `~`                   |

## 📚 支持算法

| 类别           | 算法                                                                                                     |
| -------------- | -------------------------------------------------------------------------------------------------------- |
| **个性化算法** | FedRep, FedProto, FedALA, FedPer, LG-FedAvg, FedDPC, FedPLN, FedProc, FedTGP, FedKD, FML, ProxyFL, FedFM |
| **全局基础**   | FedAvg, FedProx, Scaffold, FedDyn, MOON, Local, EFHC, FedSA, FedLSA                                      |
| **去中心化**   | L2C, PearFL, DispFL, DFedAvgM, DFedPGP, DFedSet                                                          |
| **双域半监督** | FedTest                                                                                                  |

---
