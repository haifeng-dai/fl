# 项目知识库 (Project Knowledge Base)

**生成日期:** 2026-01-27
**类型:** 联邦学习研究框架 (Unified Pipeline)
**技术栈:** Python 3.14, PyTorch 2.6+, CUDA 13.0, uv

## 概览 (Overview)

这是一个支持多 GPU 并行训练的高效联邦学习框架。实现了 **17 种 FL 算法** (包括 FedAvg, MOON, FedProto, FedALA, FedTGP 等)，支持 IID、Dirichlet 和 Pathological 数据划分。

## 目录结构 (Structure)

```
./
├── main.py              # 入口：两阶段参数解析 → 动态算法加载
├── run.sh               # 批量实验编排脚本 → 调度 scripts/*.sh
├── pyproject.toml       # uv + BasedPyright 配置
├── src/                 # 核心包
│   ├── <algo>.py        # fedavg, moon, fedproto, fedala, fedtgp 等 17 个算法
│   ├── models/          # CNN, ResNet18/50, HARCNN (返回 tuple: logits, features)
│   ├── utils/           # BaseServer, 并行化 (parallel), 聚合 (aggregate), 评估
│   └── data_gen/        # 数据集准备 (MNIST, CIFAR-10, HAR)
├── scripts/             # 网格搜索 Shell 脚本 (嵌套循环支持批量实验)
└── datasets/            # 生成的持久化分区数据
```

## 关键代码位置 (Where to Look)

| 任务 | 位置 | 备注 |
|------|----------|-------|
| 添加算法 | `src/<algo>.py` | 继承 `BaseServer`, 导出 `add_args`, `client_worker`, `Server` |
| 添加模型 | `src/models/` | 必须返回 `(logits, features)` 元组 |
| 多 GPU 训练 | `src/utils/fed_utils.py` | `_BaseServer__start_pools()` 处理 GPU 分配 |
| 数据划分 | `src/data_gen/__init__.py` | 统一的数据准备入口 |
| 并行执行 | `src/utils/parallel.py` | `run_parallel_clients()` 保证结果顺序与输入一致 |

## 开发规范 (Conventions)

- **导入**: 推荐使用绝对导入 `from .utils.fed_utils import ...`
- **类型提示**: 强烈建议使用
- **命名**: PascalCase (类), snake_case (函数/变量), UPPER_SNAKE_CASE (常量)
- **多 GPU**: 主进程必须调用 `set_start_method("spawn", force=True)`
- **资源清理**: `Server.close()` 会自动清理进程池

## 常见陷阱 (Anti-Patterns / Pitfalls)

- **DataLoader 切片**: **严禁**直接切片 Dataset (如 `dataset[0:10]`)，这通常会返回 Tensor 元组导致 DataLoader 崩溃。**必须**使用 `torch.utils.data.Subset`。
- **Python 循环 GPU 操作**: 在训练循环中避免使用 `for i in batch: tensor.to(device)`。这会严重阻塞 GPU 流水线。应使用向量化操作 (如 `global_protos[y]`)。
- **模型复制**: 在多进程 worker 中，优先使用 `get_model()` + `load_state_dict()` 而非 `copy.deepcopy()`，前者更快且不仅限于 pickle。
- **路径硬编码**: 尽量使用 `os.path.join`，避免硬编码路径分隔符。

## 常用命令 (Commands)

```bash
# 安装依赖
uv sync

# 运行单个实验
uv run main.py --algo fedavg --dataset mnist --gpus 0,1

# 运行批量实验 (推荐)
./run.sh

# 格式化代码
uv run black .
```

## 注意事项 (Notes)

- **算法支持**: 目前支持 17 种算法，包括最新的 FedALA 和 FedTGP。
- **结果保存**: 结果保存在 `results/<dataset>_<partition>_<num_clients>/` 目录下。
- **数据准备**: 首次运行会自动下载和处理数据。
