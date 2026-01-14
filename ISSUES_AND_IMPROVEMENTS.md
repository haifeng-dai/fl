# 项目问题与改进（自动生成）

此文档汇总在代码审查过程中发现的致命错误、不足之处和改进建议，供后续按任务逐项修复。

**致命错误（Blocking）**

- **参数聚合未按样本数加权**：在 `src/utils/aggregate.py` 中，聚合时未使用各客户端样本数进行加权，FedAvg 在数据量不均时需要按 `n_i / n_total` 加权。相关文件：[src/utils/aggregate.py](src/utils/aggregate.py#L1)。

- **并行执行时传递不可序列化对象**：在 [src/utils/parallel.py](src/utils/parallel.py#L1) 中，将包含 `DataLoader` 或整个 `client` 对象传入进程池，会导致 pickle 开销极大或失败。应只传递必要的配置信息或在子进程内重建 DataLoader。

- **模型输入维度硬编码**：`src/models/simple_cnn.py` 中全连接层输入维度使用 `64 * 7 * 7`，仅适用于 28x28 输入（如 MNIST）。更换数据集（如 CIFAR）会出现维度不匹配错误。

- **MOON 前向逻辑可能不符合论文设计**：`SimpleCNN.forward()` 同时返回分类 `y` 与投影 `z`，但分类器忽略投影头的变换；需明确对比学习阶段与分类阶段的连接关系并修正。

**不足之处（Deficiencies）**

- 路径和数据目录多处硬编码（例如 `./datasets`），缺少统一配置和 CLI 参数。
- 聚合实现对设备（CPU/GPU）处理不够友好：频繁 `.cpu()`/`.to()` 导致不必要的数据拷贝，影响效率。

**改进建议（Actionable improvements）**

1. **按样本数加权的 FedAvg 实现**
   - 在 `src/utils/fed_utils.py`（或客户端训练函数）中返回 `num_samples`。
   - 在 `src/utils/aggregate.py` 中修改 `param_aggregate`：接收 `weights` 或 `num_samples` 列表，按比例加权。
   - 验收标准：在非均衡合成数据上，按样本数加权的聚合结果与手算加权一致。

2. **重构并行逻辑，避免传递 DataLoader**
   - 在 `src/utils/parallel.py` 中改为传递最小化的参数（如 `client_id`, `client_config`），并在子进程内部重建 `DataLoader`。
   - 或者改用多线程（若 I/O 密集）或使用 `torch.multiprocessing` 的 `spawn` 启动方法避免 pickle 问题。
   - 验收标准：并行训练不再因序列化错误崩溃，启动时间显著降低，内存占用合理。

5. **改进聚合效率**
   - 在可能时将 `state_dicts` 先移动到目标设备（例如 GPU），批量合并再同步到主设备；减少循环内频繁拷贝。

**优先级与建议的修复顺序（高到低）**

- P1（高优先）：按样本数加权聚合（任务 1）、并行序列化修复（任务 2）。
- P2（中优先）：模型输入维度参数化与 MOON 前向修正（任务 3）。
- P3（低优先）：配置化与性能优化（任务 4、5），测试与文档（任务 6）。
