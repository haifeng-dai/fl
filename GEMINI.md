# Gemini 联邦学习开发规约 (Ray Exclusive Branch)

> **Antigravity 助手提示**: 本文档是该仓库 `ray` 分支的最高开发准则。在进行任何代码修改前，必须确保符合以下底层逻辑。

## 1. 核心架构逻辑 (System Architecture)

### 1.1 Ray 强制调度
*   **入口点**: `main.py` 不再负责具体的并行分配，它仅初始化全局 Ray 集群。
*   **调度器**: `BaseServer` 是唯一的任务调度中心。`run_clients` 方法已被重写，强制将任务包装为 Ray 远程对象。
*   **设备映射**: Worker 内部通过 Ray 资源管理器自动分配可见 GPU，逻辑上始终访问 `cuda:0`。

### 1.2 零拷贝数据流
*   **通信瓶颈**: 严禁在主循环中手动对 `state_dict` 进行大批量的 `.cpu()` 或 `deepcopy` 操作。
*   **Object Store**: 充分利用 Ray 的对象存储。大型对象（如 `global_model`）会被自动优化。

## 2. 开发者强制准则 (Mandatory Rules)

1.  **MP 绝对禁令**: 严禁引入 `torch.multiprocessing`。若发现相关代码，必须立即重构为 Ray Task。
2.  **结果隔离策略**: 必须确保 `BaseServer` 初始化时，`save_path` 和 `log_path` 分别指向 `results_ray/` 和 `logs_ray/`。
3.  **显存配额管理**: `--ray_gpu` 的值必须根据模型复杂度动态调整。ResNet18 级模型严禁设为 `>0.5`（防止并发冲突）。
4.  **模型状态导出**: 在 `client_worker` 返回前，必须执行 `{k: v.cpu().detach().clone() for k, v in state.items()}`，这是防止 Ray 对象存储出现悬空引用的关键。

## 3. 性能陷阱与优化 (Pitfalls & Optimization)

*   **ANCData 幽灵**: 在此分支下若出现 `ancdata` 错误，说明有地方误用了原生多进程通信。
*   **Serialization Error**: 传递给 `remote` 函数的参数必须是可序列化的。避免直接传递 `torch.device` 对象，应传递字符串。
*   **GPU 预热**: 第一轮训练由于 Ray 的启动开销会显著变慢，属于正常现象，性能评估应从第二轮开始。

## 4. 目录权限与隔离

*   **ReadOnly**: `results/` 和 `logs/` 对此分支应视为“只读”或“遗留数据”。
*   **WriteOnly**: 本分支产生的任何持久化数据必须进入 `results_ray/`。

---
**配置版本**: v2.0 (Ray-Integrated)
**状态**: 生产级
