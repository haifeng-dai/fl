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
3.  **Ray 显存管理**: 必须通过 `--max_workers_per_gpu` 控制并行度（框架会自动计算 1/n 的显存比例）。建议：ResNet18 设为 2，CNN 设为 3 或 4。
4.  **模型状态导出**: 在 `client_worker` 返回前，必须执行 `{k: v.cpu().detach().clone() for k, v in state.items()}`，这是防止 Ray 对象存储出现悬空引用的关键。
5.  **代码风格规范**: 必须严格遵守 **PEP 8** 标准。所有 `import` 语句必须置于文件顶部，严禁在函数或异常处理块内进行非必要的局部导入。
6.  **沟通与文档语言**: 所有输出、计划、说明以及与用户的沟通必须统一使用 **中文**。

## 3. 性能陷阱与优化 (Pitfalls & Optimization)

*   **ANCData 幽灵**: 在此分支下若出现 `ancdata` 错误，说明有地方误用了原生多进程通信。
*   **Serialization Error**: 传递给 `remote` 函数的参数必须是可序列化的。避免直接传递 `torch.device` 对象，应传递字符串。
*   **GPU 预热**: 第一轮训练由于 Ray 的启动开销会显著变慢，属于正常现象，性能评估应从第二轮开始。
