def run_parallel_clients(
    client_worker,
    num_clients,
    parameters,
    gpu_pools,
    mp=False,
):
    """
    运行并行或顺序客户端训练。
    :param client_worker: 客户端训练函数
    :param num_clients: 客户端数量
    :param parameters: 每个客户端的参数列表
    :param gpu_pools: 设备对应的进程池字典
    :param mp: 是否启用多进程
    :return: 按 client_id 顺序排列的结果列表
    """
    if not mp:
        # 顺序执行：结果天然有序，无需排序
        return [
            client_worker(client_id, parameters[client_id])[1]
            for client_id in range(num_clients)
        ]

    # 并行执行：按顺序提交任务
    async_results = [
        gpu_pools[parameters[client_id][0]].apply_async(
            client_worker, (client_id, parameters[client_id])
        )
        for client_id in range(num_clients)
    ]

    # 按提交顺序获取结果，天然有序，无需排序
    return [r.get()[1] for r in async_results]
