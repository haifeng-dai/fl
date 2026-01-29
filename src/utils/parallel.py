def run_parallel_clients(
    client_worker,
    parameters,
    gpu_pools,
    mp=False,
):
    """
    运行并行或顺序客户端训练。
    :param client_worker: 客户端训练函数
    :param parameters: 每个客户端的参数列表
    :param gpu_pools: 设备对应的进程池字典
    :param mp: 是否启用多进程
    :return: 按 client_id 顺序排列的结果列表
    """
    if not mp:
        # 顺序执行
        res = {p[0]: client_worker(p) for p in parameters}
    else:
        async_results = {
            p[0]: gpu_pools[p[1]].apply_async(client_worker, (p,)) for p in parameters
        }
        res = {i: r.get() for i, r in async_results.items()}

    return res
