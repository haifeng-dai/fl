def run_parallel_clients(
    client_worker,
    num_clients,
    parameters,
    gpu_pools,
    no_mp=False,
):
    """
    运行并行或顺序客户端训练。
    :param clients: 客户端字典 {id: client_obj}
    :param parameters: 可以是所有客户端共用的参数，也可以是 {id: params} 的字典
    :param gpu_pools: 设备对应的进程池字典
    :param no_mp: 是否禁用多进程
    """
    if no_mp:
        results = []
        for client_id in range(num_clients):
            results.append(client_worker(client_id, parameters[client_id]))
        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]

    async_results = []
    for client_id in range(num_clients):
        pool = gpu_pools[parameters[client_id][0]]

        async_results.append(
            pool.apply_async(client_worker, (client_id, parameters[client_id]))
        )

    all_results = [r.get() for r in async_results]
    all_results.sort(key=lambda x: x[0])
    return [r[1] for r in all_results]
