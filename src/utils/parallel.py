def client_worker(client, parameters):
    client.set_client(parameters)
    result = client.train()
    return client.client_id, result


def run_parallel_clients(
    clients,
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
        for client_id, client in clients.items():
            if isinstance(parameters, dict) and client_id in parameters:
                client_params = parameters[client_id]
            else:
                client_params = parameters
            results.append(client_worker(client, client_params))
        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]

    async_results = []
    for client_id, client in clients.items():
        pool = gpu_pools[client.device]

        # 如果 parameters 是字典且包含当前 client_id，则取其对应参数，否则取全局参数
        if isinstance(parameters, dict) and client_id in parameters:
            client_params = parameters[client_id]
        else:
            client_params = parameters

        async_results.append(
            pool.apply_async(
                client_worker,
                (
                    client,
                    client_params,
                )
            )
        )

    all_results = [r.get() for r in async_results]
    all_results.sort(key=lambda x: x[0])
    return [r[1] for r in all_results]