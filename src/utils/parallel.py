import torch.multiprocessing as mp


def client_worker(client_cls, client_id, device, global_params, args, return_dict):
    """
    Worker function to instantiate client and run training in a sub-process.
    """
    client = client_cls(client_id, device, args)
    updated_params = client.train(global_params)
    return_dict[client_id] = updated_params


def run_parallel_clients(client_cls, clients_info, global_params, common_args):
    manager = mp.Manager()
    return_dict = manager.dict()
    processes = []

    for client_id, device in clients_info:
        p = mp.Process(
            target=client_worker,
            args=(
                client_cls,
                client_id,
                device,
                global_params,
                common_args,
                return_dict,
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    return [return_dict[i] for i in sorted(return_dict.keys())]
