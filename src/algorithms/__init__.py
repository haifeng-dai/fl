import importlib


def load_algorithm(algo_name):
    """
    加载指定算法的核心组件。
    返回: (ServerClass, get_path_func)
    """
    module_name = algo_name.lower()
    try:
        module = importlib.import_module(f".{module_name}", __name__)
        server_cls = getattr(module, "Server")
        get_path_func = getattr(module, "get_path")
        return server_cls, get_path_func
    except ModuleNotFoundError:
        raise ValueError(f"Algorithm '{algo_name}' not found in src.algorithms.")
    except AttributeError as e:
        raise AttributeError(
            f"Algorithm '{algo_name}' is missing a required component: {e}"
        )


__all__ = ["load_algorithm"]
