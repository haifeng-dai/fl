import importlib


def load_algorithm(name: str):
    return importlib.import_module(f".{name.lower()}", __name__).Server
