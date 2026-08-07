import importlib


def process_dataset(dataset_name, output_dir="./datasets/raw"):
    """
    调用对应 dataset 的处理函数下载/生成原始数据。
    """
    module = importlib.import_module(f"src.data_gen.process.{dataset_name}")
    return module.process(output_dir)
