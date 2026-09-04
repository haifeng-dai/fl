"""FedDPC 的专属绘图辅助：基线超参与扫参统计（自 notebook 上收，消除两份重复实现）。"""

import numpy as np

# FedDPC 扫参时的固定背景超参（与 configs/algorithms.yaml 对齐）
FEDDPC_BASE_PARAMS = {
    "lamda_": 100.0,
    "lambda_p": 1.0,
    "lambda_acl": 0.01,
    "head_epochs": 10,
    "body_epochs": 1,
    "lr_head": 0.01,
    "lr_body": 0.01,
    "margin_threshold": 100.0,
    "server_epochs": 100,
    "server_lr": 0.01,
}


def load_parameter_stats(selected_group, experiments, common_args, loader, x_lim=200, *, runs):
    """
    加载并计算实验数据的 Max Acc 均值和标准差。
    返回: param_values (x), means (y), stds, param_name
    runs 必填：直读第 0..runs-1 次结果，缺失即抛错。
    """
    param_values = []
    means = []
    stds = []

    # 尝试从 label 中提取参数名
    param_name = "Parameter"
    if selected_group and "=" in selected_group[0]:
        param_name = selected_group[0].split("=")[0]

    for label in selected_group:
        if label not in experiments:
            continue
        algo_name, kwargs = experiments[label]
        merged_args = {**common_args, **kwargs}

        try:
            val = float(label.split("=")[1]) if "=" in label else float(label)
        except (IndexError, ValueError):
            val = label

        run_results = loader.load_runs(algo_name, runs=runs, **merged_args)

        m_run_max = []
        for run in run_results:
            acc_data = run.get("acc", [])
            y_data = acc_data[:x_lim]
            if y_data:
                m_run_max.append(np.max(y_data))

        if m_run_max:
            param_values.append(val)
            means.append(np.mean(m_run_max))
            stds.append(np.std(m_run_max))

    # 排序
    if all(isinstance(v, (int, float)) for v in param_values):
        data_points = sorted(zip(param_values, means, stds), key=lambda x: x[0])
        param_values = [p[0] for p in data_points]
        means = [p[1] for p in data_points]
        stds = [p[2] for p in data_points]

    return param_values, means, stds, param_name
