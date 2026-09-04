"""联邦学习绘图包:读取(results/*.pt)与绘图(figures/)的统一入口。"""

from .core import ResultLoader, batch_cached_load, freeze_cache_value
from .plotting import (
    PARAM_MAP,
    beautify_label,
    load_plot,
    load_plot_all_runs,
    plot_loss,
    plot_results,
    plot_results_split,
    print_stats,
    print_summary_table,
)
from .sfd import plot_sfd_results

__all__ = [
    "PARAM_MAP",
    "ResultLoader",
    "batch_cached_load",
    "beautify_label",
    "freeze_cache_value",
    "load_plot",
    "load_plot_all_runs",
    "plot_loss",
    "plot_results",
    "plot_results_split",
    "plot_sfd_results",
    "print_stats",
    "print_summary_table",
]
