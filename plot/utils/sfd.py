"""SFD 跨域半监督场景的专用绘图。"""

import os

import matplotlib.pyplot as plt
import numpy as np

from .plotting import beautify_label


def plot_sfd_results(results_dict, x_lim, title=None, show_proto=True):
    """
    SFD 场景三面板对比图：Overall / Source (Labeled) / Target (Unlabeled)。

    show_proto=True 时：Overall 面板叠加 acc_proto 虚线，Target 面板叠加
    source_proto_on_target 虚线。某算法缺少对应 key 时该线不画。
    三子图 xlim 在 savefig 之前设置，保证落盘图片与屏幕显示一致。
    """
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    ax_overall, ax_source, ax_target = axes
    panels = [
        (ax_overall, "Overall", "acc"),
        (ax_source, "Source (Labeled)", "acc_source"),
        (ax_target, "Target (Unlabeled)", "acc_target"),
    ]
    summary = []

    for label, data in results_dict.items():
        if data is None:
            continue
        display_label = beautify_label(label)
        row = {"Algorithm": display_label}
        for ax, panel_name, key in panels:
            series = data.get(key)
            if series is None:
                continue
            y = series[0:x_lim]
            if len(y) == 0:
                continue
            (line,) = ax.plot(y, label=display_label)
            row[panel_name] = (
                max(y),
                float(np.mean(y[-10:])) if len(y) >= 10 else float(np.mean(y)),
            )
            # Target 面板叠加源原型在目标域上的表现
            if ax is ax_target and show_proto and data.get("source_proto_on_target"):
                py = data["source_proto_on_target"][0:x_lim]
                ax.plot(
                    py,
                    linestyle="--",
                    alpha=0.8,
                    color=line.get_color(),
                    label=f"{display_label} SrcProto",
                )
        # Overall 面板叠加全局原型准确率
        if show_proto and data.get("acc_proto"):
            py = data["acc_proto"][0:x_lim]
            ax_overall.plot(
                py,
                linestyle="--",
                alpha=0.8,
                label=f"{display_label} Proto",
            )
        summary.append(row)

    for ax, panel_name, _ in panels:
        ax.set_title(panel_name, fontsize=13, fontweight="bold")
        ax.set_xlabel("Rounds")
        ax.set_ylabel("Accuracy (%)")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, x_lim)
        ax.legend(loc="lower right", fontsize=11)

    fig.suptitle(title or "SFD Comparison", fontsize=15)
    plt.tight_layout()

    save_base = (
        (title or "sfd_comparison")
        .lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("->", "to")
        .replace("__", "_")
    )
    save_name = f"sfd_{save_base}.png"
    os.makedirs("figures", exist_ok=True)
    plt.savefig(os.path.join("figures", save_name), dpi=300)
    print(f"Figure saved to figures/{save_name}")

    print("Summary of SFD Accuracy:")
    header = f"{'Algorithm':<22} | {'O-Max':>8} | {'O-Last10':>8} | {'S-Max':>8} | {'S-Last10':>8} | {'T-Max':>8} | {'T-Last10':>8}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for row in summary:
        cells = [f"{row['Algorithm']:<22}"]
        for panel in ("Overall", "Source (Labeled)", "Target (Unlabeled)"):
            if panel in row:
                cells.append(f"{row[panel][0]:>8.4f} | {row[panel][1]:>8.4f}")
            else:
                cells.append(f"{'-':>8} | {'-':>8}")
        print(" | ".join(cells))
    print("-" * len(header))

    plt.show()
