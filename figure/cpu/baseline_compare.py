import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterMathtext

def plot_latency_bar_v2_with_colors():
    # ======================
    # 数据
    # ======================
    groups = [
        "10M-40", "10M-80",
        "100M-40", "100M-80",
        "200M-40", "200M-80",
        "1G-1", "1G-10",
        "3G-1", "3G-10"
    ]

    BLB = [2196.6, 2360.8, 2026.1, 2190.2, 2016.6, 2180.8, 1849.0, 1885.9, 1847.7, 1847.7]
    BumbleBee = [63560.1, 11050.4, 49022.5, 95966.1, 48214.9, 95158.5, 1798.7, 12361.0, 1691.0, 12253.3]
    NEXUS = [17941.9] * 10
    SHAFT = [9060.8, 9120.6, 960.0, 1019.8, 509.9, 569.8, 91.5, 105.0, 31.5, 45.0]
    Hybrid = [1537, 1918, 960, 1019, 510, 572, 92, 105, 31, 45.0]

    data = [BLB, BumbleBee, NEXUS, SHAFT, Hybrid]
    labels = ["BLB", "BumbleBee", "NEXUS(Pure HE)", "SHAFT(Pure MPC)", "Hybrid Compiler"]

    # ======================
    # ⭐ 颜色控制（重点）
    # ======================
    colors = [
        "#1f77b4",  # BLB - 蓝
        "#ff7f0e",  # BumbleBee - 橙
        "#2ca02c",  # NEXUS - 绿
        "#9467bd",  # SHAFT - 紫
        "#d62728",  # Hybrid - 红
    ]

    # ======================
    # 画图参数
    # ======================
    y = np.arange(len(groups))
    bar_height = 0.15

    plt.figure(figsize=(14, 8))

    # ======================
    # 画柱状图
    # ======================
    for i in range(len(data)):
        plt.barh(
            y + i * bar_height,
            data[i],
            height=bar_height,
            label=labels[i],
            color=colors[i]   # ⭐ 应用颜色
        )

    # ======================
    # log + 10^x
    # ======================
    plt.xscale("log")
    plt.gca().xaxis.set_major_formatter(LogFormatterMathtext())

    # ======================
    # 坐标轴
    # ======================
    plt.yticks(y + 2 * bar_height, groups)
    plt.xlabel("Latency (ms)", fontsize=12)
    plt.ylabel("Bandwidth - RTT", fontsize=12)
    plt.title("Latency Comparison (Extended Methods)", fontsize=14)

    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig("latency_bar_log.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    plot_latency_bar_v2_with_colors()