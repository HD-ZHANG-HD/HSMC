import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterMathtext

def plot_latency_bar():
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

    BLB = [413.1, 577.3, 242.6, 406.7, 233.1, 397.2, 65.4, 102.4, 64.2, 101.1]
    BumbleBee = [63111.5, 110055.1, 48573.9, 95517.5, 47766.2, 94709.9, 1350.1, 11912.4, 1242.4, 1180.5]
    NEXUS = [238.6] * 10
    Hybrid = [237, 237, 201.2, 237, 180, 194, 59.1, 91.8, 56.4, 90]

    data = [BLB, BumbleBee, NEXUS, Hybrid]
    labels = ["BLB", "BumbleBee", "NEXUS(Pure HE)", "Hybrid Compiler"]

    colors = [
        "#1f77b4",  # BLB - 蓝
        "#ff7f0e",  # BumbleBee - 橙
        "#2ca02c",  # NEXUS - 绿
        "#d62728",  # Hybrid - 红
    ]

    # ======================
    # 画图参数
    # ======================
    y = np.arange(len(groups))
    bar_height = 0.2

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
    # 关键：log scale + 10^x 显示
    # ======================
    plt.xscale("log")
    plt.gca().xaxis.set_major_formatter(LogFormatterMathtext())

    # ======================
    # 坐标轴设置
    # ======================
    plt.yticks(y + 1.5 * bar_height, groups)
    plt.xlabel("Latency (ms)", fontsize=12)
    plt.ylabel("Bandwidth - RTT", fontsize=12)
    plt.title("Latency Comparison Across Bandwidth and RTT Settings", fontsize=14)

    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.legend(fontsize=11)

    plt.tight_layout()

    # 保存（可选）
    plt.savefig("latency_bar_log.png", dpi=300)

    plt.show()


if __name__ == "__main__":
    plot_latency_bar()