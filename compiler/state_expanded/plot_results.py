"""Plot the sweep CSVs in the same form as paper Figure 4 (res_cpu/res_gpu).

One subplot per bandwidth; x-axis = RTT; y-axis = latency in seconds
(log scale, since all-MPC spans 4 orders of magnitude across settings).

Run::

    python -m compiler.state_expanded.plot_results \
        --cpu-csv compiler/state_expanded/results/cpu_sweep.csv \
        --gpu-csv compiler/state_expanded/results/gpu_sweep.csv \
        --out-dir compiler/state_expanded/results/
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def _load_csv(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open() as f:
        for r in csv.DictReader(f):
            rows.append({k: (float(v) if v not in {"inf", ""} else float("inf")) if k not in {"bandwidth_label"} else v for k, v in r.items()})
    return rows


def _plot(rows: List[dict], title: str, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Group by bandwidth.
    by_bw: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        by_bw[r["bandwidth_label"]].append(r)

    bw_labels = sorted(by_bw.keys(), key=lambda s: float(by_bw[s][0]["bandwidth_bps"]))

    fig, axes = plt.subplots(1, len(bw_labels), figsize=(4 * len(bw_labels), 3.5), sharey=True)
    if len(bw_labels) == 1:
        axes = [axes]

    series = [
        ("all_HE_ms", "FHE", "tab:blue", "o"),
        ("all_MPC_ms", "MPC (BOLT)", "tab:red", "s"),
        ("hybrid_linHE_nlMPC_ms", "hybrid: lin@HE + nl@MPC", "tab:orange", "^"),
        ("hybrid_linMPC_nlHE_ms", "hybrid: lin@MPC + nl@HE", "tab:purple", "v"),
        ("hybrid_attnMPC_ffnHE_ms", "hybrid: attn@MPC + FFN@HE", "tab:brown", "P"),
        ("compiler_ms", "ours (compiler)", "tab:green", "D"),
    ]

    for ax, bw in zip(axes, bw_labels):
        records = sorted(by_bw[bw], key=lambda r: r["rtt_ms"])
        xs = [r["rtt_ms"] for r in records]
        for key, label, color, marker in series:
            ys = [r[key] / 1000.0 for r in records]
            # filter inf for plotting but keep a tiny floor
            ys = [y if not math.isinf(y) else None for y in ys]
            xs_p = [x for x, y in zip(xs, ys) if y is not None]
            ys_p = [y for y in ys if y is not None]
            if ys_p:
                ax.plot(xs_p, ys_p, marker=marker, label=label, color=color,
                        linewidth=1.6, markersize=6)
        ax.set_title(f"{bw}")
        ax.set_xlabel("RTT (ms)")
        ax.set_yscale("log")
        ax.grid(True, which="both", axis="y", linestyle=":", alpha=0.5)

    axes[0].set_ylabel("End-to-end latency (s, log)")
    axes[0].legend(fontsize=8, loc="upper left")

    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, bbox_inches="tight", dpi=150)
    print(f"[plot] wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu-csv", type=str, required=True)
    ap.add_argument("--gpu-csv", type=str, required=True)
    ap.add_argument("--out-dir", type=str, required=True)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cpu = _load_csv(Path(args.cpu_csv))
    gpu = _load_csv(Path(args.gpu_csv))

    _plot(cpu, "BERT-base secure inference, CPU (Threadripper class)",
          out_dir / "res_cpu.png")
    _plot(gpu, "BERT-base secure inference, GPU (NEXUS-CUDA ref)",
          out_dir / "res_gpu.png")


if __name__ == "__main__":
    main()
