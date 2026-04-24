"""End-to-end compiler sweep across (bandwidth, RTT) settings.

For each network setting we:
1. Compile the BERT encoder block under that setting.
2. Print compiler plan + cost breakdown.
3. Compare against all-HE, all-MPC, and static-hybrid baselines.

The profile is assumed to be built ahead of time by
``python -m compiler.state_expanded.build_profile``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List, Tuple

from .bert_graph import bert_block_graph, default_manifest
from .cost_model import NetworkSetting
from .planner import (
    compile_plan,
    evaluate_static_hybrid,
    evaluate_uniform_domain,
)
from .profile_schema import LatencyProfile


DEFAULT_BANDWIDTHS = [
    ("10Mbps", 10e6),
    ("100Mbps", 100e6),
    ("1Gbps", 1e9),
    ("3Gbps", 3e9),
]

DEFAULT_RTTS = [1, 20, 40, 80]


def run_sweep(
    profile_path: Path,
    bandwidths: List[Tuple[str, float]] | None = None,
    rtts: List[float] | None = None,
) -> None:
    bandwidths = bandwidths or DEFAULT_BANDWIDTHS
    rtts = rtts or DEFAULT_RTTS

    profile = LatencyProfile.load(profile_path)
    graph = bert_block_graph(default_manifest())

    print(f"[demo] profile={profile_path} device={profile.platform}")
    print(f"[demo] graph nodes={len(graph.nodes)} edges={len(graph.edges)}")
    print("")

    header = f"{'BW':>10s} {'RTT':>5s} | {'all-HE':>12s} {'all-MPC':>12s} " \
             f"{'static':>12s} {'compiler':>12s} {'speedup':>8s}"
    print(header)
    print("-" * len(header))

    fails = 0
    for bw_label, bw in bandwidths:
        for rtt in rtts:
            net = NetworkSetting(bandwidth_bps=bw, rtt_ms=rtt)
            all_he = evaluate_uniform_domain(graph, profile, net, "HE")
            all_mpc = evaluate_uniform_domain(graph, profile, net, "MPC")
            static_hybrid = evaluate_static_hybrid(graph, profile, net)
            try:
                plan = compile_plan(graph, profile, net)
                comp = plan.total_cost_ms
            except RuntimeError as exc:
                comp = math.inf
                fails += 1
                print(f"{bw_label:>10s} {rtt:>4g}ms | compile failed: {exc}")
                continue
            best_base = min(all_he, all_mpc, static_hybrid)
            speedup = best_base / comp if comp > 0 else float('inf')
            print(
                f"{bw_label:>10s} {rtt:>4g}ms | "
                f"{_fmt_ms(all_he):>12s} {_fmt_ms(all_mpc):>12s} "
                f"{_fmt_ms(static_hybrid):>12s} {_fmt_ms(comp):>12s} "
                f"{speedup:>7.2f}x"
            )

    if fails == 0:
        print("\n[demo] all settings compiled successfully.")
    else:
        print(f"\n[demo] {fails} settings failed.")


def _fmt_ms(x: float) -> str:
    if math.isinf(x):
        return "inf"
    if x >= 1000:
        return f"{x/1000:.2f}s"
    return f"{x:.1f}ms"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", type=str, required=True)
    ap.add_argument("--verbose-plan", action="store_true")
    args = ap.parse_args()

    run_sweep(Path(args.profile))

    if args.verbose_plan:
        profile = LatencyProfile.load(args.profile)
        graph = bert_block_graph(default_manifest())
        net = NetworkSetting(bandwidth_bps=100e6, rtt_ms=20)
        plan = compile_plan(graph, profile, net)
        print("\n" + plan.pretty())


if __name__ == "__main__":
    main()
