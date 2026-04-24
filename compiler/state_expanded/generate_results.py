"""Generate comparison-table artifacts used in paper §5.

Outputs:
- ``results/<platform>_sweep.csv`` — raw latency numbers per (BW, RTT)
- stdout — pretty table for the paper

For each (bandwidth, RTT) we report five numbers:

    all_HE       — every op in HE, with mandatory bootstrapping when
                   the HE level budget would be exceeded. Corresponds
                   to the FHE / NEXUS baseline in paper §5.
    all_MPC      — every op in MPC with BOLT-style SCI primitives.
                   Corresponds to the MPC / SHAFT baseline.
    static_hybrid - fixed "linear@HE, nonlinear@MPC" assignment, the
                   policy used by prior hybrid work (BumbleBee, BOLT).
    compiler     — our state-expanded compiler plan for that setting.
    speedup      — (best-baseline) / compiler.

Run::

    python -m compiler.state_expanded.generate_results \
        --profile compiler/state_expanded/profiles/profile_cpu.json \
        --out results/cpu_sweep.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import List, Tuple

from .bert_graph import bert_block_graph, default_manifest
from .cost_model import NetworkSetting
from .planner import (
    compile_plan,
    evaluate_named_static_hybrids,
    evaluate_uniform_domain,
)
from .profile_schema import LatencyProfile
from .published_baselines import (
    ALL_PUBLISHED,
    FULL_MODEL_SCALE,
    evaluate_published_baselines,
    extrapolate_compiler_full_model,
)


BANDWIDTHS_MBPS: List[Tuple[str, float]] = [
    ("10Mbps", 10e6),
    ("100Mbps", 100e6),
    ("1Gbps", 1e9),
    ("3Gbps", 3e9),
]
RTTS_MS: List[float] = [1.0, 20.0, 40.0, 80.0]


def _fmt_ms(x: float) -> str:
    if math.isinf(x):
        return "inf"
    if x >= 1000:
        return f"{x/1000:.2f}s"
    return f"{x:.1f}ms"


def sweep(profile: LatencyProfile) -> List[dict]:
    graph = bert_block_graph(default_manifest())
    rows: List[dict] = []
    for bw_label, bw in BANDWIDTHS_MBPS:
        for rtt in RTTS_MS:
            net = NetworkSetting(bw, rtt)
            all_he = evaluate_uniform_domain(graph, profile, net, "HE")
            all_mpc = evaluate_uniform_domain(graph, profile, net, "MPC")
            hybrids = evaluate_named_static_hybrids(graph, profile, net)
            plan = compile_plan(graph, profile, net)
            comp = plan.total_cost_ms
            domain_mix = {"HE": 0, "MPC": 0}
            for _, d in plan.node_assignment.items():
                domain_mix[d] = domain_mix.get(d, 0) + 1
            baselines = [all_he, all_mpc] + [v for v in hybrids.values() if not math.isinf(v)]
            baselines = [b for b in baselines if not math.isinf(b)]
            best_base = min(baselines) if baselines else float("inf")
            speedup = best_base / comp if comp > 0 and not math.isinf(best_base) else float("inf")
            published = evaluate_published_baselines(net)
            rows.append({
                "bandwidth_label": bw_label,
                "bandwidth_bps": bw,
                "rtt_ms": rtt,
                # 1-block seq=16 scope (current compiler profile)
                "all_HE_ms": all_he,
                "all_MPC_ms": all_mpc,
                "hybrid_linHE_nlMPC_ms": hybrids["linHE_nlMPC"],
                "hybrid_linMPC_nlHE_ms": hybrids["linMPC_nlHE"],
                "hybrid_attnMPC_ffnHE_ms": hybrids["attnMPC_ffnHE"],
                "compiler_ms": comp,
                "he_nodes": domain_mix.get("HE", 0),
                "mpc_nodes": domain_mix.get("MPC", 0),
                "conversion_ms": plan.conversion_cost_ms,
                "bootstrap_ms": plan.bootstrap_cost_ms,
                "best_baseline_ms": best_base,
                "speedup_vs_best_baseline": speedup,
                # Full-model 12-block seq=128 scope (published baselines
                # + extrapolated compiler). NOTE: these numbers are
                # not directly comparable to the columns above which
                # are at seq=16 1-block scope.
                "fm_BumbleBee_ms":   published["BumbleBee"],
                "fm_BOLT_ms":        published["BOLT"],
                "fm_SHAFT_ms":       published["SHAFT"],
                "fm_NEXUS_FHE_ms":   published["NEXUS_FHE"],
                "fm_compiler_ms":    extrapolate_compiler_full_model(comp),
            })
    return rows


def print_table(rows: List[dict], platform: str) -> None:
    hdr = (
        f"{'BW':>8s} {'RTT':>6s} | "
        f"{'all-HE':>10s} {'all-MPC':>10s} "
        f"{'linHE_nlMPC':>12s} {'linMPC_nlHE':>12s} {'attnMPC_ffnHE':>14s} "
        f"{'compiler':>10s} {'speedup':>8s}  {'mix':>6s}"
    )
    print(f"[platform={platform}] -- scope: 1-block BERT @ seq=16 (compiler profile)")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['bandwidth_label']:>8s} {r['rtt_ms']:>5g}ms | "
            f"{_fmt_ms(r['all_HE_ms']):>10s} "
            f"{_fmt_ms(r['all_MPC_ms']):>10s} "
            f"{_fmt_ms(r['hybrid_linHE_nlMPC_ms']):>12s} "
            f"{_fmt_ms(r['hybrid_linMPC_nlHE_ms']):>12s} "
            f"{_fmt_ms(r['hybrid_attnMPC_ffnHE_ms']):>14s} "
            f"{_fmt_ms(r['compiler_ms']):>10s} "
            f"{r['speedup_vs_best_baseline']:>7.2f}x  "
            f"{r['he_nodes']:>2d}/{r['mpc_nodes']:>2d}"
        )


def print_full_model_table(rows: List[dict], platform: str) -> None:
    """Full-model reference table from the measured baseline runs.

    All four baselines (NEXUS FHE / SHAFT / BumbleBee / BOLT) were
    captured end-to-end on *this* machine on the full 12-block
    BERT-base at sequence length 128. ``compiler_fm*`` is a linear
    extrapolation of the compiler's 1-block seq=16 cost by
    ``FULL_MODEL_SCALE=96``; it is an optimistic first-order estimate
    (dominant terms scale linearly in S) and should be treated as an
    order-of-magnitude reference, not a direct measurement.
    """
    hdr = (
        f"{'BW':>8s} {'RTT':>6s} | "
        f"{'NEXUS(FHE)':>12s} {'SHAFT(MPC)':>12s} "
        f"{'BumbleBee':>12s} {'BOLT':>12s} "
        f"{'compiler*':>12s} {'vs_best':>9s}"
    )
    print(f"\n[platform={platform}] -- scope: full 12-block BERT @ seq=128 (measured baselines)")
    print("NOTE: NEXUS / SHAFT / BumbleBee / BOLT are real end-to-end runs")
    print("      recorded in baseline/*. compiler* is the 1-block seq=16")
    print("      compiler result scaled by 96 (linear in S, x12 blocks).")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        bests = [r["fm_NEXUS_FHE_ms"], r["fm_SHAFT_ms"], r["fm_BumbleBee_ms"], r["fm_BOLT_ms"]]
        best = min(bests) if bests else float("inf")
        comp_fm = r["fm_compiler_ms"]
        su = best / comp_fm if comp_fm > 0 else float("inf")
        print(
            f"{r['bandwidth_label']:>8s} {r['rtt_ms']:>5g}ms | "
            f"{_fmt_ms(r['fm_NEXUS_FHE_ms']):>12s} "
            f"{_fmt_ms(r['fm_SHAFT_ms']):>12s} "
            f"{_fmt_ms(r['fm_BumbleBee_ms']):>12s} "
            f"{_fmt_ms(r['fm_BOLT_ms']):>12s} "
            f"{_fmt_ms(comp_fm):>12s} "
            f"{su:>8.2f}x"
        )


def write_csv(rows: List[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True, type=str)
    ap.add_argument("--out", required=True, type=str)
    args = ap.parse_args()

    profile = LatencyProfile.load(args.profile)
    rows = sweep(profile)
    write_csv(rows, Path(args.out))
    print_table(rows, platform=profile.platform)
    print_full_model_table(rows, platform=profile.platform)
    print(f"\n[results] wrote {args.out}")


if __name__ == "__main__":
    main()
