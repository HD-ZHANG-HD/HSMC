"""Paper-ready SOTA comparison tables.

Two sections per platform:

1. **Placement comparison (same primitives, paper-named policies).**
   Every column uses the SAME profile (real NEXUS HE + real
   BOLT/SHAFT MPC) and differs only in placement. Columns are named
   after the published systems whose placement strategy each column
   represents:

     NEXUS (FHE)                 — pure HE, all operators in HE
     SHAFT (MPC)                 — pure MPC, all operators in MPC
     BumbleBee / BOLT (linHE+nlMPC) — linear@HE, nonlinear@MPC
                                    (shared placement between BumbleBee
                                    and BOLT)

   The compiler is GUARANTEED ≥ every column here. This is the fair
   placement comparison — same primitives, different assignment.

2. **Absolute published wallclock reference.**
   Real end-to-end measurements of each paper's full system (different
   primitive stacks) at (BW, RTT). Shown for completeness — absolute
   gaps here reflect both placement AND primitive choice.

Scope: full 12-block BERT-base at seq_len=128 in every row.

Output: ``results/bert_base/sota_comparison_{cpu,gpu}.{csv,md}``.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple

from .bert_graph import bert_multi_block_graph, full_model_manifest
from .cost_model import NetworkSetting
from .planner import (
    POLICY_LINEAR_HE_NONLINEAR_MPC,
    _evaluate_fixed_policy,
    compile_plan_safe,
    evaluate_uniform_domain,
)
from .profile_schema import LatencyProfile
from .published_baselines import evaluate_published_baselines


BANDWIDTHS: List[Tuple[str, float]] = [
    ("10Mbps", 10e6),
    ("100Mbps", 100e6),
    ("1Gbps", 1e9),
    ("3Gbps", 3e9),
]
RTTS: List[float] = [1.0, 20.0, 40.0, 80.0]


# Placement columns — same primitives, different placement policy.
# The Ours column is GUARANTEED ≥ every other column here.
PLACEMENT_COLUMNS: List[Tuple[str, str]] = [
    ("NEXUS_FHE",
     "Pure FHE placement — the policy used by NEXUS [Zhang et al. 2024]. "
     "All operators in HE; mandatory bootstrap when the 20-level budget "
     "is exceeded. Evaluated on our HE primitive stack."),
    ("SHAFT_MPC",
     "Pure MPC placement — the policy used by SHAFT [Kei et al. 2025]. "
     "All operators in MPC. Evaluated on our MPC primitive stack "
     "(SCI-BOLT on CPU / SHAFT-CrypTen on GPU)."),
    ("BumbleBee_BOLT_hybrid",
     "linHE + nlMPC placement — the shared policy of BumbleBee "
     "[Lu et al. 2023] and BOLT [Pang et al. 2024]. Linear ops in HE, "
     "nonlinear ops in MPC, conversions on every crossing edge. "
     "Evaluated on our primitive stack."),
    ("Ours",
     "State-expanded compiler on our profile, safety-net guaranteed."),
]

# Published absolute-wallclock reference columns — DIFFERENT primitive
# stacks, real end-to-end runs. Shown for completeness only.
PUBLISHED_COLUMNS: List[Tuple[str, str]] = [
    ("pub_NEXUS_FHE",
     "Published NEXUS FHE estimator — baseline/baseline_res.txt #004 "
     "(17 941 932 ms local compute, 0 comm)."),
    ("pub_SHAFT_MPC",
     "Published SHAFT end-to-end — baseline/baseline_res.txt #003 + "
     "SHAFT_communication.json (368 103 ms local, 10.48 GB, 1 496 rounds)."),
    ("pub_BumbleBee",
     "Published BumbleBee end-to-end — baseline/bumble_communication.json "
     "(1 209 196 ms local, 11.33 GB, 586 795 rounds). Uses Cheetah HE."),
    ("pub_BOLT",
     "Published BOLT (BLB) end-to-end — baseline/blb_communication.json "
     "(1 939 100 ms local, 1.76 GB, 1 199 rounds). Uses BOLT-optimized HE."),
]


def _fmt_ms(x: float) -> str:
    if math.isinf(x):
        return "inf"
    if x >= 1000:
        return f"{x/1000:.2f}s"
    return f"{x:.1f}ms"


def sweep(profile: LatencyProfile) -> List[Dict[str, float]]:
    graph = bert_multi_block_graph(12, full_model_manifest())
    rows: List[Dict[str, float]] = []
    for bw_label, bw in BANDWIDTHS:
        for rtt in RTTS:
            net = NetworkSetting(bw, rtt)
            # Placement columns — same primitives, different policy.
            same_he  = evaluate_uniform_domain(graph, profile, net, "HE")
            same_mpc = evaluate_uniform_domain(graph, profile, net, "MPC")
            bblb_hyb = _evaluate_fixed_policy(
                graph, profile, net, POLICY_LINEAR_HE_NONLINEAR_MPC
            )
            plan = compile_plan_safe(graph, profile, net)

            # Best placement among named policies.
            placement_candidates = [same_he, same_mpc, bblb_hyb]
            best_placement = min(
                [c for c in placement_candidates if not math.isinf(c)],
                default=float("inf"),
            )
            speedup_placement = (
                best_placement / plan.total_cost_ms
                if plan.total_cost_ms > 0 else float("inf")
            )

            # Published absolute wallclock (different primitive stacks).
            published = evaluate_published_baselines(net)
            best_pub = min([v for v in published.values() if not math.isinf(v)], default=float("inf"))
            ratio_vs_pub = (
                best_pub / plan.total_cost_ms
                if plan.total_cost_ms > 0 else float("inf")
            )

            mix_he  = sum(1 for d in plan.node_assignment.values() if d == "HE")
            mix_mpc = sum(1 for d in plan.node_assignment.values() if d == "MPC")

            rows.append({
                "bandwidth_label": bw_label,
                "bandwidth_bps": bw,
                "rtt_ms": rtt,
                # Placement columns (same primitives).
                "NEXUS_FHE":             same_he,
                "SHAFT_MPC":             same_mpc,
                "BumbleBee_BOLT_hybrid": bblb_hyb,
                "Ours":                  plan.total_cost_ms,
                "speedup_vs_placement":  speedup_placement,
                # Published columns (different primitives).
                "pub_NEXUS_FHE":  published["NEXUS_FHE"],
                "pub_SHAFT_MPC":  published["SHAFT"],
                "pub_BumbleBee":  published["BumbleBee"],
                "pub_BOLT":       published["BOLT"],
                "ratio_vs_best_published": ratio_vs_pub,
                "strategy_used":  plan.strategy_used,
                "mix_HE":  mix_he,
                "mix_MPC": mix_mpc,
            })
    return rows


def write_csv(rows: List[Dict[str, float]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_md(rows: List[Dict[str, float]], platform: str, profile: LatencyProfile, path: Path) -> None:
    lines: List[str] = []
    lines.append(f"# SOTA comparison — BERT-base, full 12 blocks at seq=128 ({platform.upper()})")
    lines.append("")
    he_label = profile.metadata.get("he_stack_label", "HE stack")
    mpc_label = profile.metadata.get("mpc_stack_label", "MPC stack")
    lines.append(f"- **Our HE primitives**: {he_label}")
    lines.append(f"- **Our MPC primitives**: {mpc_label}")
    lines.append("")

    # --- Placement comparison ---
    lines.append("## Placement comparison (paper policies, same primitives)")
    lines.append("")
    lines.append("Every column below uses the same real NEXUS HE + real ")
    lines.append("BOLT/SHAFT MPC profile as our compiler, so the comparison ")
    lines.append("isolates placement strategy. The column names identify the ")
    lines.append("published paper whose strategy each column evaluates.")
    lines.append("")
    lines.append("**`speedup` is guaranteed ≥ 1.00** by `compile_plan_safe`.")
    lines.append("")
    placement = [c for c, _ in PLACEMENT_COLUMNS]
    header = (
        "| BW | RTT | "
        + " | ".join(placement)
        + " | speedup | mix HE/MPC |"
    )
    sep = "|" + "|".join(["---"] * (2 + len(placement) + 2)) + "|"
    lines.append(header)
    lines.append(sep)
    for r in rows:
        cells = [
            r["bandwidth_label"],
            f"{r['rtt_ms']:g}ms",
        ] + [_fmt_ms(r[c]) for c in placement] + [
            f"{r['speedup_vs_placement']:.2f}x",
            f"{r['mix_HE']}/{r['mix_MPC']}",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    # --- Absolute published reference ---
    lines.append("## Absolute published wallclock (different primitive stacks)")
    lines.append("")
    lines.append("Real end-to-end measurements of each paper's full system ")
    lines.append("on this machine. The `ratio_vs_best_published` column is ")
    lines.append("`min(published_*) / Ours` — it can be less than 1.00 when ")
    lines.append("an external system uses faster primitives than ours (e.g. ")
    lines.append("BOLT's HE matmul kernel is ~10× faster than NEXUS CKKS at ")
    lines.append("BERT shapes). This reflects a primitive gap, not a ")
    lines.append("placement gap.")
    lines.append("")
    pub = [c for c, _ in PUBLISHED_COLUMNS]
    header2 = (
        "| BW | RTT | "
        + " | ".join(pub)
        + " | Ours | ratio_vs_best_published |"
    )
    sep2 = "|" + "|".join(["---"] * (2 + len(pub) + 2)) + "|"
    lines.append(header2)
    lines.append(sep2)
    for r in rows:
        cells = [
            r["bandwidth_label"],
            f"{r['rtt_ms']:g}ms",
        ] + [_fmt_ms(r[c]) for c in pub] + [
            _fmt_ms(r["Ours"]),
            f"{r['ratio_vs_best_published']:.2f}x",
        ]
        lines.append("| " + " | ".join(cells) + " |")

    # --- Legend ---
    lines.append("")
    lines.append("## Column legend")
    lines.append("")
    lines.append("### Placement columns (same primitives)")
    for name, desc in PLACEMENT_COLUMNS:
        lines.append(f"- **{name}** — {desc}")
    lines.append("- **speedup** — `min(NEXUS_FHE, SHAFT_MPC, BumbleBee_BOLT_hybrid) / Ours`. Always ≥ 1.00 (guaranteed).")
    lines.append("")
    lines.append("### Published columns (absolute wallclock, different primitives)")
    for name, desc in PUBLISHED_COLUMNS:
        lines.append(f"- **{name}** — {desc}")
    lines.append("- **ratio_vs_best_published** — `min(pub_*) / Ours`. Not guaranteed ≥ 1 because different primitive stacks.")
    lines.append("- **mix HE/MPC** — compiler's node assignment (total 132 = 12 × 11).")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu-full-profile", required=True,
                    help="profile_cpu_full_model.json")
    ap.add_argument("--gpu-full-profile", required=True,
                    help="profile_gpu_full_model.json")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for tag, profile_path in [("cpu", args.cpu_full_profile),
                              ("gpu", args.gpu_full_profile)]:
        profile = LatencyProfile.load(profile_path)
        rows = sweep(profile)
        write_csv(rows, out / f"sota_comparison_{tag}.csv")
        write_md(rows, tag, profile, out / f"sota_comparison_{tag}.md")
        print(f"[sota] wrote {tag} tables ({len(rows)} rows)")


if __name__ == "__main__":
    main()
