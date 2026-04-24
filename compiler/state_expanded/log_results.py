"""Write the final comparison tables to ``results/<model>/``.

Two tables are produced per platform:

1. ``per_block_seq16.csv`` + ``.md``
   Apples-to-apples analytical comparison using the compiler's
   measured profile. All columns are composed with the same
   ``local + bytes*8/bw + rounds*rtt`` formula at each (bandwidth,
   RTT) setting. Scope: 1 BERT encoder block at seq_len=16.
   Strategies are descriptive labels (no paper names because each
   system below uses the repo's profile, not the paper's primitives):

     FHE_all_HE_with_bootstrap
     MPC_all_BOLT_primitives
     Hybrid_linHE_nlMPC
     Hybrid_linMPC_nlHE
     Hybrid_attnMPC_FFNHE
     Ours_state_expanded_compiler

2. ``full_model_seq128.csv`` + ``.md``
   Reference table populated from **real end-to-end runs** of each
   published baseline system, recorded in ``baseline/``. All four
   baselines are named by the paper/system they come from:

     NEXUS   (FHE)            [Zhang et al. 2024]
     SHAFT   (MPC)            [Kei et al. 2025]
     BumbleBee                [Lu et al. 2023]
     BOLT                     [Pang et al. 2024]
     Ours_compiler_extrapolated

   Scope: full 12-block BERT-base at seq_len=128.

Also writes ``README.md`` mapping each column to its source.
"""

from __future__ import annotations

import argparse
import csv
import math as math
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .bert_graph import (
    bert_block_graph,
    bert_multi_block_graph,
    default_manifest,
    full_model_manifest,
)
from .cost_model import NetworkSetting
from .planner import (
    compile_plan,
    compile_plan_safe,
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


BANDWIDTHS: List[Tuple[str, float]] = [
    ("10Mbps", 10e6),
    ("100Mbps", 100e6),
    ("1Gbps", 1e9),
    ("3Gbps", 3e9),
]
RTTS: List[float] = [1.0, 20.0, 40.0, 80.0]


# ----- column labels -----

PER_BLOCK_COLUMNS: List[Tuple[str, str]] = [
    ("FHE_all_HE_with_bootstrap",
     "Pure FHE baseline: every op runs in HE end-to-end, mandatory "
     "bootstrap inserted when the 20-level budget is exceeded. "
     "Same placement family as NEXUS [Zhang 2024]."),
    ("MPC_all_protocols",
     "Pure MPC baseline: every op runs under MPC protocols end-to-end. "
     "CPU profile uses the BOLT 2PC primitives we measured. "
     "GPU profile uses SHAFT-family (CrypTen-style 2PC on GPU) "
     "with bytes/rounds unchanged from BOLT and local_compute scaled "
     "by documented CrypTen GPU speedups."),
    ("Hybrid_linHE_nlMPC",
     "Static hybrid: linear ops -> HE, nonlinear ops -> MPC. "
     "Placement family of BumbleBee [Lu 2023] and BOLT [Pang 2024]."),
    ("Hybrid_linMPC_nlHE",
     "Static hybrid (inverse): linear ops -> MPC, nonlinear ops -> HE. "
     "Represents polynomial-approximation families (e.g. CryptoNets)."),
    ("Hybrid_attnMPC_FFNHE",
     "Static hybrid: attention block -> MPC, FFN block -> HE."),
    ("Ours_state_expanded_compiler",
     "Our compiler (state-expanded shortest path + SESE hierarchy, "
     "paper Section 4.2, with safety-net fallback)."),
]

PER_BLOCK_REF_COLUMNS: List[Tuple[str, str]] = [
    ("ref_NEXUS_per_block",
     "[Zhang et al. 2024] measured 12-block seq=128 end-to-end, "
     "divided by 96 for per-block comparison. Scope-adjusted reference."),
    ("ref_SHAFT_per_block",
     "[Kei et al. 2025] measured 12-block seq=128 end-to-end, /96. "
     "Scope-adjusted reference."),
    ("ref_BumbleBee_per_block",
     "[Lu et al. 2023] measured 12-block seq=128 end-to-end, /96. "
     "Scope-adjusted reference."),
    ("ref_BOLT_per_block",
     "[Pang et al. 2024] measured 12-block seq=128 end-to-end, /96 "
     "(BLB run). Scope-adjusted reference."),
]


FULL_MODEL_SAME_STACK_COLUMNS: List[Tuple[str, str]] = [
    ("SameStack_FHE",
     "All operators in HE with mandatory bootstrap; same NEXUS CKKS "
     "primitives as the compiler. Directly comparable."),
    ("SameStack_MPC",
     "All operators in MPC via SCI BOLT bridges; same primitives as "
     "the compiler. Directly comparable."),
    ("SameStack_linHE_nlMPC",
     "Static: linear@HE, nonlinear@MPC. Placement family of BumbleBee "
     "[Lu 2023] and BOLT [Pang 2024], evaluated against our profile."),
    ("SameStack_linMPC_nlHE",
     "Static: linear@MPC, nonlinear@HE. Inverse (CryptoNets-style)."),
    ("SameStack_attnMPC_FFNHE",
     "Static: attention block@MPC, FFN block@HE."),
    ("Ours_compiler",
     "State-expanded compiler. Guaranteed never worse than any "
     "SameStack column by `compile_plan_safe`."),
]

FULL_MODEL_PUBLISHED_COLUMNS: List[Tuple[str, str]] = [
    ("Published_NEXUS",
     "[Zhang et al. 2024] real CKKS FHE end-to-end estimate (different "
     "bootstrap policy than our profile)."),
    ("Published_SHAFT",
     "[Kei et al. 2025] real CrypTen+beaver MPC end-to-end (different "
     "MPC protocol stack than our SCI BOLT)."),
    ("Published_BumbleBee",
     "[Lu et al. 2023] real CHEETAH 2PC end-to-end (different HE and "
     "MPC primitives than ours)."),
    ("Published_BOLT",
     "[Pang et al. 2024] real BOLT-linked hybrid end-to-end (BOLT's HE "
     "matmul kernel is faster than NEXUS at BERT shapes)."),
]


def _fmt_ms(x: float) -> str:
    if math.isinf(x):
        return "inf"
    if x >= 1000:
        return f"{x/1000:.2f}s"
    return f"{x:.1f}ms"


def _sweep_per_block(profile: LatencyProfile) -> List[Dict[str, float]]:
    graph = bert_block_graph(default_manifest())
    rows: List[Dict[str, float]] = []
    for bw_label, bw in BANDWIDTHS:
        for rtt in RTTS:
            net = NetworkSetting(bw, rtt)
            all_he = evaluate_uniform_domain(graph, profile, net, "HE")
            all_mpc = evaluate_uniform_domain(graph, profile, net, "MPC")
            hybrids = evaluate_named_static_hybrids(graph, profile, net)
            plan = compile_plan_safe(graph, profile, net)
            best_baseline = min(
                all_he, all_mpc,
                hybrids["linHE_nlMPC"],
                hybrids["linMPC_nlHE"],
                hybrids["attnMPC_ffnHE"],
            )
            speedup = best_baseline / plan.total_cost_ms if plan.total_cost_ms > 0 else float("inf")

            # Reference full-model baselines, scope-scaled down to
            # 1-block seq=16 by dividing by FULL_MODEL_SCALE=96. Each
            # column here is "what that system would take per block",
            # roughly — kept for context; the primary comparison remains
            # the same-stack columns.
            published = evaluate_published_baselines(net)
            ref_BumbleBee = published["BumbleBee"] / FULL_MODEL_SCALE
            ref_BOLT      = published["BOLT"]      / FULL_MODEL_SCALE
            ref_SHAFT     = published["SHAFT"]     / FULL_MODEL_SCALE
            ref_NEXUS     = published["NEXUS_FHE"] / FULL_MODEL_SCALE

            rows.append({
                "bandwidth_label": bw_label,
                "bandwidth_bps": bw,
                "rtt_ms": rtt,
                "FHE_all_HE_with_bootstrap":     all_he,
                "MPC_all_protocols":             all_mpc,
                "Hybrid_linHE_nlMPC":            hybrids["linHE_nlMPC"],
                "Hybrid_linMPC_nlHE":            hybrids["linMPC_nlHE"],
                "Hybrid_attnMPC_FFNHE":          hybrids["attnMPC_ffnHE"],
                "Ours_state_expanded_compiler":  plan.total_cost_ms,
                "speedup_vs_best_baseline":      speedup,
                "strategy_used":                 plan.strategy_used,
                "mix_HE":       sum(1 for d in plan.node_assignment.values() if d == "HE"),
                "mix_MPC":      sum(1 for d in plan.node_assignment.values() if d == "MPC"),
                "conv_ms":      plan.conversion_cost_ms,
                "bootstrap_ms": plan.bootstrap_cost_ms,
                # Reference: published end-to-end baselines scaled to
                # 1-block scope by /96 (scope context, not direct comp).
                "ref_NEXUS_per_block":     ref_NEXUS,
                "ref_SHAFT_per_block":     ref_SHAFT,
                "ref_BumbleBee_per_block": ref_BumbleBee,
                "ref_BOLT_per_block":      ref_BOLT,
            })
    return rows


def _sweep_full_model(profile: LatencyProfile) -> List[Dict[str, float]]:
    """Full 12-block seq=128 sweep with both same-stack and published baselines.

    Same-stack baselines (FHE / MPC / three hybrids) are evaluated
    against the compiler's own profile — directly comparable because
    every column uses identical primitive costs. The compiler's output
    is GUARANTEED never worse than any same-stack baseline
    (``compile_plan_safe`` ensures this at runtime).

    Published baselines (NEXUS / SHAFT / BumbleBee / BOLT) are real
    end-to-end runs of different systems with different primitive
    stacks. They're kept for external context, not as a placement
    comparison.
    """
    graph = bert_multi_block_graph(12, full_model_manifest())
    rows: List[Dict[str, float]] = []
    for bw_label, bw in BANDWIDTHS:
        for rtt in RTTS:
            net = NetworkSetting(bw, rtt)

            # Same-stack baselines (our profile).
            all_he  = evaluate_uniform_domain(graph, profile, net, "HE")
            all_mpc = evaluate_uniform_domain(graph, profile, net, "MPC")
            hybrids = evaluate_named_static_hybrids(graph, profile, net)
            plan = compile_plan_safe(graph, profile, net)

            same_stack_best = min(
                all_he, all_mpc,
                hybrids["linHE_nlMPC"],
                hybrids["linMPC_nlHE"],
                hybrids["attnMPC_ffnHE"],
            )
            vs_same_stack = (
                same_stack_best / plan.total_cost_ms
                if plan.total_cost_ms > 0 else float("inf")
            )

            # Published real-system baselines (different primitives).
            published = evaluate_published_baselines(net)
            ext_best_candidates = [v for v in published.values() if not math.isinf(v)]
            ext_best = min(ext_best_candidates) if ext_best_candidates else float("inf")
            vs_external = (
                ext_best / plan.total_cost_ms
                if plan.total_cost_ms > 0 else float("inf")
            )

            rows.append({
                "bandwidth_label": bw_label,
                "bandwidth_bps": bw,
                "rtt_ms": rtt,
                # Same-stack baselines: fair placement comparison.
                "SameStack_FHE":           all_he,
                "SameStack_MPC":           all_mpc,
                "SameStack_linHE_nlMPC":   hybrids["linHE_nlMPC"],
                "SameStack_linMPC_nlHE":   hybrids["linMPC_nlHE"],
                "SameStack_attnMPC_FFNHE": hybrids["attnMPC_ffnHE"],
                # Published baselines: external context.
                "Published_NEXUS":         published["NEXUS_FHE"],
                "Published_SHAFT":         published["SHAFT"],
                "Published_BumbleBee":     published["BumbleBee"],
                "Published_BOLT":          published["BOLT"],
                # Compiler + guarantee metrics.
                "Ours_compiler":             plan.total_cost_ms,
                "strategy_used":             plan.strategy_used,
                "speedup_vs_same_stack":     vs_same_stack,
                "speedup_vs_external_best":  vs_external,
                "mix_HE":  sum(1 for d in plan.node_assignment.values() if d == "HE"),
                "mix_MPC": sum(1 for d in plan.node_assignment.values() if d == "MPC"),
            })
    return rows


def _write_csv(rows: List[Dict[str, float]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _write_md_per_block(
    rows: List[Dict[str, float]], platform: str, path: Path,
    profile: Optional[LatencyProfile] = None,
) -> None:
    lines: List[str] = []
    lines.append(f"# Per-block BERT results ({platform.upper()})")
    lines.append("")
    lines.append("Scope: **1 BERT encoder block, seq_len=16**.")
    lines.append("")
    if profile is not None:
        he_label = profile.metadata.get("he_stack_label", "HE stack")
        mpc_label = profile.metadata.get("mpc_stack_label", "MPC stack")
        lines.append(f"**HE primitives**: {he_label}")
        lines.append(f"**MPC primitives**: {mpc_label}")
        lines.append("")
    lines.append("`speedup` is `min(every same-stack baseline) / compiler`. The column ")
    lines.append("is **guaranteed ≥ 1.00** by `compile_plan_safe`: if the state-expanded")
    lines.append("compiler would ever produce a plan worse than any static baseline")
    lines.append("under the cost model, the safety net falls back to that baseline.")
    lines.append("")
    lines.append("## Same-stack placement comparison")
    lines.append("")
    cols = [c for c, _ in PER_BLOCK_COLUMNS]
    header = (
        "| BW | RTT | "
        + " | ".join(c.replace("_", " ") for c in cols)
        + " | speedup | strategy | HE/MPC | Conv | BS |"
    )
    sep = "|" + "|".join(["---"] * (3 + len(cols) + 4)) + "|"
    lines.append(header)
    lines.append(sep)
    for r in rows:
        cells = [
            r["bandwidth_label"],
            f"{r['rtt_ms']:g}ms",
        ] + [_fmt_ms(r[c]) for c in cols] + [
            f"{r['speedup_vs_best_baseline']:.2f}x",
            r.get("strategy_used", "compiler"),
            f"{r['mix_HE']}/{r['mix_MPC']}",
            _fmt_ms(r["conv_ms"]),
            _fmt_ms(r["bootstrap_ms"]),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## Published-baseline reference (scope-adjusted to per-block)")
    lines.append("")
    lines.append("The four published systems ran full 12-block BERT-base at ")
    lines.append("seq=128; values here are divided by 96 (12 blocks × seq-ratio 8) ")
    lines.append("for per-block context. They use different primitive stacks than")
    lines.append("our compiler, so this is a scope-adjusted external reference, ")
    lines.append("not a direct placement comparison.")
    lines.append("")
    ref_cols = [c for c, _ in PER_BLOCK_REF_COLUMNS]
    hdr2 = "| BW | RTT | " + " | ".join(c.replace("_", " ") for c in ref_cols) + " |"
    sep2 = "|" + "|".join(["---"] * (3 + len(ref_cols))) + "|"
    lines.append(hdr2)
    lines.append(sep2)
    for r in rows:
        cells = [
            r["bandwidth_label"],
            f"{r['rtt_ms']:g}ms",
        ] + [_fmt_ms(r[c]) for c in ref_cols]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## Column legend")
    lines.append("")
    lines.append("### Same-stack columns (primary comparison)")
    for name, desc in PER_BLOCK_COLUMNS:
        lines.append(f"- **{name.replace('_', ' ')}** — {desc}")
    lines.append("- **speedup** — `min(every same-stack baseline) / Ours`. "
                 "Always ≥ 1.00 by construction (safety-net fallback).")
    lines.append("- **strategy** — which policy produced the reported cost: "
                 "`compiler` if the state-expanded plan was best, else the name "
                 "of the static baseline that tied or beat it.")
    lines.append("")
    lines.append("### Published-reference columns (scope-adjusted)")
    for name, desc in PER_BLOCK_REF_COLUMNS:
        lines.append(f"- **{name.replace('_', ' ')}** — {desc}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def _write_md_full_model(rows: List[Dict[str, float]], platform: str, path: Path) -> None:
    lines: List[str] = []
    lines.append(f"# Full-model BERT results ({platform.upper()})")
    lines.append("")
    lines.append("Scope: **full 12-block BERT-base, seq_len=128**.")
    lines.append("")
    lines.append("## Placement comparison (same primitive stack)")
    lines.append("")
    lines.append("Every column below uses the same real NEXUS CKKS profile and ")
    lines.append("the same real SCI BOLT MPC profile as the compiler — only the ")
    lines.append("placement policy differs. `Ours_compiler` is **guaranteed** ")
    lines.append("never slower than any `SameStack_*` baseline by")
    lines.append("`compile_plan_safe`: if the state-expanded search ever fails")
    lines.append("to beat a static policy under this cost model, the safety net")
    lines.append("falls back to that policy.")
    lines.append("")
    ss_cols = [c for c, _ in FULL_MODEL_SAME_STACK_COLUMNS]
    header = (
        "| BW | RTT | "
        + " | ".join(ss_cols)
        + " | speedup | strategy |"
    )
    sep = "|" + "|".join(["---"] * (2 + len(ss_cols) + 3)) + "|"
    lines.append(header)
    lines.append(sep)
    for r in rows:
        cells = [
            r["bandwidth_label"],
            f"{r['rtt_ms']:g}ms",
        ] + [_fmt_ms(r[c]) for c in ss_cols] + [
            f"{r['speedup_vs_same_stack']:.2f}x",
            r.get("strategy_used", "compiler"),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## External reference (different primitive stacks)")
    lines.append("")
    lines.append("These numbers are real end-to-end runs of four published ")
    lines.append("systems on this machine. Each uses a different HE or MPC ")
    lines.append("primitive stack than ours, so the comparison mixes placement ")
    lines.append("and primitive-level differences. Shown for context only — they ")
    lines.append("are not the paper's direct point of comparison for placement.")
    lines.append("")
    pub_cols = [c for c, _ in FULL_MODEL_PUBLISHED_COLUMNS]
    header = (
        "| BW | RTT | "
        + " | ".join(pub_cols)
        + " | Ours_compiler | vs_external |"
    )
    sep = "|" + "|".join(["---"] * (2 + len(pub_cols) + 3)) + "|"
    lines.append(header)
    lines.append(sep)
    for r in rows:
        cells = [
            r["bandwidth_label"],
            f"{r['rtt_ms']:g}ms",
        ] + [_fmt_ms(r[c]) for c in pub_cols] + [
            _fmt_ms(r["Ours_compiler"]),
            f"{r['speedup_vs_external_best']:.2f}x",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## Column legend")
    lines.append("")
    lines.append("### Same-stack (direct placement comparison)")
    for name, desc in FULL_MODEL_SAME_STACK_COLUMNS:
        lines.append(f"- **{name}** — {desc}")
    lines.append("")
    lines.append("### Published external baselines")
    for name, desc in FULL_MODEL_PUBLISHED_COLUMNS:
        lines.append(f"- **{name}** — {desc}")
    lines.append("")
    lines.append("### Derived metrics")
    lines.append("- **speedup** (same-stack) — `min(SameStack_*) / Ours_compiler`. ")
    lines.append("  **Always ≥ 1.00** by construction.")
    lines.append("- **vs_external** — `min(Published_*) / Ours_compiler`. Can be ")
    lines.append("  less than 1.00 when an external system has a faster ")
    lines.append("  primitive stack (e.g. BOLT's HE matmul is roughly 10× faster ")
    lines.append("  than NEXUS CKKS at BERT shapes). This reflects a primitive ")
    lines.append("  gap, not a placement gap.")
    lines.append("- **strategy** — `compiler` when the state-expanded plan was ")
    lines.append("  selected; otherwise the name of the static policy that tied ")
    lines.append("  or beat it (which would indicate a compiler bug).")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def _write_readme(path: Path) -> None:
    content = f"""# BERT-base result tables

## Files

- `per_block_seq16_cpu.csv`, `per_block_seq16_cpu.md`
  Apples-to-apples analytical comparison at 1-block seq=16 scope on CPU.
- `per_block_seq16_gpu.csv`, `per_block_seq16_gpu.md`
  Same comparison on GPU (NEXUS-CUDA speedups derived from CPU profile).
- `full_model_seq128.csv`, `full_model_seq128.md`
  Real end-to-end measurements of four published baselines at 12-block
  seq=128 scope, swept across network settings.
- `res_cpu.png`, `res_gpu.png`
  Line plots of the per-block table (log y-axis, one subplot per BW).

## Strategy naming convention

- Where a strategy corresponds exactly to a published system, the
  column uses the paper/system name (e.g. `BumbleBee`, `BOLT`, `SHAFT`,
  `NEXUS`).
- Where a strategy is evaluated *against our measured profile* rather
  than from a specific published run, the column uses a descriptive
  label (e.g. `Hybrid_linHE_nlMPC`) with the corresponding paper family
  cited in the column legend of each table.

## Provenance

- HE numbers: real NEXUS CKKS wallclock from `NEXUS/build/bin/main`.
- MPC numbers: real SCI BOLT wallclock + bytes + rounds from our
  instrumented bridges (see `bridge/*.cpp` and
  `compiler/state_expanded/profiler_mpc.py`).
- Bootstrap cost: real `NEXUS/build/bin/bootstrapping`.
- Full-model baselines: real runs recorded under `baseline/`.

## Reproduce

```
# 1. build per-block profile (seq=16) on CPU
python -m compiler.state_expanded._assemble_profile \\
    --device cpu --scope per_block \\
    --output compiler/state_expanded/profiles/profile_cpu.json

# 2. build full-model profile (seq=128) on CPU
python -m compiler.state_expanded._assemble_profile \\
    --device cpu --scope full_model \\
    --output compiler/state_expanded/profiles/profile_cpu_full_model.json

# 3. derive GPU profile from CPU (NEXUS-CUDA speedups)
python -m compiler.state_expanded._assemble_profile \\
    --device cuda \\
    --from-cpu-profile compiler/state_expanded/profiles/profile_cpu.json \\
    --output compiler/state_expanded/profiles/profile_gpu.json

# 4. log all tables + plots
python -m compiler.state_expanded.log_results \\
    --cpu-profile compiler/state_expanded/profiles/profile_cpu.json \\
    --gpu-profile compiler/state_expanded/profiles/profile_gpu.json \\
    --full-model-profile compiler/state_expanded/profiles/profile_cpu_full_model.json \\
    --out-dir compiler/state_expanded/results/bert_base
```
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu-profile", required=True,
                    help="per-block seq=16 CPU profile (profile_cpu.json)")
    ap.add_argument("--gpu-profile", required=True,
                    help="per-block seq=16 GPU profile (profile_gpu.json)")
    ap.add_argument("--full-model-profile", required=True,
                    help="full-model seq=128 CPU profile "
                         "(profile_cpu_full_model.json)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--plot-src-dir",
                    default="compiler/state_expanded/results",
                    help="directory containing res_cpu.png / res_gpu.png to copy")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cpu_profile = LatencyProfile.load(args.cpu_profile)
    gpu_profile = LatencyProfile.load(args.gpu_profile)
    full_profile = LatencyProfile.load(args.full_model_profile)

    per_block_cpu = _sweep_per_block(cpu_profile)
    per_block_gpu = _sweep_per_block(gpu_profile)
    full_model = _sweep_full_model(full_profile)

    _write_csv(per_block_cpu, out / "per_block_seq16_cpu.csv")
    _write_csv(per_block_gpu, out / "per_block_seq16_gpu.csv")
    _write_csv(full_model,    out / "full_model_seq128.csv")

    _write_md_per_block(per_block_cpu, "cpu", out / "per_block_seq16_cpu.md", profile=cpu_profile)
    _write_md_per_block(per_block_gpu, "gpu", out / "per_block_seq16_gpu.md", profile=gpu_profile)
    _write_md_full_model(full_model, "cpu+gpu (hw-independent refs)",
                         out / "full_model_seq128.md")

    _write_readme(out / "README.md")

    plot_src = Path(args.plot_src_dir)
    for name in ("res_cpu.png", "res_gpu.png"):
        src = plot_src / name
        if src.exists():
            shutil.copyfile(src, out / name)

    print(f"[log_results] wrote tables to {out}")
    for p in sorted(out.glob("*")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
