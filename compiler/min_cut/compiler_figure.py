from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from compiler.min_cut.cost_model import CostModel
    from compiler.min_cut.domain_assignment import assign_domains_min_cut, load_graph_json
    from compiler.min_cut.plan_builder import build_execution_plan
    from compiler.min_cut.profiler_db import ProfilerDB
else:
    from .cost_model import CostModel
    from .domain_assignment import assign_domains_min_cut, load_graph_json
    from .plan_builder import build_execution_plan
    from .profiler_db import ProfilerDB


def _collect_case_times(case: Dict[str, str], base_dir: Path) -> Dict[str, float]:
    profiler_file = base_dir / "test" / case["profiler_json"]
    graph_file = base_dir / "test" / case["graph_json"]

    db = ProfilerDB.from_json(profiler_file)
    cost_model = CostModel(db=db, default_strategy="auto")
    graph = load_graph_json(graph_file)

    result = assign_domains_min_cut(graph, cost_model)
    plan = build_execution_plan(graph, result.assignment, cost_model, include_baselines=True)

    optimized = float(plan["cost_breakdown"]["total_cost_ms"])  # type: ignore[index]
    baselines = plan["baselines"]  # type: ignore[index]

    return {
        "optimized": optimized,
        "all_he": float(baselines["all_he_total_ms"]),
        "all_mpc": float(baselines["all_mpc_total_ms"]),
        "hybrid": float(baselines["hybrid_linear_he_nonlinear_mpc_total_ms"]),
    }


def generate_figure(output_file: str = "compiler_case_times.png") -> Path:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required. Install it with: pip install matplotlib"
        ) from exc

    here = Path(__file__).resolve().parent
    case_path = here / "test" / "cases.json"
    cases: List[Dict[str, str]] = json.loads(case_path.read_text())["cases"]

    labels: List[str] = []
    optimized_vals: List[float] = []
    all_he_vals: List[float] = []
    all_mpc_vals: List[float] = []
    hybrid_vals: List[float] = []

    for case in cases:
        labels.append(case["name"])
        times = _collect_case_times(case, here)
        optimized_vals.append(times["optimized"])
        all_he_vals.append(times["all_he"])
        all_mpc_vals.append(times["all_mpc"])
        hybrid_vals.append(times["hybrid"])

    x_positions = list(range(len(labels)))
    width = 0.2

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.8), 6))
    ax.bar([x - 1.5 * width for x in x_positions], all_he_vals, width=width, label="Pure HE")
    ax.bar([x - 0.5 * width for x in x_positions], all_mpc_vals, width=width, label="Pure MPC")
    ax.bar([x + 0.5 * width for x in x_positions], hybrid_vals, width=width, label="Hybrid Manual")
    ax.bar([x + 1.5 * width for x in x_positions], optimized_vals, width=width, label="Min-Cut Compiler")

    ax.set_title("Min-Cut Compiler Case Runtime Comparison")
    ax.set_ylabel("LLatency (ms)")
    ax.set_xlabel("Cases")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()

    output_path = here / output_file
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def main() -> None:
    output_path = generate_figure()
    print(f"Saved figure: {output_path}")


if __name__ == "__main__":
    main()
