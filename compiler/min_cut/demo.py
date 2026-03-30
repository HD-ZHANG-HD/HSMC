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


def _print_case_header(case: Dict[str, str]) -> None:
    print("\n" + "=" * 78)
    print(f"[case] {case['name']}")
    print(f"[desc] {case['description']}")
    print("=" * 78)


def _print_node_costs(result) -> None:
    print("[node_costs]")
    for node_id, costs in sorted(result.per_node_costs.items()):
        print(f"  - {node_id}: HE={costs['HE']:.3f} ms, MPC={costs['MPC']:.3f} ms")


def _print_assignment(result) -> None:
    print("[assignment]")
    for node_id, domain in sorted(result.assignment.items()):
        print(f"  - {node_id}: {domain}")


def _print_plan(plan: Dict[str, object]) -> None:
    print("[plan_steps]")
    for step in plan["steps"]:  # type: ignore[index]
        kind = step["kind"]  # type: ignore[index]
        if kind == "operator":
            print(
                "  - {sid} operator {nid}:{op}@{dom} {lat:.3f} ms".format(
                    sid=step["step_id"],
                    nid=step["node_id"],
                    op=step["op_type"],
                    dom=step["domain"],
                    lat=step["estimated_latency_ms"],
                )
            )
        else:
            print(
                "  - {sid} conversion {src}->{dst} {fd}->{td} {lat:.3f} ms".format(
                    sid=step["step_id"],
                    src=step["from_node"],
                    dst=step["to_node"],
                    fd=step["from_domain"],
                    td=step["to_domain"],
                    lat=step["estimated_latency_ms"],
                )
            )
    cb = plan["cost_breakdown"]  # type: ignore[index]
    print(
        "[cost] nodes={n:.3f} ms, conv={c:.3f} ms, total={t:.3f} ms".format(
            n=cb["node_cost_ms"], c=cb["conversion_cost_ms"], t=cb["total_cost_ms"]
        )
    )
    baselines = plan.get("baselines", {})
    if baselines:
        print(
            "[baseline] all_HE={he:.3f} ms, all_MPC={mpc:.3f} ms, hybrid(linear=HE,nonlinear=MPC)={hyb:.3f} ms".format(
                he=baselines["all_he_total_ms"],
                mpc=baselines["all_mpc_total_ms"],
                hyb=baselines["hybrid_linear_he_nonlinear_mpc_total_ms"],
            )
        )


def run_demo() -> None:
    here = Path(__file__).resolve().parent
    case_path = here / "test" / "cases.json"
    cases: List[Dict[str, str]] = json.loads(case_path.read_text())["cases"]

    wins = 0
    for case in cases:
        _print_case_header(case)
        profiler_file = here / "test" / case["profiler_json"]
        graph_file = here / "test" / case["graph_json"]

        db = ProfilerDB.from_json(profiler_file)
        cm = CostModel(db=db, default_strategy="auto")
        graph = load_graph_json(graph_file)

        result = assign_domains_min_cut(graph, cm)
        plan = build_execution_plan(graph, result.assignment, cm, include_baselines=True)

        _print_node_costs(result)
        _print_assignment(result)
        _print_plan(plan)

        total = plan["cost_breakdown"]["total_cost_ms"]  # type: ignore[index]
        all_he = plan["baselines"]["all_he_total_ms"]  # type: ignore[index]
        all_mpc = plan["baselines"]["all_mpc_total_ms"]  # type: ignore[index]
        is_better = total <= all_he + 1e-9 and total <= all_mpc + 1e-9
        print(f"[check] optimized <= baselines: {is_better}")
        if is_better:
            wins += 1

    print("\n" + "-" * 78)
    print(f"[summary] {wins}/{len(cases)} cases optimized <= all-HE and all-MPC baselines")
    if wins != len(cases):
        raise SystemExit("Some cases failed baseline comparison")


if __name__ == "__main__":
    run_demo()

