from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from operators.residual_add.method_runtime_default import run_residual_add_semantic
from runtime import BackendType, ExecutionContext


def main() -> None:
    a = np.random.standard_normal((1, 2, 4))
    b = np.random.standard_normal((1, 2, 4))
    ctx_he = ExecutionContext()
    ctx_mpc = ExecutionContext()

    y_he = run_residual_add_semantic([a, b], backend=BackendType.HE, ctx=ctx_he)
    y_mpc = run_residual_add_semantic([a, b], backend=BackendType.MPC, ctx=ctx_mpc)
    assert y_he.shape == a.shape
    assert y_mpc.shape == a.shape
    assert np.allclose(y_he, a + b)
    assert np.allclose(y_mpc, a + b)
    assert any("[residual_add_semantic]" in step for step in ctx_he.trace)
    assert any("[residual_add_semantic]" in step for step in ctx_mpc.trace)

    try:
        run_residual_add_semantic([a, np.random.standard_normal((1, 2, 5))], backend=BackendType.MPC)
    except ValueError as exc:
        print(f"[restricted-contract] shape_rejected=True reason={exc}")
    else:
        raise AssertionError("expected shape rejection for Residual_Add")

    print("[semantic-test] Residual_Add@HE shape_ok=True trace_ok=True")
    print("[semantic-test] Residual_Add@MPC shape_ok=True trace_ok=True")


if __name__ == "__main__":
    main()
