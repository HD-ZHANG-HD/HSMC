from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from operators.ffn_linear_2.method_mpc_bolt import BertBoltFfnLinear2Config, run_bert_bolt_ffn_linear2_mpc
from runtime.types import ExecutionContext


def main() -> None:
    x = np.random.standard_normal((1, 2, 8))
    ctx = ExecutionContext()
    y, _, _ = run_bert_bolt_ffn_linear2_mpc(
        x,
        out_dim=4,
        ctx=ctx,
        cfg=BertBoltFfnLinear2Config(ell=37, scale=12, nthreads=2, port=None, weight_seed=2234),
    )
    assert y.shape == (1, 2, 4), f"shape mismatch {y.shape}"
    assert np.isfinite(y).all(), "non-finite output"
    assert any("[ffn_linear2_wrapper]" in step for step in ctx.trace), "missing semantic trace"
    print("[wrapper-test] FFN_Linear_2@MPC shape_ok=True trace_ok=True")


if __name__ == "__main__":
    main()
