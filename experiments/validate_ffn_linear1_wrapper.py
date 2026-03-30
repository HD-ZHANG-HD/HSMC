from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from operators.linear_ffn1.method_mpc_bolt import (
    BertBoltFfnLinear1Config,
    deterministic_ffn_linear1_params,
    run_bert_bolt_ffn_linear1_mpc,
)
from runtime.types import ExecutionContext


def check_case(name: str, b: int, s: int, h: int, out_dim: int) -> None:
    x = np.random.standard_normal((b, s, h))
    cfg = BertBoltFfnLinear1Config(weight_seed=1234, port=None, nthreads=2, ell=37, scale=12)
    y, w, bias = run_bert_bolt_ffn_linear1_mpc(x, out_dim=out_dim, ctx=ExecutionContext(), cfg=cfg)
    ref_w, ref_b = deterministic_ffn_linear1_params(h, out_dim, seed=1234)
    ref = x @ ref_w + ref_b
    mae = float(np.mean(np.abs(y - ref)))
    assert y.shape == (b, s, out_dim), f"{name}: shape mismatch {y.shape}"
    assert np.isfinite(y).all(), f"{name}: non-finite output"
    assert mae < 1.0, f"{name}: MAE too high {mae}"
    print(f"[wrapper-test] {name}: shape_ok=True mae={mae:.6f}")


def main() -> None:
    check_case("small_sanity", 1, 2, 4, 8)
    check_case("multi_batch", 4, 2, 8, 16)
    for seq in (1, 4, 8):
        check_case(f"variable_seq_s{seq}", 2, seq, 8, 16)
    check_case("non_bert_dim", 2, 3, 5, 7)


if __name__ == "__main__":
    main()
