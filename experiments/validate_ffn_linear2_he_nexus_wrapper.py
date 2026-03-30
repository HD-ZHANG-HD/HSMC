from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from operators.ffn_linear_2.method_he_nexus import NexusHeLinearFfn2Config, run_nexus_linear_ffn2_he
from runtime.types import ExecutionContext


def main() -> None:
    x = np.random.standard_normal((1, 2, 1536))
    ctx = ExecutionContext(
        params={
            "ffn_linear2_he_nexus_hidden_size": 1536,
            "ffn_linear2_he_nexus_out_dim": 768,
            "ffn_linear2_he_nexus_max_tokens": 4096,
            "ffn_linear2_he_nexus_poly_degree": 4096,
            "ffn_linear2_he_nexus_weight_seed": 2234,
        }
    )
    y = run_nexus_linear_ffn2_he(
        x,
        ctx=ctx,
        cfg=NexusHeLinearFfn2Config(hidden_size=1536, out_dim=768),
    )
    assert y.shape == (1, 2, 768), f"shape mismatch {y.shape}"
    assert np.isfinite(y).all(), "non-finite output"
    assert any("[ffn_linear2_he_nexus]" in step for step in ctx.trace), "missing semantic trace"

    try:
        run_nexus_linear_ffn2_he(np.random.standard_normal((1, 2, 768)), ctx=ctx)
    except ValueError as exc:
        print(f"[restricted-contract] hidden_rejected=True reason={exc}")
    else:
        raise AssertionError("expected hidden-size rejection for FFN_Linear_2@HE")

    print("[wrapper-test] FFN_Linear_2@HE shape_ok=True trace_ok=True")


if __name__ == "__main__":
    main()
