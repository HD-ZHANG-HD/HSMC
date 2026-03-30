from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from operators.layernorm.method_he_nexus import NexusHeLayerNormConfig, run_nexus_layernorm_he
from runtime.types import ExecutionContext


def main() -> None:
    x = np.random.standard_normal((1, 2, 768))
    y = run_nexus_layernorm_he(
        x,
        cfg=NexusHeLayerNormConfig(hidden_size=768, max_tokens=16, packed_len=1024),
    )
    assert y.shape == x.shape, f"shape mismatch: {y.shape} != {x.shape}"
    assert np.isfinite(y).all(), "non-finite output"

    ctx = ExecutionContext(params={"layernorm_weight": np.ones((768,), dtype=np.float64)})
    try:
        run_nexus_layernorm_he(x, ctx=ctx)
    except ValueError as exc:
        print(f"[restricted-contract] affine_rejected=True reason={exc}")
    else:
        raise AssertionError("expected affine weight rejection under restricted contract")

    try:
        run_nexus_layernorm_he(np.random.standard_normal((1, 2, 4)))
    except ValueError as exc:
        print(f"[restricted-contract] hidden_rejected=True reason={exc}")
    else:
        raise AssertionError("expected hidden-size rejection under restricted contract")

    print(f"[wrapper-test] output_shape={y.shape} finite_ok=True")


if __name__ == "__main__":
    main()
