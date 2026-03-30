from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from operators.layernorm.method_mpc_bolt import BertBoltLayerNormConfig, run_bert_bolt_layernorm_mpc
from runtime.types import ExecutionContext


def main() -> None:
    x = np.random.standard_normal((1, 2, 4))
    # Broadcast affine params [H]
    weight = np.ones((4,), dtype=np.float64)
    bias = np.zeros((4,), dtype=np.float64)
    ctx = ExecutionContext(params={"layernorm_weight": weight, "layernorm_bias": bias})
    y = run_bert_bolt_layernorm_mpc(
        x,
        ctx=ctx,
        cfg=BertBoltLayerNormConfig(ell=37, scale=12, nthreads=2, port=None),
    )
    print("input_shape:", x.shape)
    print("output_shape:", y.shape)
    print("output_sample:", y.reshape(-1)[:6])


if __name__ == "__main__":
    main()
