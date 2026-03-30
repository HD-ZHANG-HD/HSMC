from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from operators.gelu.method_mpc_bolt import BertBoltGeluConfig, run_bert_bolt_gelu_mpc
from runtime.types import ExecutionContext


def main() -> None:
    x = np.random.standard_normal((1, 2, 4))
    ctx = ExecutionContext()
    y = run_bert_bolt_gelu_mpc(
        x,
        ctx=ctx,
        cfg=BertBoltGeluConfig(ell=37, scale=12, nthreads=2, port=None),
    )
    print("input_shape:", x.shape)
    print("output_shape:", y.shape)
    print("output_sample:", y.reshape(-1)[:4])


if __name__ == "__main__":
    main()
