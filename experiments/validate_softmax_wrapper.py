from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from operators.softmax.method_mpc_bolt import BertBoltSoftmaxConfig, run_bert_bolt_softmax_mpc
from runtime.types import ExecutionContext


def main() -> None:
    # Attention-like shape: [batch, heads, seq, seq], softmax on last axis
    x = np.random.standard_normal((1, 2, 4, 4))
    ctx = ExecutionContext()
    y = run_bert_bolt_softmax_mpc(
        x,
        ctx=ctx,
        cfg=BertBoltSoftmaxConfig(ell=37, scale=12, nthreads=2, port=None),
    )
    print("input_shape:", x.shape)
    print("output_shape:", y.shape)
    print("row_sums_sample:", y.reshape(-1, y.shape[-1])[:2].sum(axis=-1))


if __name__ == "__main__":
    main()
