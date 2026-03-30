from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from operators.softmax.method_he_nexus import NexusHeSoftmaxConfig, run_nexus_softmax_he
from runtime.types import ExecutionContext


def _softmax_ref(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    ex = np.exp(x)
    return ex / ex.sum(axis=-1, keepdims=True)


def run_case(name: str, shape: tuple[int, ...]) -> None:
    x = np.random.uniform(-8.0, 2.0, size=shape)
    y = run_nexus_softmax_he(
        x,
        ctx=ExecutionContext(),
        cfg=NexusHeSoftmaxConfig(inverse_iterations=4, sum_scale_factor=0.01, eps=1e-8),
    )
    ref = _softmax_ref(x)
    mae = float(np.mean(np.abs(y - ref)))
    row_sums = y.reshape(-1, y.shape[-1]).sum(axis=-1)
    assert y.shape == x.shape, f"{name}: shape mismatch {y.shape} vs {x.shape}"
    assert np.isfinite(y).all(), f"{name}: non-finite output"
    assert np.all(row_sums > 0.0) and np.all(row_sums < 2.5), f"{name}: invalid row sums"
    assert mae < 0.25, f"{name}: MAE too high {mae}"
    print(f"[wrapper-test] {name}: shape_ok=True mae={mae:.6f} row_sum_ok=True")


def main() -> None:
    run_case("small_sanity", (1, 2, 4, 4))
    run_case("multi_batch", (4, 2, 8, 8))
    run_case("nontrivial_dim", (2, 3, 5, 7))


if __name__ == "__main__":
    main()
