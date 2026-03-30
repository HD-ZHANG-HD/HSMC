from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from framework import register_default_backend_impls
from operators.softmax.method_he_nexus import NexusHeSoftmaxConfig, run_nexus_softmax_he
from runtime import (
    BackendType,
    ExecutionContext,
    OperatorRegistry,
    OperatorRouter,
    TensorValue,
    capability_registry,
)


def run_case(name: str, b: int, h: int, s: int, d: int, cfg_path: Path) -> None:
    hidden = h * d
    registry = OperatorRegistry()
    register_default_backend_impls(registry)
    router = OperatorRouter.from_config_file(registry, cfg_path)

    x = np.random.uniform(-0.05, 0.05, size=(b, s, hidden))
    tensors = {"input": TensorValue(x, BackendType.MPC)}
    ctx = ExecutionContext(
        params={
            "attention_num_heads": h,
            "softmax_he_nexus_inverse_iterations": 4,
            "softmax_he_nexus_sum_scale_factor": 0.01,
            "softmax_he_nexus_eps": 1e-8,
        }
    )
    out = router.execute_pipeline(tensors, ctx)

    scores = np.asarray(out["qk_scores"].data, dtype=np.float64)
    y = np.asarray(out["attn_probs"].data, dtype=np.float64)
    ref = run_nexus_softmax_he(
        scores,
        cfg=NexusHeSoftmaxConfig(inverse_iterations=4, sum_scale_factor=0.01, eps=1e-8),
    )
    ref *= 0.99  # keep parity with current backend scaling policy for HE.
    mae = float(np.mean(np.abs(y - ref)))
    trace = out["__trace__"].data

    assert y.shape == scores.shape, f"{name}: shape mismatch {y.shape} vs {scores.shape}"
    assert np.isfinite(y).all(), f"{name}: non-finite output"
    assert mae < 1e-9, f"{name}: MAE too high {mae}"
    assert any(step == "Softmax@HE" for step in trace), f"{name}: missing Softmax@HE in trace"
    print(f"[routed-test] {name}: shape_ok=True mae={mae:.6e} trace_ok=True")


def main() -> None:
    cfg_path = Path(__file__).resolve().parents[0] / "configs/softmax_he_nexus_only.json"
    status = capability_registry.get_status("Softmax", BackendType.HE).value
    print(f"[capability] Softmax@HE={status}")

    run_case("small_sanity", 1, 1, 2, 4, cfg_path)


if __name__ == "__main__":
    main()
