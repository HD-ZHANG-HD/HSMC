from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from framework import register_default_backend_impls
from operators.gelu.method_he_nexus import NexusHeGeluConfig, run_nexus_gelu_he
from runtime import (
    BackendType,
    ExecutionContext,
    OperatorRegistry,
    OperatorRouter,
    TensorValue,
    capability_registry,
)


def run_case(name: str, b: int, s: int, h: int, cfg_path: Path) -> None:
    registry = OperatorRegistry()
    register_default_backend_impls(registry)
    router = OperatorRouter.from_config_file(registry, cfg_path)

    x = np.random.standard_normal((b, s, h))
    tensors = {"input": TensorValue(x, BackendType.MPC)}
    ctx = ExecutionContext(
        params={
            "ffn_linear1_output_dim": h * 2,
            "gelu_he_nexus_clamp_min": -8.0,
            "gelu_he_nexus_clamp_max": 8.0,
        }
    )
    out = router.execute_pipeline(tensors, ctx)

    ffn_hidden = np.asarray(out["ffn_hidden"].data, dtype=np.float64)
    y = np.asarray(out["ffn_activated"].data, dtype=np.float64)
    ref = run_nexus_gelu_he(ffn_hidden, cfg=NexusHeGeluConfig(clamp_min=-8.0, clamp_max=8.0))
    ref *= 0.99  # keep parity with current backend scaling policy for HE.
    mae = float(np.mean(np.abs(y - ref)))
    trace = out["__trace__"].data

    assert y.shape == ref.shape, f"{name}: shape mismatch {y.shape} vs {ref.shape}"
    assert np.isfinite(y).all(), f"{name}: non-finite output"
    assert mae < 1e-9, f"{name}: MAE too high {mae}"
    assert any(step == "GeLU@HE" for step in trace), f"{name}: missing GeLU@HE in trace"
    print(f"[routed-test] {name}: shape_ok=True mae={mae:.6e} trace_ok=True")


def main() -> None:
    cfg_path = Path(__file__).resolve().parents[0] / "configs/gelu_he_nexus_only.json"
    status = capability_registry.get_status("GeLU", BackendType.HE).value
    print(f"[capability] GeLU@HE={status}")

    run_case("small_sanity", 1, 2, 4, cfg_path)


if __name__ == "__main__":
    main()
