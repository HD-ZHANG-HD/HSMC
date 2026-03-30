from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from framework import register_default_backend_impls
from runtime import (
    BackendType,
    ExecutionContext,
    OperatorRegistry,
    OperatorRouter,
    TensorValue,
    capability_registry,
)
from operators.linear_ffn1.method_mpc_bolt import deterministic_ffn_linear1_params


def run_case(name: str, b: int, s: int, h: int, out_dim: int, cfg_path: Path) -> None:
    registry = OperatorRegistry()
    register_default_backend_impls(registry)
    router = OperatorRouter.from_config_file(registry, cfg_path)

    x = np.random.standard_normal((b, s, h))
    tensors = {"input": TensorValue(x, BackendType.MPC)}
    ctx = ExecutionContext(
        params={
            "ffn_linear1_output_dim": out_dim,
            "ffn_linear1_weight_seed": 1234,
            "ffn_linear1_ell": 37,
            "ffn_linear1_scale": 12,
            "ffn_linear1_nthreads": 2,
            "ffn_linear1_port": None,
        }
    )
    out = router.execute_pipeline(tensors, ctx)

    ffn_hidden = np.asarray(out["ffn_hidden"].data, dtype=np.float64)
    assert ffn_hidden.shape == (b, s, out_dim), f"{name}: shape mismatch {ffn_hidden.shape}"
    w, bias = deterministic_ffn_linear1_params(h, out_dim, seed=1234)
    attn_norm = np.asarray(out["attn_norm"].data, dtype=np.float64)
    ref = attn_norm @ w + bias
    mae = float(np.mean(np.abs(ffn_hidden - ref)))
    assert mae < 1.0, f"{name}: MAE too high {mae}"
    trace = out["__trace__"].data
    assert any(step == "FFN_Linear_1@MPC" for step in trace), f"{name}: missing router trace FFN_Linear_1@MPC"
    print(f"[routed-test] {name}: shape_ok=True mae={mae:.6f} trace_ok=True")


def main() -> None:
    cfg_path = Path(__file__).resolve().parents[0] / "configs/ffn_linear1_mpc_only.json"
    status = capability_registry.get_status("FFN_Linear_1", BackendType.MPC).value
    print(f"[capability] FFN_Linear_1@MPC={status}")

    run_case("small_sanity", 1, 2, 4, 8, cfg_path)
    run_case("multi_batch", 4, 2, 8, 16, cfg_path)
    for seq in (1, 4, 8):
        run_case(f"variable_seq_s{seq}", 2, seq, 8, 16, cfg_path)
    run_case("non_bert_dim", 2, 3, 5, 7, cfg_path)


if __name__ == "__main__":
    main()
