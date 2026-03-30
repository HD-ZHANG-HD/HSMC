from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from framework import register_default_backend_impls  # noqa: E402
from runtime import (  # noqa: E402
    BackendType,
    ExecutionContext,
    OperatorRegistry,
    OperatorRouter,
    TensorValue,
    capability_registry,
)


def to_bhsd(x_bsh: np.ndarray, heads: int) -> np.ndarray:
    bsz, seq, hidden = x_bsh.shape
    if hidden % heads != 0:
        raise ValueError(f"hidden size {hidden} is not divisible by heads={heads}")
    head_dim = hidden // heads
    return x_bsh.reshape(bsz, seq, heads, head_dim).transpose(0, 2, 1, 3)


def run_case(name: str, bsz: int, heads: int, seq: int, head_dim: int, cfg_path: Path) -> None:
    hidden = heads * head_dim
    registry = OperatorRegistry()
    register_default_backend_impls(registry)
    router = OperatorRouter.from_config_file(registry, cfg_path)

    x = np.random.standard_normal((bsz, seq, hidden))
    ctx = ExecutionContext(
        params={
            "attention_num_heads": heads,
            "attention_ell": 37,
            "attention_scale": 12,
            "attention_nthreads": 2,
            "attention_port": None,
            "attention_return_canonical": False,
        }
    )
    outputs = router.execute_pipeline({"input": TensorValue(x, BackendType.MPC)}, ctx)

    qkv_out = np.asarray(outputs["qkv_out"].data, dtype=np.float64)
    q = to_bhsd(qkv_out[0], heads)
    k = to_bhsd(qkv_out[1], heads)
    v = to_bhsd(qkv_out[2], heads)

    qk_scores = np.asarray(outputs["qk_scores"].data, dtype=np.float64)
    ref_qk = q @ np.swapaxes(k, -1, -2)
    qk_mae = float(np.mean(np.abs(qk_scores - ref_qk)))

    attn_probs = np.asarray(outputs["attn_probs"].data, dtype=np.float64)
    if attn_probs.ndim == 3:
        attn_probs = attn_probs[:, np.newaxis, :, :]
    context = np.asarray(outputs["context"].data, dtype=np.float64)
    ref_context_canonical = attn_probs @ v
    ref_context_flat = ref_context_canonical.transpose(0, 2, 1, 3).reshape(bsz, seq, hidden)
    context_mae = float(np.mean(np.abs(context - ref_context_flat)))

    assert qk_scores.shape == (bsz, heads, seq, seq), f"{name}: qk shape mismatch {qk_scores.shape}"
    assert context.shape == (bsz, seq, hidden), f"{name}: context shape mismatch {context.shape}"
    assert np.isfinite(qk_scores).all(), f"{name}: qk non-finite output"
    assert np.isfinite(context).all(), f"{name}: context non-finite output"
    assert qk_mae < 1.0, f"{name}: qk MAE too high {qk_mae}"
    assert context_mae < 1.0, f"{name}: context MAE too high {context_mae}"

    trace = outputs["__trace__"].data
    assert any(step == "Attention_QK_MatMul@MPC" for step in trace), f"{name}: missing Attention_QK_MatMul@MPC in trace"
    assert any(step == "Attention_V_MatMul@MPC" for step in trace), f"{name}: missing Attention_V_MatMul@MPC in trace"
    print(
        f"[routed-test] {name}: qk_shape_ok=True context_shape_ok=True "
        f"qk_mae={qk_mae:.6f} context_mae={context_mae:.6f} trace_ok=True"
    )


def main() -> None:
    cfg_path = Path(__file__).resolve().parents[0] / "configs/attention_matmul_mpc_only.json"
    print(
        "[capability] Attention_QK_MatMul@MPC="
        f"{capability_registry.get_status('Attention_QK_MatMul', BackendType.MPC).value}"
    )
    print(
        "[capability] Attention_V_MatMul@MPC="
        f"{capability_registry.get_status('Attention_V_MatMul', BackendType.MPC).value}"
    )

    run_case("small_sanity", 1, 1, 2, 4, cfg_path)
    run_case("multi_head", 1, 4, 8, 16, cfg_path)
    run_case("multi_batch", 4, 2, 8, 16, cfg_path)


if __name__ == "__main__":
    main()

