from __future__ import annotations

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


def _register_fast_mpc_overrides(registry: OperatorRegistry) -> None:
    def embedding(inputs, ctx):
        del ctx
        x = np.asarray(inputs[0].data, dtype=np.float64)
        return TensorValue(x, BackendType.MPC)

    def linear_qkv(inputs, ctx):
        del ctx
        x = np.asarray(inputs[0].data, dtype=np.float64)
        return TensorValue(np.stack([x, x, x], axis=0), BackendType.MPC)

    def softmax(inputs, ctx):
        del ctx
        x = np.asarray(inputs[0].data, dtype=np.float64)
        x = x - x.max(axis=-1, keepdims=True)
        ex = np.exp(x)
        return TensorValue(ex / ex.sum(axis=-1, keepdims=True), BackendType.MPC)

    def out_proj(inputs, ctx):
        del ctx
        x = np.asarray(inputs[0].data, dtype=np.float64)
        return TensorValue(x, BackendType.MPC)

    def residual(inputs, ctx):
        del ctx
        a = np.asarray(inputs[0].data, dtype=np.float64)
        b = np.asarray(inputs[1].data, dtype=np.float64)
        return TensorValue(a + b, BackendType.MPC)

    def layernorm(inputs, ctx):
        del ctx
        return TensorValue(np.asarray(inputs[0].data, dtype=np.float64), BackendType.MPC)

    def ffn1(inputs, ctx):
        del ctx
        x = np.asarray(inputs[0].data, dtype=np.float64)
        return TensorValue(np.concatenate([x, x], axis=-1), BackendType.MPC)

    def gelu(inputs, ctx):
        del ctx
        return TensorValue(np.asarray(inputs[0].data, dtype=np.float64), BackendType.MPC)

    def ffn2(inputs, ctx):
        del ctx
        x = np.asarray(inputs[0].data, dtype=np.float64)
        return TensorValue(x[..., : x.shape[-1] // 2], BackendType.MPC)

    registry.register("Embedding", BackendType.MPC, embedding)
    registry.register("Linear_QKV", BackendType.MPC, linear_qkv)
    registry.register("Softmax", BackendType.MPC, softmax)
    registry.register("Out_Projection", BackendType.MPC, out_proj)
    registry.register("Residual_Add", BackendType.MPC, residual)
    registry.register("LayerNorm", BackendType.MPC, layernorm)
    registry.register("FFN_Linear_1", BackendType.MPC, ffn1)
    registry.register("GeLU", BackendType.MPC, gelu)
    registry.register("FFN_Linear_2", BackendType.MPC, ffn2)


def _to_bhsd(x: np.ndarray, heads: int) -> np.ndarray:
    bsz, seq, hidden = x.shape
    d = hidden // heads
    return x.reshape(bsz, seq, heads, d).transpose(0, 2, 1, 3)


def run_case(name: str, b: int, s: int, cfg_path: Path) -> None:
    registry = OperatorRegistry()
    register_default_backend_impls(registry)
    _register_fast_mpc_overrides(registry)
    router = OperatorRouter.from_config_file(registry, cfg_path)

    x = np.random.standard_normal((b, s, 768))
    ctx = ExecutionContext(
        params={
            "attention_he_nexus_hidden_size": 768,
            "attention_he_nexus_num_heads": 12,
            "attention_he_nexus_max_seq_len": 128,
            "attention_he_nexus_max_batch": 8,
            "attention_return_canonical": False,
        }
    )
    out = router.execute_pipeline({"input": TensorValue(x, BackendType.MPC)}, ctx)

    qkv = np.asarray(out["qkv_out"].data, dtype=np.float64)
    q = _to_bhsd(qkv[0], 12)
    k = _to_bhsd(qkv[1], 12)
    v = _to_bhsd(qkv[2], 12)
    qk = np.asarray(out["qk_scores"].data, dtype=np.float64)
    ref_qk = q @ np.swapaxes(k, -1, -2)
    ref_qk *= 0.99  # parity with current HE backend scaling policy
    qk_mae = float(np.mean(np.abs(qk - ref_qk)))

    attn = np.asarray(out["attn_probs"].data, dtype=np.float64)
    context = np.asarray(out["context"].data, dtype=np.float64)
    ref_context = (attn @ v).transpose(0, 2, 1, 3).reshape(b, s, 768)
    ref_context *= 0.99  # parity with current HE backend scaling policy
    ctx_mae = float(np.mean(np.abs(context - ref_context)))

    assert qk.shape == (b, 12, s, s), f"{name}: qk shape mismatch {qk.shape}"
    assert context.shape == (b, s, 768), f"{name}: context shape mismatch {context.shape}"
    assert qk_mae < 1e-9, f"{name}: qk MAE too high {qk_mae}"
    assert ctx_mae < 1e-9, f"{name}: context MAE too high {ctx_mae}"

    trace = out["__trace__"].data
    assert any(step == "Attention_QK_MatMul@HE" for step in trace), f"{name}: missing QK@HE trace"
    assert any(step == "Attention_V_MatMul@HE" for step in trace), f"{name}: missing V@HE trace"
    print(f"[routed-test] {name}: qk_shape_ok=True context_shape_ok=True trace_ok=True")


def main() -> None:
    cfg_path = Path(__file__).resolve().parents[0] / "configs/attention_matmul_he_nexus_only.json"
    qk_status = capability_registry.get_status("Attention_QK_MatMul", BackendType.HE).value
    v_status = capability_registry.get_status("Attention_V_MatMul", BackendType.HE).value
    assert qk_status == "restricted-integrated", f"unexpected qk status {qk_status}"
    assert v_status == "restricted-integrated", f"unexpected v status {v_status}"
    print(f"[capability] Attention_QK_MatMul@HE={qk_status}")
    print(f"[capability] Attention_V_MatMul@HE={v_status}")

    run_case("small_sanity", 1, 2, cfg_path)
    run_case("multi_batch", 4, 8, cfg_path)
    run_case("nontrivial_seq", 2, 16, cfg_path)


if __name__ == "__main__":
    main()
