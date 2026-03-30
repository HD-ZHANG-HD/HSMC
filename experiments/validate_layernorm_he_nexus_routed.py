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

    def qk(inputs, ctx):
        del ctx
        qkv = np.asarray(inputs[0].data, dtype=np.float64)
        q = qkv[0]
        bsz, seq, _ = q.shape
        scores = np.zeros((bsz, 1, seq, seq), dtype=np.float64)
        return TensorValue(scores, BackendType.MPC)

    def softmax(inputs, ctx):
        del ctx
        x = np.asarray(inputs[0].data, dtype=np.float64)
        denom = np.maximum(np.sum(np.ones_like(x), axis=-1, keepdims=True), 1.0)
        return TensorValue(np.ones_like(x) / denom, BackendType.MPC)

    def v_matmul(inputs, ctx):
        del ctx
        qkv = np.asarray(inputs[1].data, dtype=np.float64)
        v = qkv[2]
        return TensorValue(v, BackendType.MPC)

    def out_proj(inputs, ctx):
        del ctx
        return TensorValue(np.asarray(inputs[0].data, dtype=np.float64), BackendType.MPC)

    def residual_add(inputs, ctx):
        del ctx
        a = np.asarray(inputs[0].data, dtype=np.float64)
        b = np.asarray(inputs[1].data, dtype=np.float64)
        return TensorValue(a + b, BackendType.MPC)

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
    registry.register("Attention_QK_MatMul", BackendType.MPC, qk)
    registry.register("Softmax", BackendType.MPC, softmax)
    registry.register("Attention_V_MatMul", BackendType.MPC, v_matmul)
    registry.register("Out_Projection", BackendType.MPC, out_proj)
    registry.register("Residual_Add", BackendType.MPC, residual_add)
    registry.register("FFN_Linear_1", BackendType.MPC, ffn1)
    registry.register("GeLU", BackendType.MPC, gelu)
    registry.register("FFN_Linear_2", BackendType.MPC, ffn2)


def main() -> None:
    cfg_path = Path(__file__).resolve().parents[0] / "configs/layernorm_he_nexus_only.json"
    registry = OperatorRegistry()
    register_default_backend_impls(registry)
    _register_fast_mpc_overrides(registry)
    router = OperatorRouter.from_config_file(registry, cfg_path)

    x = np.random.standard_normal((1, 2, 768))
    ctx = ExecutionContext(
        params={
            "layernorm_he_nexus_hidden_size": 768,
            "layernorm_he_nexus_max_tokens": 16,
            "layernorm_he_nexus_packed_len": 1024,
        }
    )
    out = router.execute_pipeline({"input": TensorValue(x, BackendType.MPC)}, ctx)

    y = np.asarray(out["attn_norm"].data, dtype=np.float64)
    assert y.shape == x.shape, f"shape mismatch {y.shape}"
    assert np.isfinite(y).all(), "non-finite output"

    trace = out["__trace__"].data
    assert any(step == "LayerNorm@HE" for step in trace), "missing LayerNorm@HE trace"
    assert any("[layernorm_he_nexus]" in step for step in trace), "missing layernorm_he_nexus trace"
    assert any("MPC_to_HE@method_default[mock]" in step for step in trace), "missing MPC_to_HE conversion trace"
    assert any("HE_to_MPC@method_default[mock]" in step for step in trace), "missing HE_to_MPC conversion trace"

    status = capability_registry.get_status("LayerNorm", BackendType.HE).value
    assert status == "restricted-integrated", f"unexpected capability status: {status}"
    print(f"[capability] LayerNorm@HE={status}")
    print("[routed-test] shape_ok=True trace_ok=True conversions_ok=True")


if __name__ == "__main__":
    main()
