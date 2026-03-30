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
        qkv = np.stack([x, x, x], axis=0)
        return TensorValue(qkv, BackendType.MPC)

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

    def layernorm(inputs, ctx):
        del ctx
        return TensorValue(np.asarray(inputs[0].data, dtype=np.float64), BackendType.MPC)

    def gelu(inputs, ctx):
        del ctx
        x = np.asarray(inputs[0].data, dtype=np.float64)
        return TensorValue(x, BackendType.MPC)

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
    registry.register("LayerNorm", BackendType.MPC, layernorm)
    registry.register("GeLU", BackendType.MPC, gelu)
    registry.register("FFN_Linear_2", BackendType.MPC, ffn2)


def run_case(name: str, b: int, s: int, cfg_path: Path) -> None:
    registry = OperatorRegistry()
    register_default_backend_impls(registry)
    _register_fast_mpc_overrides(registry)
    router = OperatorRouter.from_config_file(registry, cfg_path)

    x = np.random.standard_normal((b, s, 768))
    ctx = ExecutionContext(
        params={
            "ffn_linear1_he_nexus_hidden_size": 768,
            "ffn_linear1_he_nexus_out_dim": 64,
            "ffn_linear1_he_nexus_max_tokens": 4096,
            "ffn_linear1_he_nexus_poly_degree": 4096,
            "ffn_linear1_he_nexus_weight_seed": 2024,
        }
    )
    out = router.execute_pipeline({"input": TensorValue(x, BackendType.MPC)}, ctx)

    y = np.asarray(out["ffn_hidden"].data, dtype=np.float64)
    assert y.shape == (b, s, 64), f"{name}: shape mismatch {y.shape}"
    assert np.isfinite(y).all(), f"{name}: non-finite output"

    trace = out["__trace__"].data
    assert any(step == "FFN_Linear_1@HE" for step in trace), f"{name}: missing FFN_Linear_1@HE in trace"
    print(f"[routed-test] {name}: shape_ok=True trace_ok=True")


def main() -> None:
    cfg_path = Path(__file__).resolve().parents[0] / "configs/ffn_linear1_he_nexus_only.json"
    status = capability_registry.get_status("FFN_Linear_1", BackendType.HE).value
    assert status == "restricted-integrated", f"unexpected capability status: {status}"
    print(f"[capability] FFN_Linear_1@HE={status}")

    run_case("small_sanity", 1, 2, cfg_path)
    run_case("multi_batch", 4, 8, cfg_path)
    run_case("nontrivial_tokens", 8, 16, cfg_path)


if __name__ == "__main__":
    main()
