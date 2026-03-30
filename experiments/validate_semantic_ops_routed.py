from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from framework import register_default_backend_impls
from runtime import BackendType, ExecutionContext, OperatorRegistry, OperatorRouter, TensorValue, capability_registry


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
        return TensorValue(qkv[2], BackendType.MPC)

    def out_proj(inputs, ctx):
        del ctx
        return TensorValue(np.asarray(inputs[0].data, dtype=np.float64), BackendType.MPC)

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
    registry.register("Attention_QK_MatMul", BackendType.MPC, qk)
    registry.register("Softmax", BackendType.MPC, softmax)
    registry.register("Attention_V_MatMul", BackendType.MPC, v_matmul)
    registry.register("Out_Projection", BackendType.MPC, out_proj)
    registry.register("LayerNorm", BackendType.MPC, layernorm)
    registry.register("FFN_Linear_1", BackendType.MPC, ffn1)
    registry.register("GeLU", BackendType.MPC, gelu)
    registry.register("FFN_Linear_2", BackendType.MPC, ffn2)


def validate_residual_routed() -> None:
    registry = OperatorRegistry()
    register_default_backend_impls(registry)
    _register_fast_mpc_overrides(registry)
    router = OperatorRouter(
        registry,
        {
            "Embedding": BackendType.MPC,
            "Linear_QKV": BackendType.MPC,
            "Attention_QK_MatMul": BackendType.MPC,
            "Softmax": BackendType.MPC,
            "Attention_V_MatMul": BackendType.MPC,
            "Out_Projection": BackendType.MPC,
            "Residual_Add": BackendType.HE,
            "LayerNorm": BackendType.MPC,
            "FFN_Linear_1": BackendType.MPC,
            "GeLU": BackendType.MPC,
            "FFN_Linear_2": BackendType.MPC,
        },
    )
    x = np.random.standard_normal((1, 2, 768))
    out = router.execute_pipeline({"input": TensorValue(x, BackendType.MPC)}, ExecutionContext())
    y = np.asarray(out["attn_residual"].data, dtype=np.float64)
    assert y.shape == x.shape
    trace = out["__trace__"].data
    assert any(step == "Residual_Add@HE" for step in trace)
    assert any("[residual_add_semantic]" in step for step in trace)
    print("[routed-test] Residual_Add semantic trace_ok=True shape_ok=True")


def validate_ffn2_routed() -> None:
    cfg_path = Path(__file__).resolve().parents[0] / "configs/ffn_linear2_he_nexus_only.json"
    registry = OperatorRegistry()
    register_default_backend_impls(registry)
    _register_fast_mpc_overrides(registry)
    router = OperatorRouter.from_config_file(registry, cfg_path)

    x = np.random.standard_normal((1, 2, 768))
    ctx = ExecutionContext(
        params={
            "ffn_linear2_he_nexus_hidden_size": 1536,
            "ffn_linear2_he_nexus_out_dim": 768,
            "ffn_linear2_he_nexus_max_tokens": 4096,
            "ffn_linear2_he_nexus_poly_degree": 4096,
            "ffn_linear2_he_nexus_weight_seed": 2234,
        }
    )
    out = router.execute_pipeline({"input": TensorValue(x, BackendType.MPC)}, ctx)
    y = np.asarray(out["ffn_out"].data, dtype=np.float64)
    assert y.shape == x.shape
    trace = out["__trace__"].data
    assert any(step == "FFN_Linear_2@HE" for step in trace)
    assert any("[ffn_linear2_he_nexus]" in step for step in trace)
    assert any("MPC_to_HE@method_default[mock]" in step for step in trace)

    status_he = capability_registry.get_status("FFN_Linear_2", BackendType.HE).value
    status_mpc = capability_registry.get_status("FFN_Linear_2", BackendType.MPC).value
    residual_he = capability_registry.get_status("Residual_Add", BackendType.HE).value
    residual_mpc = capability_registry.get_status("Residual_Add", BackendType.MPC).value
    print(f"[capability] FFN_Linear_2@HE={status_he}")
    print(f"[capability] FFN_Linear_2@MPC={status_mpc}")
    print(f"[capability] Residual_Add@HE={residual_he}")
    print(f"[capability] Residual_Add@MPC={residual_mpc}")
    print("[routed-test] FFN_Linear_2 semantic trace_ok=True shape_ok=True")


def main() -> None:
    validate_residual_routed()
    validate_ffn2_routed()


if __name__ == "__main__":
    main()
