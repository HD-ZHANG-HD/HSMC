from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from runtime import (  # noqa: E402
    BackendType,
    ExecutionContext,
    OperatorRegistry,
    OperatorRouter,
    TensorValue,
    conversion_capability_registry,
    conversion_manager,
)


def _log(ctx: ExecutionContext, op_name: str, backend: BackendType) -> None:
    ctx.trace.append(f"{op_name}@{backend.value}")


def _make_embedding(backend: BackendType):
    def fn(inputs, ctx):
        _log(ctx, "Embedding", backend)
        x = np.asarray(inputs[0].data, dtype=np.float64)
        return TensorValue(x, backend)

    return fn


def _make_linear_qkv(backend: BackendType):
    def fn(inputs, ctx):
        _log(ctx, "Linear_QKV", backend)
        x = np.asarray(inputs[0].data, dtype=np.float64)
        return TensorValue(np.stack([x, x, x], axis=0), backend)

    return fn


def _make_qk(backend: BackendType):
    def fn(inputs, ctx):
        _log(ctx, "Attention_QK_MatMul", backend)
        qkv = np.asarray(inputs[0].data, dtype=np.float64)
        q = qkv[0]
        k = qkv[1]
        scores = np.matmul(q, np.swapaxes(k, -1, -2))[:, np.newaxis, :, :]
        return TensorValue(scores, backend)

    return fn


def _make_softmax(backend: BackendType):
    def fn(inputs, ctx):
        _log(ctx, "Softmax", backend)
        x = np.asarray(inputs[0].data, dtype=np.float64)
        x = x - np.max(x, axis=-1, keepdims=True)
        ex = np.exp(x)
        return TensorValue(ex / np.sum(ex, axis=-1, keepdims=True), backend)

    return fn


def _make_v_matmul(backend: BackendType):
    def fn(inputs, ctx):
        _log(ctx, "Attention_V_MatMul", backend)
        attn = np.asarray(inputs[0].data, dtype=np.float64)
        qkv = np.asarray(inputs[1].data, dtype=np.float64)
        v = qkv[2][:, np.newaxis, :, :]
        context = np.matmul(attn, v).reshape(qkv.shape[1], qkv.shape[2], qkv.shape[3])
        return TensorValue(context, backend)

    return fn


def _make_identity(op_name: str, backend: BackendType):
    def fn(inputs, ctx):
        _log(ctx, op_name, backend)
        x = np.asarray(inputs[0].data, dtype=np.float64)
        return TensorValue(x, backend)

    return fn


def _make_residual(backend: BackendType):
    def fn(inputs, ctx):
        _log(ctx, "Residual_Add", backend)
        a = np.asarray(inputs[0].data, dtype=np.float64)
        b = np.asarray(inputs[1].data, dtype=np.float64)
        return TensorValue(a + b, backend)

    return fn


def build_test_registry() -> OperatorRegistry:
    registry = OperatorRegistry()
    for backend in (BackendType.HE, BackendType.MPC):
        registry.register("Embedding", backend, _make_embedding(backend))
        registry.register("Linear_QKV", backend, _make_linear_qkv(backend))
        registry.register("Attention_QK_MatMul", backend, _make_qk(backend))
        registry.register("Softmax", backend, _make_softmax(backend))
        registry.register("Attention_V_MatMul", backend, _make_v_matmul(backend))
        registry.register("Out_Projection", backend, _make_identity("Out_Projection", backend))
        registry.register("Residual_Add", backend, _make_residual(backend))
        registry.register("LayerNorm", backend, _make_identity("LayerNorm", backend))
        registry.register("FFN_Linear_1", backend, _make_identity("FFN_Linear_1", backend))
        registry.register("GeLU", backend, _make_identity("GeLU", backend))
        registry.register("FFN_Linear_2", backend, _make_identity("FFN_Linear_2", backend))
    return registry


def validate_explicit_conversion_invocation() -> None:
    he_tensor = TensorValue(np.ones((1, 2, 3)), BackendType.HE, {"token": "x"})
    mpc_tensor = TensorValue(np.ones((1, 2, 3)), BackendType.MPC, {"token": "y"})
    ctx = ExecutionContext()

    he_to_mpc = conversion_manager.convert(he_tensor, BackendType.MPC, ctx)
    mpc_to_he = conversion_manager.convert(mpc_tensor, BackendType.HE, ctx)

    assert he_to_mpc.domain == BackendType.MPC
    assert mpc_to_he.domain == BackendType.HE
    assert he_to_mpc.meta == he_tensor.meta
    assert mpc_to_he.meta == mpc_tensor.meta
    assert he_to_mpc.data.shape == he_tensor.data.shape
    assert mpc_to_he.data.shape == mpc_tensor.data.shape
    assert any("HE_to_MPC@method_default[mock]" in step for step in ctx.trace)
    assert any("MPC_to_HE@method_default[mock]" in step for step in ctx.trace)

    he_to_mpc_status = conversion_capability_registry.get_status(
        BackendType.HE, BackendType.MPC, "method_default"
    ).value
    mpc_to_he_status = conversion_capability_registry.get_status(
        BackendType.MPC, BackendType.HE, "method_default"
    ).value
    assert he_to_mpc_status == "mock"
    assert mpc_to_he_status == "mock"

    print("[conversion-explicit] HE_to_MPC and MPC_to_HE invocation ok")
    print(f"[capability] HE_to_MPC@method_default={he_to_mpc_status}")
    print(f"[capability] MPC_to_HE@method_default={mpc_to_he_status}")


def _run_route_case(name: str, op_backend_map: dict[str, BackendType], expected_conversion: str) -> None:
    registry = build_test_registry()
    router = OperatorRouter(registry, op_backend_map)
    x = np.random.standard_normal((1, 2, 4))
    ctx = ExecutionContext()
    outputs = router.execute_pipeline({"input": TensorValue(x, op_backend_map["Embedding"])}, ctx)

    final_tensor = outputs["ffn_out"]
    assert np.asarray(final_tensor.data).shape == (1, 2, 4), f"{name}: output shape mismatch"
    trace = outputs["__trace__"].data
    assert any(expected_conversion in step for step in trace), f"{name}: missing conversion trace {expected_conversion}"
    print(f"[conversion-route] {name}: shape_ok=True trace_ok=True conversion={expected_conversion}")


def validate_routed_conversion_events() -> None:
    he_to_mpc_map = {
        "Embedding": BackendType.HE,
        "Linear_QKV": BackendType.HE,
        "Attention_QK_MatMul": BackendType.MPC,
        "Softmax": BackendType.MPC,
        "Attention_V_MatMul": BackendType.MPC,
        "Out_Projection": BackendType.MPC,
        "Residual_Add": BackendType.MPC,
        "LayerNorm": BackendType.MPC,
        "FFN_Linear_1": BackendType.MPC,
        "GeLU": BackendType.MPC,
        "FFN_Linear_2": BackendType.MPC,
    }
    mpc_to_he_map = {
        "Embedding": BackendType.MPC,
        "Linear_QKV": BackendType.MPC,
        "Attention_QK_MatMul": BackendType.MPC,
        "Softmax": BackendType.MPC,
        "Attention_V_MatMul": BackendType.MPC,
        "Out_Projection": BackendType.HE,
        "Residual_Add": BackendType.HE,
        "LayerNorm": BackendType.HE,
        "FFN_Linear_1": BackendType.HE,
        "GeLU": BackendType.HE,
        "FFN_Linear_2": BackendType.HE,
    }
    mixed_both_map = {
        "Embedding": BackendType.HE,
        "Linear_QKV": BackendType.HE,
        "Attention_QK_MatMul": BackendType.MPC,
        "Softmax": BackendType.MPC,
        "Attention_V_MatMul": BackendType.MPC,
        "Out_Projection": BackendType.HE,
        "Residual_Add": BackendType.MPC,
        "LayerNorm": BackendType.MPC,
        "FFN_Linear_1": BackendType.HE,
        "GeLU": BackendType.MPC,
        "FFN_Linear_2": BackendType.HE,
    }

    _run_route_case("he_to_mpc_edge", he_to_mpc_map, "HE_to_MPC@method_default[mock]")
    _run_route_case("mpc_to_he_edge", mpc_to_he_map, "MPC_to_HE@method_default[mock]")

    registry = build_test_registry()
    router = OperatorRouter(registry, mixed_both_map)
    x = np.random.standard_normal((1, 2, 4))
    ctx = ExecutionContext()
    outputs = router.execute_pipeline({"input": TensorValue(x, BackendType.HE)}, ctx)
    trace = outputs["__trace__"].data
    assert any("HE_to_MPC@method_default[mock]" in step for step in trace), "mixed_both: missing HE_to_MPC"
    assert any("MPC_to_HE@method_default[mock]" in step for step in trace), "mixed_both: missing MPC_to_HE"
    print("[conversion-route] mixed_both: he_to_mpc_ok=True mpc_to_he_ok=True")


def main() -> None:
    validate_explicit_conversion_invocation()
    validate_routed_conversion_events()


if __name__ == "__main__":
    main()
