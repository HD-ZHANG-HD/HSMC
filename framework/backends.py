from __future__ import annotations

from typing import Any, List

import numpy as np

from operators.attention_qk_matmul.method_mpc import (
    BertBoltAttentionQkMatMulConfig,
    run_bert_bolt_attention_qk_matmul_mpc,
)
from operators.attention_qk_matmul.method_he_nexus import (
    NexusHeAttentionQkMatMulConfig,
    run_nexus_attention_qk_matmul_he,
)
from operators.attention_v_matmul.method_mpc import (
    BertBoltAttentionVMatMulConfig,
    run_bert_bolt_attention_v_matmul_mpc,
)
from operators.attention_v_matmul.method_he_nexus import (
    NexusHeAttentionVMatMulConfig,
    run_nexus_attention_v_matmul_he,
)
from operators.ffn_linear_2.method_he_nexus import NexusHeLinearFfn2Config, run_nexus_linear_ffn2_he
from operators.ffn_linear_2.method_mpc_bolt import BertBoltFfnLinear2Config, run_bert_bolt_ffn_linear2_mpc
from operators.linear_ffn1.method_he_nexus import NexusHeLinearFfn1Config, run_nexus_linear_ffn1_he
from operators.linear_ffn1.method_mpc_bolt import BertBoltFfnLinear1Config, run_bert_bolt_ffn_linear1_mpc
from operators.gelu.method_mpc_bolt import BertBoltGeluConfig, run_bert_bolt_gelu_mpc
from operators.gelu.method_he_nexus import NexusHeGeluConfig, run_nexus_gelu_he
from operators.layernorm.method_he_nexus import NexusHeLayerNormConfig, run_nexus_layernorm_he
from operators.layernorm.method_mpc_bolt import BertBoltLayerNormConfig, run_bert_bolt_layernorm_mpc
from operators.residual_add.method_runtime_default import ResidualAddConfig, run_residual_add_semantic
from operators.softmax.method_mpc_bolt import BertBoltSoftmaxConfig, run_bert_bolt_softmax_mpc
from operators.softmax.method_he_nexus import NexusHeSoftmaxConfig, run_nexus_softmax_he
from runtime.capabilities import CapabilityStatus, capability_registry
from runtime.operator_registry import OperatorRegistry
from runtime.types import BackendType, ExecutionContext, TensorValue


def _arr(t: TensorValue) -> np.ndarray:
    return np.asarray(t.data, dtype=np.float64)


def _linear(x: np.ndarray, out_dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((x.shape[-1], out_dim))
    b = rng.standard_normal((out_dim,))
    return x @ w + b


def _backend_scale(backend: BackendType) -> float:
    return {
        BackendType.MPC: 1.0,
        BackendType.HE: 0.99,
        BackendType.HYBRID: 1.01,
    }[backend]


def _log(ctx: ExecutionContext, op_name: str, backend: BackendType) -> None:
    ctx.trace.append(f"{op_name}@{backend.value}")


def _raise_he_unsupported(op_name: str) -> None:
    raise NotImplementedError(
        f"{op_name}@HE is marked unsupported in this stage. "
        "See operators/*/method_he_nexus.py for integration notes."
    )


def register_default_backend_impls(registry: OperatorRegistry) -> None:
    for backend in BackendType:
        _register_with_method_aliases(registry, "Embedding", backend, _mk_embedding(backend), _method_name("Embedding", backend))
        _register_with_method_aliases(
            registry, "Linear_QKV", backend, _mk_linear_qkv(backend), _method_name("Linear_QKV", backend)
        )
        _register_with_method_aliases(
            registry,
            "Attention_QK_MatMul",
            backend,
            _mk_qk_matmul(backend),
            _method_name("Attention_QK_MatMul", backend),
        )
        _register_with_method_aliases(registry, "Softmax", backend, _mk_softmax(backend), _method_name("Softmax", backend))
        _register_with_method_aliases(
            registry,
            "Attention_V_MatMul",
            backend,
            _mk_v_matmul(backend),
            _method_name("Attention_V_MatMul", backend),
        )
        _register_with_method_aliases(
            registry, "Out_Projection", backend, _mk_out_proj(backend), _method_name("Out_Projection", backend)
        )
        _register_with_method_aliases(
            registry, "Residual_Add", backend, _mk_residual_add(backend), _method_name("Residual_Add", backend)
        )
        _register_with_method_aliases(
            registry, "LayerNorm", backend, _mk_layer_norm(backend), _method_name("LayerNorm", backend)
        )
        _register_with_method_aliases(
            registry, "FFN_Linear_1", backend, _mk_ffn_1(backend), _method_name("FFN_Linear_1", backend)
        )
        _register_with_method_aliases(registry, "GeLU", backend, _mk_gelu(backend), _method_name("GeLU", backend))
        _register_with_method_aliases(
            registry, "FFN_Linear_2", backend, _mk_ffn_2(backend), _method_name("FFN_Linear_2", backend)
        )
    # Explicit backend capability/status registry.
    capability_registry.set_status("GeLU", BackendType.MPC, CapabilityStatus.REAL_INTEGRATED)
    capability_registry.set_status("Softmax", BackendType.MPC, CapabilityStatus.REAL_INTEGRATED)
    capability_registry.set_status("LayerNorm", BackendType.MPC, CapabilityStatus.REAL_INTEGRATED)
    capability_registry.set_status("FFN_Linear_1", BackendType.MPC, CapabilityStatus.REAL_INTEGRATED)
    capability_registry.set_status("Attention_QK_MatMul", BackendType.MPC, CapabilityStatus.REAL_INTEGRATED)
    capability_registry.set_status("Attention_V_MatMul", BackendType.MPC, CapabilityStatus.REAL_INTEGRATED)
    capability_registry.set_status("Residual_Add", BackendType.MPC, CapabilityStatus.REAL_INTEGRATED)
    capability_registry.set_status("FFN_Linear_2", BackendType.MPC, CapabilityStatus.REAL_INTEGRATED)
    capability_registry.set_status("GeLU", BackendType.HE, CapabilityStatus.REAL_INTEGRATED)
    capability_registry.set_status("Softmax", BackendType.HE, CapabilityStatus.REAL_INTEGRATED)
    capability_registry.set_status("Residual_Add", BackendType.HE, CapabilityStatus.REAL_INTEGRATED)
    capability_registry.set_status("LayerNorm", BackendType.HE, CapabilityStatus.RESTRICTED_INTEGRATED)
    capability_registry.set_status("FFN_Linear_1", BackendType.HE, CapabilityStatus.RESTRICTED_INTEGRATED)
    capability_registry.set_status("FFN_Linear_2", BackendType.HE, CapabilityStatus.RESTRICTED_INTEGRATED)
    capability_registry.set_status("Attention_QK_MatMul", BackendType.HE, CapabilityStatus.RESTRICTED_INTEGRATED)
    capability_registry.set_status("Attention_V_MatMul", BackendType.HE, CapabilityStatus.RESTRICTED_INTEGRATED)


def _register_with_method_aliases(
    registry: OperatorRegistry,
    op_name: str,
    backend: BackendType,
    fn,
    explicit_method_name: str,
) -> None:
    registry.register(op_name, backend, fn)
    if explicit_method_name != "method_default":
        registry.register(op_name, backend, fn, method_name=explicit_method_name)


def _method_name(op_name: str, backend: BackendType) -> str:
    if op_name == "Residual_Add":
        return "method_runtime_default"
    if op_name in {"Embedding", "Linear_QKV", "Out_Projection"}:
        return "method_runtime_default"
    if backend == BackendType.MPC:
        if op_name in {"Attention_QK_MatMul", "Attention_V_MatMul"}:
            return "method_mpc"
        return "method_mpc_bolt"
    if backend == BackendType.HE:
        return "method_he_nexus"
    return "method_runtime_default"


def _mk_embedding(backend: BackendType):
    def fn(inputs: List[TensorValue], ctx: ExecutionContext) -> TensorValue:
        _log(ctx, "Embedding", backend)
        x = _arr(inputs[0]) * _backend_scale(backend)
        return TensorValue(x, backend)

    return fn


def _mk_linear_qkv(backend: BackendType):
    def fn(inputs: List[TensorValue], ctx: ExecutionContext) -> TensorValue:
        _log(ctx, "Linear_QKV", backend)
        x = _arr(inputs[0])
        q = _linear(x, x.shape[-1], 11)
        k = _linear(x, x.shape[-1], 13)
        v = _linear(x, x.shape[-1], 17)
        return TensorValue(np.stack([q, k, v], axis=0) * _backend_scale(backend), backend)

    return fn


def _mk_qk_matmul(backend: BackendType):
    def fn(inputs: List[TensorValue], ctx: ExecutionContext) -> TensorValue:
        _log(ctx, "Attention_QK_MatMul", backend)
        if backend == BackendType.MPC:
            scores = run_bert_bolt_attention_qk_matmul_mpc(
                [_arr(t) for t in inputs],
                ctx=ctx,
                cfg=BertBoltAttentionQkMatMulConfig(
                    ell=int(ctx.params.get("attention_qk_ell", ctx.params.get("attention_ell", 37))),
                    scale=int(ctx.params.get("attention_qk_scale", ctx.params.get("attention_scale", 12))),
                    nthreads=int(
                        ctx.params.get("attention_qk_nthreads", ctx.params.get("attention_nthreads", 2))
                    ),
                    port=ctx.params.get("attention_qk_port", ctx.params.get("attention_port")),
                ),
            )
        elif backend == BackendType.HE:
            scores = run_nexus_attention_qk_matmul_he(
                [_arr(t) for t in inputs],
                ctx=ctx,
                cfg=NexusHeAttentionQkMatMulConfig(
                    hidden_size=int(ctx.params.get("attention_he_nexus_hidden_size", 768)),
                    num_heads=int(ctx.params.get("attention_he_nexus_num_heads", 12)),
                    max_seq_len=int(ctx.params.get("attention_he_nexus_max_seq_len", 128)),
                    max_batch=int(ctx.params.get("attention_he_nexus_max_batch", 8)),
                ),
            )
        else:
            qkv = _arr(inputs[0])
            q, k = qkv[0], qkv[1]
            scores = q @ np.swapaxes(k, -1, -2)
        return TensorValue(scores * _backend_scale(backend), backend)

    return fn


def _mk_softmax(backend: BackendType):
    def fn(inputs: List[TensorValue], ctx: ExecutionContext) -> TensorValue:
        _log(ctx, "Softmax", backend)
        x = _arr(inputs[0])
        if backend == BackendType.MPC:
            y = run_bert_bolt_softmax_mpc(
                x,
                ctx=ctx,
                cfg=BertBoltSoftmaxConfig(
                    ell=int(ctx.params.get("softmax_ell", 37)),
                    scale=int(ctx.params.get("softmax_scale", 12)),
                    nthreads=int(ctx.params.get("softmax_nthreads", 2)),
                    port=ctx.params.get("softmax_port"),
                ),
            )
        elif backend == BackendType.HE:
            y = run_nexus_softmax_he(
                x,
                ctx=ctx,
                cfg=NexusHeSoftmaxConfig(
                    inverse_iterations=int(ctx.params.get("softmax_he_nexus_inverse_iterations", 4)),
                    sum_scale_factor=float(ctx.params.get("softmax_he_nexus_sum_scale_factor", 0.01)),
                    eps=float(ctx.params.get("softmax_he_nexus_eps", 1e-8)),
                ),
            )
        else:
            x = x - x.max(axis=-1, keepdims=True)
            ex = np.exp(x)
            y = ex / ex.sum(axis=-1, keepdims=True)
        return TensorValue(y * _backend_scale(backend), backend)

    return fn


def _mk_v_matmul(backend: BackendType):
    def fn(inputs: List[TensorValue], ctx: ExecutionContext) -> TensorValue:
        _log(ctx, "Attention_V_MatMul", backend)
        if backend == BackendType.MPC:
            ctx_out = run_bert_bolt_attention_v_matmul_mpc(
                [_arr(t) for t in inputs],
                ctx=ctx,
                cfg=BertBoltAttentionVMatMulConfig(
                    ell=int(ctx.params.get("attention_v_ell", ctx.params.get("attention_ell", 37))),
                    scale=int(ctx.params.get("attention_v_scale", ctx.params.get("attention_scale", 12))),
                    nthreads=int(ctx.params.get("attention_v_nthreads", ctx.params.get("attention_nthreads", 2))),
                    port=ctx.params.get("attention_v_port", ctx.params.get("attention_port")),
                ),
            )
        elif backend == BackendType.HE:
            ctx_out = run_nexus_attention_v_matmul_he(
                [_arr(t) for t in inputs],
                ctx=ctx,
                cfg=NexusHeAttentionVMatMulConfig(
                    hidden_size=int(ctx.params.get("attention_he_nexus_hidden_size", 768)),
                    num_heads=int(ctx.params.get("attention_he_nexus_num_heads", 12)),
                    max_seq_len=int(ctx.params.get("attention_he_nexus_max_seq_len", 128)),
                    max_batch=int(ctx.params.get("attention_he_nexus_max_batch", 8)),
                    return_canonical=bool(ctx.params.get("attention_return_canonical", False)),
                ),
            )
        else:
            attn = _arr(inputs[0])
            qkv = _arr(inputs[1])
            v = qkv[2]
            ctx_out = attn @ v
        return TensorValue(ctx_out * _backend_scale(backend), backend)

    return fn


def _mk_out_proj(backend: BackendType):
    def fn(inputs: List[TensorValue], ctx: ExecutionContext) -> TensorValue:
        _log(ctx, "Out_Projection", backend)
        x = _arr(inputs[0])
        return TensorValue(_linear(x, x.shape[-1], 19) * _backend_scale(backend), backend)

    return fn


def _mk_residual_add(backend: BackendType):
    def fn(inputs: List[TensorValue], ctx: ExecutionContext) -> TensorValue:
        _log(ctx, "Residual_Add", backend)
        x = run_residual_add_semantic(
            [_arr(inputs[0]), _arr(inputs[1])],
            backend=backend,
            ctx=ctx,
            cfg=ResidualAddConfig(require_same_shape=True),
        )
        return TensorValue(x * _backend_scale(backend), backend)

    return fn


def _mk_layer_norm(backend: BackendType):
    def fn(inputs: List[TensorValue], ctx: ExecutionContext) -> TensorValue:
        _log(ctx, "LayerNorm", backend)
        x = _arr(inputs[0])
        if backend == BackendType.MPC:
            y = run_bert_bolt_layernorm_mpc(
                x,
                ctx=ctx,
                cfg=BertBoltLayerNormConfig(
                    ell=int(ctx.params.get("layernorm_ell", 37)),
                    scale=int(ctx.params.get("layernorm_scale", 12)),
                    nthreads=int(ctx.params.get("layernorm_nthreads", 2)),
                    port=ctx.params.get("layernorm_port"),
                ),
            )
        elif backend == BackendType.HE:
            y = run_nexus_layernorm_he(
                x,
                ctx=ctx,
                cfg=NexusHeLayerNormConfig(
                    hidden_size=int(ctx.params.get("layernorm_he_nexus_hidden_size", 768)),
                    max_tokens=int(ctx.params.get("layernorm_he_nexus_max_tokens", 16)),
                    packed_len=int(ctx.params.get("layernorm_he_nexus_packed_len", 1024)),
                    eps=float(ctx.params.get("layernorm_he_nexus_eps", 1e-8)),
                ),
            )
        else:
            mu = x.mean(axis=-1, keepdims=True)
            var = ((x - mu) ** 2).mean(axis=-1, keepdims=True)
            y = (x - mu) / np.sqrt(var + 1e-6)
        return TensorValue(y * _backend_scale(backend), backend)

    return fn


def _mk_ffn_1(backend: BackendType):
    def fn(inputs: List[TensorValue], ctx: ExecutionContext) -> TensorValue:
        _log(ctx, "FFN_Linear_1", backend)
        x = _arr(inputs[0])
        if backend == BackendType.MPC:
            out_dim = int(ctx.params.get("ffn_linear1_output_dim", x.shape[-1] * 2))
            y, _, _ = run_bert_bolt_ffn_linear1_mpc(
                x,
                out_dim=out_dim,
                ctx=ctx,
                cfg=BertBoltFfnLinear1Config(
                    ell=int(ctx.params.get("ffn_linear1_ell", 37)),
                    scale=int(ctx.params.get("ffn_linear1_scale", 12)),
                    nthreads=int(ctx.params.get("ffn_linear1_nthreads", 2)),
                    port=ctx.params.get("ffn_linear1_port"),
                    weight_seed=int(ctx.params.get("ffn_linear1_weight_seed", 1234)),
                ),
            )
        elif backend == BackendType.HE:
            y = run_nexus_linear_ffn1_he(
                x,
                ctx=ctx,
                cfg=NexusHeLinearFfn1Config(
                    hidden_size=int(ctx.params.get("ffn_linear1_he_nexus_hidden_size", 768)),
                    out_dim=int(ctx.params.get("ffn_linear1_he_nexus_out_dim", 64)),
                    max_tokens=int(ctx.params.get("ffn_linear1_he_nexus_max_tokens", 4096)),
                    poly_modulus_degree=int(ctx.params.get("ffn_linear1_he_nexus_poly_degree", 4096)),
                    weight_seed=int(ctx.params.get("ffn_linear1_he_nexus_weight_seed", 1234)),
                ),
            )
        else:
            y = _linear(x, x.shape[-1] * 2, 23)
        return TensorValue(y * _backend_scale(backend), backend)

    return fn


def _mk_gelu(backend: BackendType):
    def fn(inputs: List[TensorValue], ctx: ExecutionContext) -> TensorValue:
        _log(ctx, "GeLU", backend)
        x = _arr(inputs[0])
        if backend == BackendType.MPC:
            y = run_bert_bolt_gelu_mpc(
                x,
                ctx=ctx,
                cfg=BertBoltGeluConfig(
                    ell=int(ctx.params.get("gelu_ell", 37)),
                    scale=int(ctx.params.get("gelu_scale", 12)),
                    nthreads=int(ctx.params.get("gelu_nthreads", 2)),
                    port=ctx.params.get("gelu_port"),
                ),
            )
        elif backend == BackendType.HE:
            y = run_nexus_gelu_he(
                x,
                ctx=ctx,
                cfg=NexusHeGeluConfig(
                    clamp_min=float(ctx.params.get("gelu_he_nexus_clamp_min", -8.0)),
                    clamp_max=float(ctx.params.get("gelu_he_nexus_clamp_max", 8.0)),
                ),
            )
        else:
            y = 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))
        return TensorValue(y * _backend_scale(backend), backend)

    return fn


def _mk_ffn_2(backend: BackendType):
    def fn(inputs: List[TensorValue], ctx: ExecutionContext) -> TensorValue:
        _log(ctx, "FFN_Linear_2", backend)
        x = _arr(inputs[0])
        if backend == BackendType.MPC:
            out_dim = int(ctx.params.get("ffn_linear2_output_dim", x.shape[-1] // 2))
            y, _, _ = run_bert_bolt_ffn_linear2_mpc(
                x,
                out_dim=out_dim,
                ctx=ctx,
                cfg=BertBoltFfnLinear2Config(
                    ell=int(ctx.params.get("ffn_linear2_ell", 37)),
                    scale=int(ctx.params.get("ffn_linear2_scale", 12)),
                    nthreads=int(ctx.params.get("ffn_linear2_nthreads", 2)),
                    port=ctx.params.get("ffn_linear2_port"),
                    weight_seed=int(ctx.params.get("ffn_linear2_weight_seed", 2234)),
                ),
            )
        elif backend == BackendType.HE:
            y = run_nexus_linear_ffn2_he(
                x,
                ctx=ctx,
                cfg=NexusHeLinearFfn2Config(
                    hidden_size=int(ctx.params.get("ffn_linear2_he_nexus_hidden_size", x.shape[-1])),
                    out_dim=int(ctx.params.get("ffn_linear2_he_nexus_out_dim", x.shape[-1] // 2)),
                    max_tokens=int(ctx.params.get("ffn_linear2_he_nexus_max_tokens", 4096)),
                    poly_modulus_degree=int(ctx.params.get("ffn_linear2_he_nexus_poly_degree", 4096)),
                    weight_seed=int(ctx.params.get("ffn_linear2_he_nexus_weight_seed", 2234)),
                ),
            )
        else:
            y = _linear(x, x.shape[-1] // 2, 29)
        return TensorValue(y * _backend_scale(backend), backend)

    return fn
