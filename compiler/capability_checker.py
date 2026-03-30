from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Tuple


Shape = Tuple[int, ...]
Attributes = Mapping[str, object]
ContractFn = Callable[[Shape, Attributes], bool]


STATUS_REAL = "real-integrated"
STATUS_RESTRICTED = "restricted-integrated"
STATUS_MOCK = "mock"


@dataclass(frozen=True)
class MethodCapability:
    op_type: str
    domain: str
    method: str
    status: str
    contract: ContractFn
    notes: str = ""


@dataclass
class CapabilityChecker:
    registry: Dict[str, List[MethodCapability]] = field(default_factory=dict)

    def register(self, capability: MethodCapability) -> None:
        self.registry.setdefault(capability.op_type, []).append(capability)

    def get_method_specs(self, op_type: str) -> List[MethodCapability]:
        return list(self.registry.get(op_type, []))

    def is_method_valid(
        self,
        op_type: str,
        method: str,
        input_shape: Shape | List[int],
        attributes: Attributes | None = None,
    ) -> bool:
        shape = _shape(input_shape)
        attrs = dict(attributes or {})
        for spec in self.registry.get(op_type, []):
            if spec.method != method:
                continue
            return spec.contract(shape, attrs)
        return False

    def get_valid_methods(
        self,
        op_type: str,
        input_shape: Shape | List[int],
        attributes: Attributes | None = None,
    ) -> List[str]:
        shape = _shape(input_shape)
        attrs = dict(attributes or {})
        valid: List[str] = []
        for spec in self.registry.get(op_type, []):
            if spec.contract(shape, attrs):
                valid.append(spec.method)
        return valid

    def snapshot(self) -> Dict[str, List[Dict[str, str]]]:
        return {
            op_type: [
                {
                    "domain": spec.domain,
                    "method": spec.method,
                    "status": spec.status,
                    "notes": spec.notes,
                }
                for spec in specs
            ]
            for op_type, specs in self.registry.items()
        }


def _shape(values: Shape | List[int]) -> Shape:
    return tuple(int(v) for v in values)


def _same_rank(shape: Shape, rank: int) -> bool:
    return len(shape) == rank


def _get_int(attrs: Attributes, key: str, default: int) -> int:
    value = attrs.get(key, default)
    return int(value)


def _get_shape(attrs: Attributes, key: str) -> Shape | None:
    value = attrs.get(key)
    if value is None:
        return None
    return tuple(int(v) for v in value)  # type: ignore[arg-type]


def _basic_hidden_tensor(shape: Shape) -> bool:
    return _same_rank(shape, 3) and all(dim >= 1 for dim in shape)


def _always_valid(shape: Shape, attrs: Attributes) -> bool:
    del shape, attrs
    return True


def _layernorm_he_nexus(shape: Shape, attrs: Attributes) -> bool:
    if not _same_rank(shape, 3):
        return False
    bsz, seq, hidden = shape
    if hidden != 768:
        return False
    if bsz < 1 or seq < 1 or (bsz * seq) > 16:
        return False
    if attrs.get("layernorm_weight") is not None:
        return False
    if attrs.get("layernorm_bias") is not None:
        return False
    return True


def _layernorm_mpc_bolt(shape: Shape, attrs: Attributes) -> bool:
    del attrs
    return _basic_hidden_tensor(shape)


def _attention_qk_he_nexus(shape: Shape, attrs: Attributes) -> bool:
    if not _same_rank(shape, 4):
        return False
    packed, bsz, seq, hidden = shape
    if packed != 3 or hidden != 768:
        return False
    if bsz < 1 or bsz > 8 or seq < 1 or seq > 128:
        return False
    if attrs.get("packed_qkv") is False:
        return False
    if _get_int(attrs, "num_heads", 12) != 12:
        return False
    return True


def _attention_qk_mpc(shape: Shape, attrs: Attributes) -> bool:
    del attrs
    return (_same_rank(shape, 3) and shape[-1] >= 1) or (_same_rank(shape, 4) and shape[0] == 3)


def _attention_v_he_nexus(shape: Shape, attrs: Attributes) -> bool:
    del attrs
    return _same_rank(shape, 4) and shape[1] == 12 and shape[2] == shape[3]


def _attention_v_mpc(shape: Shape, attrs: Attributes) -> bool:
    del attrs
    return _same_rank(shape, 3) or _same_rank(shape, 4)


def _ffn_linear1_he_nexus(shape: Shape, attrs: Attributes) -> bool:
    if not _same_rank(shape, 3):
        return False
    bsz, seq, hidden = shape
    if hidden != 768:
        return False
    if bsz < 1 or seq < 1 or (bsz * seq) > 4096:
        return False
    out_dim = _get_int(attrs, "out_dim", 64)
    if out_dim != 64:
        return False
    weight_shape = _get_shape(attrs, "weight_shape")
    bias_shape = _get_shape(attrs, "bias_shape")
    if weight_shape is not None and weight_shape != (768, 64):
        return False
    if bias_shape is not None and bias_shape != (64,):
        return False
    return True


def _ffn_linear1_mpc(shape: Shape, attrs: Attributes) -> bool:
    del attrs
    return _basic_hidden_tensor(shape)


def _ffn_linear2_he_nexus(shape: Shape, attrs: Attributes) -> bool:
    if not _same_rank(shape, 3):
        return False
    bsz, seq, hidden = shape
    if bsz < 1 or seq < 1 or (bsz * seq) > 4096:
        return False
    expected_hidden = _get_int(attrs, "hidden_size", hidden)
    if hidden != expected_hidden:
        return False
    expected_out = _get_int(attrs, "out_dim", 768)
    weight_shape = _get_shape(attrs, "weight_shape")
    bias_shape = _get_shape(attrs, "bias_shape")
    if weight_shape is not None and weight_shape != (hidden, expected_out):
        return False
    if bias_shape is not None and bias_shape != (expected_out,):
        return False
    return True


def _ffn_linear2_mpc(shape: Shape, attrs: Attributes) -> bool:
    del attrs
    return _basic_hidden_tensor(shape)


def _embedding_runtime_default(shape: Shape, attrs: Attributes) -> bool:
    del attrs
    return _same_rank(shape, 2) and all(dim >= 1 for dim in shape)


def _linear_qkv_runtime_default(shape: Shape, attrs: Attributes) -> bool:
    del attrs
    return _basic_hidden_tensor(shape)


def _out_projection_runtime_default(shape: Shape, attrs: Attributes) -> bool:
    del attrs
    return _basic_hidden_tensor(shape)


def _residual_add_runtime_default(shape: Shape, attrs: Attributes) -> bool:
    del attrs
    return _basic_hidden_tensor(shape)


def build_default_capability_checker() -> CapabilityChecker:
    checker = CapabilityChecker()

    for capability in [
        MethodCapability(
            op_type="LayerNorm",
            domain="HE",
            method="method_he_nexus",
            status=STATUS_RESTRICTED,
            contract=_layernorm_he_nexus,
            notes="Restricted NEXUS LayerNorm: [B,S,768], B*S<=16, no affine weight/bias.",
        ),
        MethodCapability(
            op_type="LayerNorm",
            domain="MPC",
            method="method_mpc_bolt",
            status=STATUS_REAL,
            contract=_layernorm_mpc_bolt,
            notes="MPC LayerNorm bridge path with generic [B,S,H]-style tensor acceptance.",
        ),
        MethodCapability(
            op_type="Attention_QK_MatMul",
            domain="HE",
            method="method_he_nexus",
            status=STATUS_RESTRICTED,
            contract=_attention_qk_he_nexus,
            notes="Restricted NEXUS attention QK path: packed qkv [3,B,S,768], heads=12, B<=8, S<=128.",
        ),
        MethodCapability(
            op_type="Attention_QK_MatMul",
            domain="MPC",
            method="method_mpc",
            status=STATUS_REAL,
            contract=_attention_qk_mpc,
            notes="MPC QK path accepts packed qkv or canonical tensor inputs.",
        ),
        MethodCapability(
            op_type="Attention_V_MatMul",
            domain="HE",
            method="method_he_nexus",
            status=STATUS_RESTRICTED,
            contract=_attention_v_he_nexus,
            notes="Restricted NEXUS attention V path: attn_probs [B,12,S,S] plus packed qkv side input.",
        ),
        MethodCapability(
            op_type="Attention_V_MatMul",
            domain="MPC",
            method="method_mpc",
            status=STATUS_REAL,
            contract=_attention_v_mpc,
            notes="MPC V path accepts current routed attention layouts.",
        ),
        MethodCapability(
            op_type="FFN_Linear_1",
            domain="HE",
            method="method_he_nexus",
            status=STATUS_RESTRICTED,
            contract=_ffn_linear1_he_nexus,
            notes="Restricted NEXUS FFN_Linear_1: [B,S,768], B*S<=4096, out_dim=64.",
        ),
        MethodCapability(
            op_type="FFN_Linear_1",
            domain="MPC",
            method="method_mpc_bolt",
            status=STATUS_REAL,
            contract=_ffn_linear1_mpc,
            notes="MPC FFN_Linear_1 bridge path.",
        ),
        MethodCapability(
            op_type="FFN_Linear_2",
            domain="HE",
            method="method_he_nexus",
            status=STATUS_RESTRICTED,
            contract=_ffn_linear2_he_nexus,
            notes="Restricted NEXUS FFN_Linear_2 lowered to the same matrix-mul primitive family.",
        ),
        MethodCapability(
            op_type="FFN_Linear_2",
            domain="MPC",
            method="method_mpc_bolt",
            status=STATUS_REAL,
            contract=_ffn_linear2_mpc,
            notes="MPC FFN_Linear_2 bridge path.",
        ),
        MethodCapability(
            op_type="Softmax",
            domain="HE",
            method="method_he_nexus",
            status=STATUS_REAL,
            contract=_always_valid,
            notes="Approximate HE softmax wrapper; contract kept broad here.",
        ),
        MethodCapability(
            op_type="Softmax",
            domain="MPC",
            method="method_mpc_bolt",
            status=STATUS_REAL,
            contract=_always_valid,
            notes="MPC softmax bridge path.",
        ),
        MethodCapability(
            op_type="GeLU",
            domain="HE",
            method="method_he_nexus",
            status=STATUS_REAL,
            contract=_always_valid,
            notes="Approximate HE GeLU wrapper.",
        ),
        MethodCapability(
            op_type="GeLU",
            domain="MPC",
            method="method_mpc_bolt",
            status=STATUS_REAL,
            contract=_always_valid,
            notes="MPC GeLU bridge path.",
        ),
        MethodCapability(
            op_type="Residual_Add",
            domain="HE",
            method="method_runtime_default",
            status=STATUS_REAL,
            contract=_residual_add_runtime_default,
            notes="Semantic residual add lowered to backend-native add.",
        ),
        MethodCapability(
            op_type="Residual_Add",
            domain="MPC",
            method="method_runtime_default",
            status=STATUS_REAL,
            contract=_residual_add_runtime_default,
            notes="Semantic residual add lowered to backend-native add.",
        ),
        MethodCapability(
            op_type="Embedding",
            domain="HE",
            method="method_runtime_default",
            status=STATUS_MOCK,
            contract=_embedding_runtime_default,
            notes="Semantic embedding default path.",
        ),
        MethodCapability(
            op_type="Embedding",
            domain="MPC",
            method="method_runtime_default",
            status=STATUS_MOCK,
            contract=_embedding_runtime_default,
            notes="Semantic embedding default path.",
        ),
        MethodCapability(
            op_type="Linear_QKV",
            domain="HE",
            method="method_runtime_default",
            status=STATUS_MOCK,
            contract=_linear_qkv_runtime_default,
            notes="Semantic QKV projection default path.",
        ),
        MethodCapability(
            op_type="Linear_QKV",
            domain="MPC",
            method="method_runtime_default",
            status=STATUS_MOCK,
            contract=_linear_qkv_runtime_default,
            notes="Semantic QKV projection default path.",
        ),
        MethodCapability(
            op_type="Out_Projection",
            domain="HE",
            method="method_runtime_default",
            status=STATUS_MOCK,
            contract=_out_projection_runtime_default,
            notes="Semantic output projection default path.",
        ),
        MethodCapability(
            op_type="Out_Projection",
            domain="MPC",
            method="method_runtime_default",
            status=STATUS_MOCK,
            contract=_out_projection_runtime_default,
            notes="Semantic output projection default path.",
        ),
    ]:
        checker.register(capability)

    return checker


default_capability_checker = build_default_capability_checker()


def is_method_valid(
    op_type: str,
    method: str,
    input_shape: Shape | List[int],
    attributes: Attributes | None = None,
) -> bool:
    return default_capability_checker.is_method_valid(op_type, method, input_shape, attributes)


def get_valid_methods(
    op_type: str,
    input_shape: Shape | List[int],
    attributes: Attributes | None = None,
) -> List[str]:
    return default_capability_checker.get_valid_methods(op_type, input_shape, attributes)
