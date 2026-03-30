from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class OperatorSpec:
    name: str
    input_names: List[str]
    output_name: str


BERT_OPERATOR_SEQUENCE: List[OperatorSpec] = [
    OperatorSpec("Embedding", ["input"], "embedding_out"),
    OperatorSpec("Linear_QKV", ["embedding_out"], "qkv_out"),
    OperatorSpec("Attention_QK_MatMul", ["qkv_out"], "qk_scores"),
    OperatorSpec("Softmax", ["qk_scores"], "attn_probs"),
    OperatorSpec("Attention_V_MatMul", ["attn_probs", "qkv_out"], "context"),
    OperatorSpec("Out_Projection", ["context"], "attn_proj"),
    OperatorSpec("Residual_Add", ["attn_proj", "embedding_out"], "attn_residual"),
    OperatorSpec("LayerNorm", ["attn_residual"], "attn_norm"),
    OperatorSpec("FFN_Linear_1", ["attn_norm"], "ffn_hidden"),
    OperatorSpec("GeLU", ["ffn_hidden"], "ffn_activated"),
    OperatorSpec("FFN_Linear_2", ["ffn_activated"], "ffn_out"),
]

