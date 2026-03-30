from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from backends.he_nexus_attention_adapter import (
    NexusAttentionRestrictedConfig,
    run_nexus_attention_qk_restricted_adapter,
)
from runtime.types import ExecutionContext


@dataclass
class NexusHeAttentionQkMatMulConfig:
    """
    Restricted NEXUS-backed HE adapter for Attention_QK_MatMul.

    Wrapped NEXUS internals:
    - he_compiler/NEXUS/src/matrix_mul.cpp (MMEvaluator matrix-mul packing model)

    Restricted contract:
    - inputs: packed qkv only, shape [3,B,S,768]
    - heads fixed at 12 (head_dim=64)
    - 1<=B<=8, 1<=S<=128
    - output: [B,12,S,S]

    Status label:
    - restricted-integrated
    """

    hidden_size: int = 768
    num_heads: int = 12
    max_seq_len: int = 128
    max_batch: int = 8


def run_nexus_attention_qk_matmul_he(
    inputs: list[np.ndarray],
    ctx: ExecutionContext | None = None,
    cfg: NexusHeAttentionQkMatMulConfig | None = None,
) -> np.ndarray:
    del ctx
    cfg = cfg or NexusHeAttentionQkMatMulConfig()
    if len(inputs) != 1:
        raise ValueError(
            "Restricted Attention_QK_MatMul HE adapter requires one packed qkv input: [3,B,S,768]"
        )
    qkv_packed = np.asarray(inputs[0], dtype=np.float64)
    return run_nexus_attention_qk_restricted_adapter(
        qkv_packed,
        cfg=NexusAttentionRestrictedConfig(
            hidden_size=cfg.hidden_size,
            num_heads=cfg.num_heads,
            max_seq_len=cfg.max_seq_len,
            max_batch=cfg.max_batch,
        ),
    )
