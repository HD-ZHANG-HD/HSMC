from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from backends.he_nexus_linear_ffn1_adapter import (
    NexusLinearFfn1RestrictedAdapterConfig,
    run_nexus_linear_ffn1_restricted_adapter,
)
from runtime.types import ExecutionContext


@dataclass
class NexusHeLinearFfn1Config:
    """
    Restricted NEXUS-backed HE adapter for FFN_Linear_1.

    Wrapped NEXUS internals:
    - he_compiler/NEXUS/src/matrix_mul.cpp:
      - MMEvaluator::matrix_mul
      - row-pack logic from MM_test()

    Restricted contract:
    - input x: [B,S,H], with H=768
    - output y: [B,S,64]
    - requires 1 <= B*S <= 4096
    - optional weight/bias must be [768,64] and [64]

    Status label:
    - restricted-integrated
    """

    hidden_size: int = 768
    out_dim: int = 64
    max_tokens: int = 4096
    poly_modulus_degree: int = 4096
    weight_seed: int = 1234


def run_nexus_linear_ffn1_he(
    x: np.ndarray,
    ctx: ExecutionContext | None = None,
    cfg: NexusHeLinearFfn1Config | None = None,
) -> np.ndarray:
    cfg = cfg or NexusHeLinearFfn1Config()
    params = {} if ctx is None else ctx.params

    weight = params.get("ffn_linear1_he_nexus_weight")
    bias = params.get("ffn_linear1_he_nexus_bias")
    y, _, _ = run_nexus_linear_ffn1_restricted_adapter(
        np.asarray(x, dtype=np.float64),
        cfg=NexusLinearFfn1RestrictedAdapterConfig(
            hidden_size=int(params.get("ffn_linear1_he_nexus_hidden_size", cfg.hidden_size)),
            out_dim=int(params.get("ffn_linear1_he_nexus_out_dim", cfg.out_dim)),
            max_tokens=int(params.get("ffn_linear1_he_nexus_max_tokens", cfg.max_tokens)),
            poly_modulus_degree=int(params.get("ffn_linear1_he_nexus_poly_degree", cfg.poly_modulus_degree)),
            weight_seed=int(params.get("ffn_linear1_he_nexus_weight_seed", cfg.weight_seed)),
        ),
        weight=weight,
        bias=bias,
    )
    return y
