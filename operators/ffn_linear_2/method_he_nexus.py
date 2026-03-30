from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from backends.he_nexus_linear_ffn1_adapter import (
    NexusLinearFfn1RestrictedAdapterConfig,
    run_nexus_linear_ffn1_restricted_adapter,
)
from runtime.types import ExecutionContext


@dataclass
class NexusHeLinearFfn2Config:
    hidden_size: int = 1536
    out_dim: int = 768
    max_tokens: int = 4096
    poly_modulus_degree: int = 4096
    weight_seed: int = 2234


def _log(ctx: ExecutionContext | None, message: str) -> None:
    if ctx is not None:
        ctx.trace.append(message)


def run_nexus_linear_ffn2_he(
    x: np.ndarray,
    ctx: ExecutionContext | None = None,
    cfg: NexusHeLinearFfn2Config | None = None,
) -> np.ndarray:
    cfg = cfg or NexusHeLinearFfn2Config()
    params = {} if ctx is None else ctx.params

    weight = params.get("ffn_linear2_he_nexus_weight")
    bias = params.get("ffn_linear2_he_nexus_bias")
    _log(
        ctx,
        "[ffn_linear2_he_nexus] lowered_to=FFN_Linear_1.method_he_nexus "
        "primitive=MMEvaluator::matrix_mul",
    )
    y, _, _ = run_nexus_linear_ffn1_restricted_adapter(
        np.asarray(x, dtype=np.float64),
        cfg=NexusLinearFfn1RestrictedAdapterConfig(
            hidden_size=int(params.get("ffn_linear2_he_nexus_hidden_size", cfg.hidden_size)),
            out_dim=int(params.get("ffn_linear2_he_nexus_out_dim", cfg.out_dim)),
            max_tokens=int(params.get("ffn_linear2_he_nexus_max_tokens", cfg.max_tokens)),
            poly_modulus_degree=int(params.get("ffn_linear2_he_nexus_poly_degree", cfg.poly_modulus_degree)),
            weight_seed=int(params.get("ffn_linear2_he_nexus_weight_seed", cfg.weight_seed)),
        ),
        weight=weight,
        bias=bias,
    )
    return y
