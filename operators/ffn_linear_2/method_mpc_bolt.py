from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from operators.linear_ffn1.method_mpc_bolt import BertBoltFfnLinear1Config, run_bert_bolt_ffn_linear1_mpc
from runtime.types import ExecutionContext


@dataclass
class BertBoltFfnLinear2Config:
    ell: int = 37
    scale: int = 12
    nthreads: int = 2
    address: str = "127.0.0.1"
    port: int | None = None
    weight_seed: int = 2234


def _log(ctx: ExecutionContext | None, message: str) -> None:
    if ctx is not None:
        ctx.trace.append(message)


def run_bert_bolt_ffn_linear2_mpc(
    x: np.ndarray,
    out_dim: int,
    ctx: ExecutionContext | None = None,
    cfg: BertBoltFfnLinear2Config | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cfg = cfg or BertBoltFfnLinear2Config()
    _log(
        ctx,
        "[ffn_linear2_wrapper] lowered_to=FFN_Linear_1.method_mpc_bolt "
        "primitive=NonLinear::n_matrix_mul_iron(...)",
    )
    return run_bert_bolt_ffn_linear1_mpc(
        np.asarray(x, dtype=np.float64),
        out_dim=out_dim,
        ctx=ctx,
        cfg=BertBoltFfnLinear1Config(
            ell=cfg.ell,
            scale=cfg.scale,
            nthreads=cfg.nthreads,
            address=cfg.address,
            port=cfg.port,
            weight_seed=cfg.weight_seed,
        ),
    )
