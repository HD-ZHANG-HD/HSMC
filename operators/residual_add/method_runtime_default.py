from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from runtime.types import BackendType, ExecutionContext


@dataclass
class ResidualAddConfig:
    require_same_shape: bool = True


def _log(ctx: ExecutionContext | None, message: str) -> None:
    if ctx is not None:
        ctx.trace.append(message)


def run_residual_add_semantic(
    inputs: list[np.ndarray],
    backend: BackendType,
    ctx: ExecutionContext | None = None,
    cfg: ResidualAddConfig | None = None,
) -> np.ndarray:
    cfg = cfg or ResidualAddConfig()
    if len(inputs) != 2:
        raise ValueError(f"Residual_Add expects exactly two inputs, got {len(inputs)}")
    a = np.asarray(inputs[0], dtype=np.float64)
    b = np.asarray(inputs[1], dtype=np.float64)
    if cfg.require_same_shape and a.shape != b.shape:
        raise ValueError(f"Residual_Add requires identical input shapes, got {a.shape} and {b.shape}")
    _log(ctx, f"[residual_add_semantic] backend={backend.value} lowered_to=backend_tensor_add")
    return a + b
