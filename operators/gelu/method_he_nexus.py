from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from backends.he_nexus import NexusHeGeluBridgeConfig, run_nexus_gelu_bridge
from runtime.types import ExecutionContext


@dataclass
class NexusHeGeluConfig:
    """
    HE GeLU wrapper using NEXUS-based approximation.

    Wrapped NEXUS references:
    - he_compiler/NEXUS/src/gelu.cpp -> GeLUEvaluator::gelu
    - he_compiler/NEXUS/data/data_generation.py -> gelu calibration target

    Shape contract:
    - input: any float tensor, typically [B,S,H]
    - output: same shape as input

    Parameter handling:
    - clamp_min / clamp_max bound the plaintext input before approximation.

    Approximation note:
    - This is an approximate HE-style method based on NEXUS math.
    - Python wrapper execution here is plaintext emulation, not ciphertext execution.
    """

    clamp_min: float = -8.0
    clamp_max: float = 8.0


def run_nexus_gelu_he(
    x: np.ndarray,
    ctx: ExecutionContext | None = None,
    cfg: NexusHeGeluConfig | None = None,
) -> np.ndarray:
    del ctx  # Reserved for future runtime controls.
    cfg = cfg or NexusHeGeluConfig()
    bridge_cfg = NexusHeGeluBridgeConfig(clamp_min=cfg.clamp_min, clamp_max=cfg.clamp_max)
    return run_nexus_gelu_bridge(np.asarray(x, dtype=np.float64), bridge_cfg)
