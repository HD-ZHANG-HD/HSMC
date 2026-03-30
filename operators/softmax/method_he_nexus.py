from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from backends.he_nexus import NexusHeSoftmaxBridgeConfig, run_nexus_softmax_bridge
from runtime.types import ExecutionContext


@dataclass
class NexusHeSoftmaxConfig:
    """
    HE Softmax wrapper using NEXUS-based approximation.

    Wrapped NEXUS references:
    - he_compiler/NEXUS/src/softmax.cpp -> SoftmaxEvaluator::softmax
    - he_compiler/NEXUS/src/ckks_evaluator.cpp ->
      CKKSEvaluator::exp / CKKSEvaluator::inverse

    Shape contract:
    - input: tensor with ndim>=2, softmax applied over last axis
    - output: same shape as input

    Parameter handling:
    - inverse_iterations controls NEXUS-style inverse approximation depth.
    - sum_scale_factor mirrors NEXUS pre/post scaling around inverse.

    Approximation note:
    - This method is approximate by design.
    - Python wrapper execution here is plaintext emulation, not ciphertext execution.
    """

    inverse_iterations: int = 4
    sum_scale_factor: float = 0.01
    eps: float = 1e-8


def run_nexus_softmax_he(
    x: np.ndarray,
    ctx: ExecutionContext | None = None,
    cfg: NexusHeSoftmaxConfig | None = None,
) -> np.ndarray:
    del ctx  # Reserved for future runtime controls.
    cfg = cfg or NexusHeSoftmaxConfig()
    bridge_cfg = NexusHeSoftmaxBridgeConfig(
        inverse_iterations=cfg.inverse_iterations,
        sum_scale_factor=cfg.sum_scale_factor,
        eps=cfg.eps,
    )
    return run_nexus_softmax_bridge(np.asarray(x, dtype=np.float64), bridge_cfg)
