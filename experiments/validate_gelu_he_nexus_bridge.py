from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from backends.he_nexus import NexusHeGeluBridgeConfig, run_nexus_gelu_bridge


def _gelu_ref(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))


def run_case(name: str, shape: tuple[int, ...]) -> None:
    x = np.random.standard_normal(shape)
    y = run_nexus_gelu_bridge(x, NexusHeGeluBridgeConfig(clamp_min=-8.0, clamp_max=8.0))
    ref = _gelu_ref(np.clip(x, -8.0, 8.0))
    mae = float(np.mean(np.abs(y - ref)))
    assert y.shape == x.shape, f"{name}: shape mismatch {y.shape} vs {x.shape}"
    assert np.isfinite(y).all(), f"{name}: non-finite output"
    assert mae < 1e-9, f"{name}: MAE too high {mae}"
    print(f"[bridge-test] {name}: shape_ok=True mae={mae:.6e}")


def main() -> None:
    run_case("small_sanity", (1, 2, 4))
    run_case("multi_batch", (4, 3, 8))
    run_case("nontrivial_dim", (2, 5, 7))


if __name__ == "__main__":
    main()
