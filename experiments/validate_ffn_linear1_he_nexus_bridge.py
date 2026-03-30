from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from backends.he_nexus_linear_ffn1_adapter import (
    NexusLinearFfn1RestrictedAdapterConfig,
    run_nexus_linear_ffn1_restricted_adapter,
)


def run_case(name: str, b: int, s: int) -> None:
    cfg = NexusLinearFfn1RestrictedAdapterConfig()
    x = np.random.standard_normal((b, s, cfg.hidden_size))
    y, w, bias = run_nexus_linear_ffn1_restricted_adapter(x, cfg=cfg)
    ref = x.reshape(b * s, cfg.hidden_size) @ w + bias
    ref = ref.reshape(b, s, cfg.out_dim)
    mae = float(np.mean(np.abs(y - ref)))
    assert y.shape == (b, s, cfg.out_dim), f"{name}: shape mismatch {y.shape}"
    assert np.isfinite(y).all(), f"{name}: non-finite output"
    assert mae < 1e-9, f"{name}: MAE too high {mae}"
    print(f"[bridge-test] {name}: shape_ok=True mae={mae:.6e}")


def main() -> None:
    run_case("small_sanity", 1, 2)
    run_case("multi_batch", 4, 8)
    run_case("nontrivial_tokens", 8, 16)


if __name__ == "__main__":
    main()
