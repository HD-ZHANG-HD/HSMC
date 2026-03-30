from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from backends.he_nexus_attention_adapter import (
    NexusAttentionRestrictedConfig,
    run_nexus_attention_qk_restricted_adapter,
)


def _to_bhsd(x: np.ndarray, heads: int) -> np.ndarray:
    bsz, seq, hidden = x.shape
    d = hidden // heads
    return x.reshape(bsz, seq, heads, d).transpose(0, 2, 1, 3)


def run_case(name: str, b: int, s: int) -> None:
    cfg = NexusAttentionRestrictedConfig()
    qkv = np.random.standard_normal((3, b, s, cfg.hidden_size))
    out = run_nexus_attention_qk_restricted_adapter(qkv, cfg=cfg)
    q = _to_bhsd(qkv[0], cfg.num_heads)
    k = _to_bhsd(qkv[1], cfg.num_heads)
    ref = q @ np.swapaxes(k, -1, -2)
    mae = float(np.mean(np.abs(out - ref)))
    assert out.shape == (b, cfg.num_heads, s, s), f"{name}: shape mismatch {out.shape}"
    assert mae < 1e-9, f"{name}: MAE too high {mae}"
    print(f"[bridge-test] {name}: shape_ok=True mae={mae:.6e}")


def main() -> None:
    run_case("small_sanity", 1, 2)
    run_case("multi_batch", 4, 8)
    run_case("nontrivial_seq", 2, 16)


if __name__ == "__main__":
    main()
