from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from operators.linear_ffn1.method_he_nexus import NexusHeLinearFfn1Config, run_nexus_linear_ffn1_he
from runtime.types import ExecutionContext


def run_supported_case(name: str, b: int, s: int) -> None:
    cfg = NexusHeLinearFfn1Config()
    ctx = ExecutionContext(params={"ffn_linear1_he_nexus_weight_seed": 2024})
    x = np.random.standard_normal((b, s, cfg.hidden_size))
    y = run_nexus_linear_ffn1_he(x, ctx=ctx, cfg=cfg)
    assert y.shape == (b, s, cfg.out_dim), f"{name}: shape mismatch {y.shape}"
    assert np.isfinite(y).all(), f"{name}: non-finite output"
    print(f"[wrapper-test] {name}: shape_ok=True")


def run_restriction_checks() -> None:
    cfg = NexusHeLinearFfn1Config()
    try:
        run_nexus_linear_ffn1_he(np.zeros((1, 2, 512)), ctx=ExecutionContext(), cfg=cfg)
    except ValueError:
        print("[wrapper-test] restriction_hidden_ok=True")
    else:
        raise AssertionError("Expected restricted hidden-size validation failure")

    try:
        run_nexus_linear_ffn1_he(np.zeros((100, 100, 768)), ctx=ExecutionContext(), cfg=cfg)
    except ValueError:
        print("[wrapper-test] restriction_tokens_ok=True")
    else:
        raise AssertionError("Expected restricted token-count validation failure")


def main() -> None:
    run_supported_case("small_sanity", 1, 2)
    run_supported_case("multi_batch", 4, 8)
    run_supported_case("nontrivial_tokens", 8, 16)
    run_restriction_checks()


if __name__ == "__main__":
    main()
