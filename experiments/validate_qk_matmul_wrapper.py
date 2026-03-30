from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from operators.attention_qk_matmul.method_mpc import (  # noqa: E402
    BertBoltAttentionQkMatMulConfig,
    run_bert_bolt_attention_qk_matmul_mpc,
)
from runtime.types import ExecutionContext  # noqa: E402


def make_packed_qkv(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    bsz, heads, seq, head_dim = q.shape
    hidden = heads * head_dim
    q3 = q.transpose(0, 2, 1, 3).reshape(bsz, seq, hidden)
    k3 = k.transpose(0, 2, 1, 3).reshape(bsz, seq, hidden)
    v3 = v.transpose(0, 2, 1, 3).reshape(bsz, seq, hidden)
    return np.stack([q3, k3, v3], axis=0)


def run_case(name: str, bsz: int, heads: int, seq: int, head_dim: int) -> None:
    q = np.random.standard_normal((bsz, heads, seq, head_dim))
    k = np.random.standard_normal((bsz, heads, seq, head_dim))
    v = np.random.standard_normal((bsz, heads, seq, head_dim))
    packed = make_packed_qkv(q, k, v)
    ref = q @ np.swapaxes(k, -1, -2)

    ctx = ExecutionContext(params={"attention_num_heads": heads})
    cfg = BertBoltAttentionQkMatMulConfig(ell=37, scale=12, nthreads=2, port=None)
    out = run_bert_bolt_attention_qk_matmul_mpc([packed], ctx=ctx, cfg=cfg)
    mae = float(np.mean(np.abs(out - ref)))

    assert out.shape == (bsz, heads, seq, seq), f"{name}: shape mismatch {out.shape}"
    assert np.isfinite(out).all(), f"{name}: non-finite output"
    assert mae < 1.0, f"{name}: MAE too high {mae}"
    print(f"[wrapper-test] {name}: shape_ok=True mae={mae:.6f}")


def run_presplit_check() -> None:
    bsz, heads, seq, head_dim = 1, 2, 4, 8
    q = np.random.standard_normal((bsz, heads, seq, head_dim))
    k = np.random.standard_normal((bsz, heads, seq, head_dim))
    ref = q @ np.swapaxes(k, -1, -2)
    ctx = ExecutionContext(params={"attention_num_heads": heads})
    out = run_bert_bolt_attention_qk_matmul_mpc([q, k], ctx=ctx, cfg=BertBoltAttentionQkMatMulConfig(port=None))
    mae = float(np.mean(np.abs(out - ref)))
    assert out.shape == ref.shape
    assert mae < 1.0, f"presplit_check: MAE too high {mae}"
    print(f"[wrapper-test] presplit_qk: shape_ok=True mae={mae:.6f}")


def main() -> None:
    run_case("small_sanity", 1, 1, 2, 4)
    run_case("multi_head", 1, 4, 8, 16)
    run_case("multi_batch", 4, 2, 8, 16)
    run_presplit_check()


if __name__ == "__main__":
    main()

