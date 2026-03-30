from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from operators.attention_v_matmul.method_mpc import (  # noqa: E402
    BertBoltAttentionVMatMulConfig,
    run_bert_bolt_attention_v_matmul_mpc,
)
from runtime.types import ExecutionContext  # noqa: E402


def make_packed_qkv(v: np.ndarray) -> np.ndarray:
    bsz, heads, seq, head_dim = v.shape
    hidden = heads * head_dim
    zero = np.zeros((bsz, seq, hidden), dtype=np.float64)
    v3 = v.transpose(0, 2, 1, 3).reshape(bsz, seq, hidden)
    return np.stack([zero, zero, v3], axis=0)


def random_attn(bsz: int, heads: int, seq: int) -> np.ndarray:
    logits = np.random.standard_normal((bsz, heads, seq, seq))
    logits = logits - logits.max(axis=-1, keepdims=True)
    ex = np.exp(logits)
    return ex / ex.sum(axis=-1, keepdims=True)


def run_case(name: str, bsz: int, heads: int, seq: int, head_dim: int) -> None:
    attn = random_attn(bsz, heads, seq)
    v = np.random.standard_normal((bsz, heads, seq, head_dim))
    packed = make_packed_qkv(v)
    ref_canonical = attn @ v
    ref_flat = ref_canonical.transpose(0, 2, 1, 3).reshape(bsz, seq, heads * head_dim)

    cfg = BertBoltAttentionVMatMulConfig(ell=37, scale=12, nthreads=2, port=None)
    ctx_flat = ExecutionContext(params={"attention_num_heads": heads, "attention_return_canonical": False})
    out_flat = run_bert_bolt_attention_v_matmul_mpc([attn, packed], ctx=ctx_flat, cfg=cfg)
    mae_flat = float(np.mean(np.abs(out_flat - ref_flat)))
    assert out_flat.shape == (bsz, seq, heads * head_dim), f"{name}: flat shape mismatch {out_flat.shape}"
    assert np.isfinite(out_flat).all(), f"{name}: non-finite flat output"
    assert mae_flat < 1.0, f"{name}: flat MAE too high {mae_flat}"

    ctx_canonical = ExecutionContext(params={"attention_num_heads": heads, "attention_return_canonical": True})
    out_canonical = run_bert_bolt_attention_v_matmul_mpc([attn, packed], ctx=ctx_canonical, cfg=cfg)
    mae_canonical = float(np.mean(np.abs(out_canonical - ref_canonical)))
    assert out_canonical.shape == (bsz, heads, seq, head_dim), (
        f"{name}: canonical shape mismatch {out_canonical.shape}"
    )
    assert np.isfinite(out_canonical).all(), f"{name}: non-finite canonical output"
    assert mae_canonical < 1.0, f"{name}: canonical MAE too high {mae_canonical}"
    print(
        f"[wrapper-test] {name}: flat_shape_ok=True canonical_shape_ok=True "
        f"mae_flat={mae_flat:.6f} mae_canonical={mae_canonical:.6f}"
    )


def run_presplit_check() -> None:
    bsz, heads, seq, head_dim = 1, 2, 4, 8
    attn = random_attn(bsz, heads, seq)
    v = np.random.standard_normal((bsz, heads, seq, head_dim))
    ref = attn @ v
    out = run_bert_bolt_attention_v_matmul_mpc(
        [attn, v],
        ctx=ExecutionContext(params={"attention_num_heads": heads, "attention_return_canonical": True}),
        cfg=BertBoltAttentionVMatMulConfig(port=None),
    )
    mae = float(np.mean(np.abs(out - ref)))
    assert out.shape == ref.shape
    assert mae < 1.0, f"presplit_check: MAE too high {mae}"
    print(f"[wrapper-test] presplit_v: shape_ok=True mae={mae:.6f}")


def main() -> None:
    run_case("small_sanity", 1, 1, 2, 4)
    run_case("multi_head", 1, 4, 8, 16)
    run_case("multi_batch", 4, 2, 8, 16)
    run_presplit_check()


if __name__ == "__main__":
    main()

