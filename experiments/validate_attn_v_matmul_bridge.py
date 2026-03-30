from __future__ import annotations

import random
import socket
import subprocess
import tempfile
import time
from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))


def port_available(port: int) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def choose_port_block(block_size: int = 64, trials: int = 200) -> int:
    for _ in range(trials):
        start = random.randint(22000, 50000 - block_size - 1)
        if all(port_available(start + i) for i in range(block_size)):
            return start
    raise RuntimeError("No free contiguous port block found.")


def share_encode(x: np.ndarray, ell: int, scale: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    mask = (1 << ell) - 1
    q = np.round(x.reshape(-1) * (1 << scale)).astype(np.int64)
    q_u = (q & mask).astype(np.uint64)
    rng = np.random.default_rng(seed)
    sh0 = rng.integers(0, 1 << ell, size=q_u.size, dtype=np.uint64)
    sh1 = (q_u - sh0) & np.uint64(mask)
    return sh0, sh1


def decode(sh0: np.ndarray, sh1: np.ndarray, ell: int, scale: int, shape: tuple[int, ...]) -> np.ndarray:
    mask = np.uint64((1 << ell) - 1)
    c = (sh0 + sh1) & mask
    signed = c.astype(np.int64)
    signed = np.where(signed >= (1 << (ell - 1)), signed - (1 << ell), signed)
    return (signed.astype(np.float64) / float(1 << scale)).reshape(shape)


def random_attn(bsz: int, heads: int, seq: int) -> np.ndarray:
    logits = np.random.standard_normal((bsz, heads, seq, seq))
    logits = logits - logits.max(axis=-1, keepdims=True)
    ex = np.exp(logits)
    return ex / ex.sum(axis=-1, keepdims=True)


def run_case(name: str, bsz: int, heads: int, seq: int, head_dim: int, bridge: Path) -> None:
    ell, scale, nthreads = 37, 12, 2
    attn = random_attn(bsz, heads, seq)
    v = np.random.standard_normal((bsz, heads, seq, head_dim))

    a = attn.reshape(bsz * heads, seq, seq)
    b = v.reshape(bsz * heads, seq, head_dim)
    ref = attn @ v

    a0, a1 = share_encode(a, ell, scale, seed=149)
    b0, b1 = share_encode(b, ell, scale, seed=151)

    n = bsz * heads
    dim1, dim2, dim3 = seq, seq, head_dim
    for attempt in range(1, 6):
        port = choose_port_block()
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            p1_a, p2_a = td_path / "p1_a.bin", td_path / "p2_a.bin"
            p1_b, p2_b = td_path / "p1_b.bin", td_path / "p2_b.bin"
            p1_out, p2_out = td_path / "p1_out.bin", td_path / "p2_out.bin"
            a0.tofile(p1_a)
            a1.tofile(p2_a)
            b0.tofile(p1_b)
            b1.tofile(p2_b)

            c1 = [
                str(bridge),
                "--party",
                "1",
                "--port",
                str(port),
                "--address",
                "127.0.0.1",
                "--nthreads",
                str(nthreads),
                "--ell",
                str(ell),
                "--scale",
                str(scale),
                "--n",
                str(n),
                "--dim1",
                str(dim1),
                "--dim2",
                str(dim2),
                "--dim3",
                str(dim3),
                "--input_a",
                str(p1_a),
                "--input_b",
                str(p1_b),
                "--output",
                str(p1_out),
            ]
            c2 = [
                str(bridge),
                "--party",
                "2",
                "--port",
                str(port),
                "--address",
                "127.0.0.1",
                "--nthreads",
                str(nthreads),
                "--ell",
                str(ell),
                "--scale",
                str(scale),
                "--n",
                str(n),
                "--dim1",
                str(dim1),
                "--dim2",
                str(dim2),
                "--dim3",
                str(dim3),
                "--input_a",
                str(p2_a),
                "--input_b",
                str(p2_b),
                "--output",
                str(p2_out),
            ]
            p1 = subprocess.Popen(c1, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            time.sleep(0.6)
            p2 = subprocess.Popen(c2, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                o1, e1 = p1.communicate(timeout=180)
                o2, e2 = p2.communicate(timeout=180)
            except subprocess.TimeoutExpired:
                p1.kill()
                p2.kill()
                print(f"[bridge-test] {name}: attempt={attempt} timeout_retry=True")
                continue
            if p1.returncode != 0 or p2.returncode != 0:
                if "address already in use" in (e1 + e2).lower():
                    print(f"[bridge-test] {name}: attempt={attempt} bind_conflict_retry=True")
                    continue
                raise RuntimeError(
                    f"{name}: bridge failed rc=({p1.returncode},{p2.returncode})\n"
                    f"party1_stdout={o1}\nparty1_stderr={e1}\nparty2_stdout={o2}\nparty2_stderr={e2}"
                )

            y0 = np.fromfile(p1_out, dtype=np.uint64)
            y1 = np.fromfile(p2_out, dtype=np.uint64)
            out = decode(y0, y1, ell, scale, (bsz, heads, seq, head_dim))
            mae = float(np.mean(np.abs(out - ref)))
            assert out.shape == (bsz, heads, seq, head_dim), f"{name}: shape mismatch {out.shape}"
            assert np.isfinite(out).all(), f"{name}: non-finite output"
            assert mae < 1.0, f"{name}: MAE too high {mae}"
            print(f"[bridge-test] {name}: shape_ok=True mae={mae:.6f}")
            return
    raise RuntimeError(f"{name}: bridge validation failed after retries")


def main() -> None:
    bridge = Path("/home/hedong/project/he_compiler/EzPC_bolt/EzPC/SCI/build/bin/BOLT_ATTN_V_MATMUL_MPC_BRIDGE")
    if not bridge.exists():
        raise FileNotFoundError(f"Bridge binary not found: {bridge}")
    run_case("small_sanity", 1, 1, 2, 4, bridge)
    run_case("multi_head", 1, 4, 8, 16, bridge)
    run_case("multi_batch", 4, 2, 8, 16, bridge)


if __name__ == "__main__":
    main()

