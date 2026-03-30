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

from operators.linear_ffn1.method_mpc_bolt import deterministic_ffn_linear1_params


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


def share_encode(x: np.ndarray, ell: int, s: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    mask = (1 << ell) - 1
    q = np.round(x.reshape(-1) * (1 << s)).astype(np.int64)
    q_u = (q & mask).astype(np.uint64)
    rng = np.random.default_rng(seed)
    sh1 = rng.integers(0, 1 << ell, size=q_u.size, dtype=np.uint64)
    sh2 = (q_u - sh1) & np.uint64(mask)
    return sh1, sh2


def decode(sh1: np.ndarray, sh2: np.ndarray, ell: int, s: int, shape: tuple[int, ...]) -> np.ndarray:
    mask = np.uint64((1 << ell) - 1)
    c = (sh1 + sh2) & mask
    signed = c.astype(np.int64)
    signed = np.where(signed >= (1 << (ell - 1)), signed - (1 << ell), signed)
    return (signed.astype(np.float64) / (1 << s)).reshape(shape)


def main() -> None:
    bridge = Path("/home/hedong/project/he_compiler/EzPC_bolt/EzPC/SCI/build/bin/BOLT_FFN_LINEAR1_BRIDGE")
    if not bridge.exists():
        raise FileNotFoundError(f"Bridge binary not found: {bridge}")

    ell, s, nthreads = 37, 12, 2
    bsz, seq, h, out_dim = 1, 2, 4, 8
    n = bsz * seq
    x = np.random.standard_normal((bsz, seq, h))
    w, b = deterministic_ffn_linear1_params(h, out_dim, seed=1234)
    w_rep = np.broadcast_to(w.reshape(1, h, out_dim), (n, h, out_dim)).copy()
    b_rep = np.broadcast_to(b.reshape(1, out_dim), (n, out_dim)).copy()

    x1, x2 = share_encode(x.reshape(n, h), ell, s, seed=61)
    w1, w2 = share_encode(w_rep, ell, s, seed=67)
    b1, b2 = share_encode(b_rep, ell, s, seed=71)
    for attempt in range(1, 6):
        port = choose_port_block()
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            p1_in, p2_in = td / "p1_in.bin", td / "p2_in.bin"
            p1_w, p2_w = td / "p1_w.bin", td / "p2_w.bin"
            p1_b, p2_b = td / "p1_b.bin", td / "p2_b.bin"
            p1_out, p2_out = td / "p1_out.bin", td / "p2_out.bin"
            x1.tofile(p1_in); x2.tofile(p2_in)
            w1.tofile(p1_w); w2.tofile(p2_w)
            b1.tofile(p1_b); b2.tofile(p2_b)

            c1 = [
                str(bridge), "--party", "1", "--port", str(port), "--address", "127.0.0.1",
                "--nthreads", str(nthreads), "--ell", str(ell), "--scale", str(s),
                "--n", str(n), "--h", str(h), "--i", str(out_dim),
                "--input", str(p1_in), "--weight", str(p1_w), "--bias", str(p1_b), "--output", str(p1_out),
            ]
            c2 = [
                str(bridge), "--party", "2", "--port", str(port), "--address", "127.0.0.1",
                "--nthreads", str(nthreads), "--ell", str(ell), "--scale", str(s),
                "--n", str(n), "--h", str(h), "--i", str(out_dim),
                "--input", str(p2_in), "--weight", str(p2_w), "--bias", str(p2_b), "--output", str(p2_out),
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
                print(f"attempt={attempt} timeout; retrying")
                continue
            print("attempt:", attempt)
            print("party1_rc:", p1.returncode)
            print("party2_rc:", p2.returncode)
            print("party1_log:", o1.strip())
            print("party2_log:", o2.strip())
            if e1.strip():
                print("party1_err:", e1.strip())
            if e2.strip():
                print("party2_err:", e2.strip())
            if p1.returncode != 0 or p2.returncode != 0:
                if "Address already in use" in (e1 + e2):
                    print("bind conflict; retrying")
                    continue
                raise RuntimeError("FFN_Linear_1 bridge standalone validation failed.")

            y1 = np.fromfile(p1_out, dtype=np.uint64)
            y2 = np.fromfile(p2_out, dtype=np.uint64)
            y = decode(y1, y2, ell, s, (bsz, seq, out_dim))
            print("recombine_success:", y.shape == (bsz, seq, out_dim))
            print("output_sample:", y.reshape(-1)[:6])
            return
    raise RuntimeError("FFN_Linear_1 bridge standalone validation failed after retries.")


if __name__ == "__main__":
    main()
