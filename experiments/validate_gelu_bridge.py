from __future__ import annotations

import subprocess
import tempfile
import time
from pathlib import Path
import random
import socket

import numpy as np


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
    raise RuntimeError("No free contiguous port block found for standalone bridge test.")


def main() -> None:
    bridge = Path("/home/hedong/project/he_compiler/EzPC_bolt/EzPC/SCI/build/bin/BOLT_GELU_BRIDGE")
    if not bridge.exists():
        raise FileNotFoundError(f"Bridge binary not found: {bridge}")

    ell, s, nthreads, port = 37, 12, 2, choose_port_block()
    x = np.random.standard_normal((1, 2, 4))
    size = x.size
    mask = (1 << ell) - 1
    q = np.round(x.reshape(-1) * (1 << s)).astype(np.int64)
    q_u64 = (q & mask).astype(np.uint64)
    rng = np.random.default_rng(0)
    share1 = rng.integers(0, 1 << ell, size=size, dtype=np.uint64)
    share2 = (q_u64 - share1) & np.uint64(mask)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        in1, in2 = td / "p1_in.bin", td / "p2_in.bin"
        out1, out2 = td / "p1_out.bin", td / "p2_out.bin"
        share1.tofile(in1)
        share2.tofile(in2)

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
            str(s),
            "--size",
            str(size),
            "--input",
            str(in1),
            "--output",
            str(out1),
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
            str(s),
            "--size",
            str(size),
            "--input",
            str(in2),
            "--output",
            str(out2),
        ]
        p1 = subprocess.Popen(c1, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        time.sleep(0.3)
        p2 = subprocess.Popen(c2, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        o1, e1 = p1.communicate(timeout=120)
        o2, e2 = p2.communicate(timeout=120)
        print("party1_rc:", p1.returncode)
        print("party2_rc:", p2.returncode)
        print("party1_log:", o1.strip())
        print("party2_log:", o2.strip())
        if e1.strip():
            print("party1_err:", e1.strip())
        if e2.strip():
            print("party2_err:", e2.strip())
        if p1.returncode != 0 or p2.returncode != 0:
            raise RuntimeError("Bridge standalone validation failed.")

        y1 = np.fromfile(out1, dtype=np.uint64)
        y2 = np.fromfile(out2, dtype=np.uint64)
        y_u = (y1 + y2) & np.uint64(mask)
        signed = y_u.astype(np.int64)
        signed = np.where(signed >= (1 << (ell - 1)), signed - (1 << ell), signed)
        y = (signed.astype(np.float64) / (1 << s)).reshape(x.shape)
        print("recombine_success:", y.shape == x.shape)
        print("output_sample:", y.reshape(-1)[:4])


if __name__ == "__main__":
    main()
