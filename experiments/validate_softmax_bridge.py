from __future__ import annotations

import random
import socket
import subprocess
import tempfile
import time
from pathlib import Path

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
    raise RuntimeError("No free contiguous port block found.")


def main() -> None:
    bridge = Path("/home/hedong/project/he_compiler/EzPC_bolt/EzPC/SCI/build/bin/BOLT_SOFTMAX_BRIDGE")
    if not bridge.exists():
        raise FileNotFoundError(f"Softmax bridge binary not found: {bridge}")

    ell, s, nthreads = 37, 12, 2
    x = np.random.standard_normal((2, 4))  # dim=2 rows, array_size=4
    dim, array_size = x.shape
    size = dim * array_size
    mask = (1 << ell) - 1
    q = np.round(x.reshape(-1) * (1 << s)).astype(np.int64)
    q_u64 = (q & mask).astype(np.uint64)
    rng = np.random.default_rng(0)
    share1 = rng.integers(0, 1 << ell, size=size, dtype=np.uint64)
    share2 = (q_u64 - share1) & np.uint64(mask)
    port = choose_port_block()

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        in1, in2 = td / "p1_in.bin", td / "p2_in.bin"
        out1, out2 = td / "p1_out.bin", td / "p2_out.bin"
        l1, l2 = td / "p1_l.bin", td / "p2_l.bin"
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
            "--dim",
            str(dim),
            "--array_size",
            str(array_size),
            "--input",
            str(in1),
            "--output",
            str(out1),
            "--l_out",
            str(l1),
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
            "--dim",
            str(dim),
            "--array_size",
            str(array_size),
            "--input",
            str(in2),
            "--output",
            str(out2),
            "--l_out",
            str(l2),
        ]
        p1 = subprocess.Popen(c1, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        time.sleep(0.3)
        p2 = subprocess.Popen(c2, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        o1, e1 = p1.communicate(timeout=180)
        o2, e2 = p2.communicate(timeout=180)
        print("party1_rc:", p1.returncode)
        print("party2_rc:", p2.returncode)
        print("party1_log:", o1.strip())
        print("party2_log:", o2.strip())
        if e1.strip():
            print("party1_err:", e1.strip())
        if e2.strip():
            print("party2_err:", e2.strip())
        if p1.returncode != 0 or p2.returncode != 0:
            raise RuntimeError("Softmax bridge standalone validation failed.")

        y1 = np.fromfile(out1, dtype=np.uint64)
        y2 = np.fromfile(out2, dtype=np.uint64)
        l_1 = np.fromfile(l1, dtype=np.uint64)
        l_2 = np.fromfile(l2, dtype=np.uint64)

        y_u = (y1 + y2) & np.uint64(mask)
        l_u = (l_1 + l_2) & np.uint64(mask)

        y_signed = y_u.astype(np.int64)
        l_signed = l_u.astype(np.int64)
        cut = 1 << (ell - 1)
        y_signed = np.where(y_signed >= cut, y_signed - (1 << ell), y_signed)
        l_signed = np.where(l_signed >= cut, l_signed - (1 << ell), l_signed)
        y = (y_signed.astype(np.float64) / (1 << s)).reshape(dim, array_size)
        l_val = (l_signed.astype(np.float64) / (1 << s)).reshape(dim, array_size)
        print("recombine_output_success:", y.shape == (dim, array_size))
        print("recombine_l_success:", l_val.shape == (dim, array_size))
        print("output_sample:", y.reshape(-1)[:4])
        print("l_sample:", l_val.reshape(-1)[:4])


if __name__ == "__main__":
    main()
