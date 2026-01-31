"""QSP の符号関数近似を可視化する検証スクリプト。

目的
  ・get_sign_angles_cached(degree, delta) で得た位相角列が、
    期待どおりに符号関数を近似しているかを確認する
  ・Im<U10> と Re<U00> のどちらに符号近似が現れるかを確認する
  ・H を前後で挟むことで位相オラクル化できるかを確認する

実行
  python tools/qsp_response_check.py --degree 79 --delta 20
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import matplotlib.pyplot as plt


# src を import できるようにする
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_THIS_DIR)
_SRC_DIR = os.path.join(_ROOT_DIR, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from oracles.qsp_oracle import get_sign_angles_cached  # noqa: E402


def rx(phi: float) -> np.ndarray:
    c = math.cos(phi)
    s = math.sin(phi)
    return np.array([[c, 1j * s], [1j * s, c]], dtype=np.complex128)


def wz(a: float) -> np.ndarray:
    return np.array([[np.exp(-1j * a), 0.0], [0.0, np.exp(+1j * a)]], dtype=np.complex128)


def h_gate() -> np.ndarray:
    inv = 1.0 / math.sqrt(2.0)
    return np.array([[inv, inv], [inv, -inv]], dtype=np.complex128)


def qsp_unitary(a: float, angles: list[float], wrap_h: bool) -> np.ndarray:
    U = np.eye(2, dtype=np.complex128)

    if wrap_h:
        H = h_gate()
        U = H @ U

    U = rx(-float(angles[0])) @ U
    W = wz(float(a))

    for k in range(1, len(angles)):
        U = W @ U
        U = rx(-float(angles[k])) @ U

    if wrap_h:
        H = h_gate()
        U = H @ U

    return U


def sign(x: np.ndarray) -> np.ndarray:
    y = np.zeros_like(x, dtype=np.float64)
    y[x > 0] = 1.0
    y[x < 0] = -1.0
    return y


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--degree", type=int, default=79)
    ap.add_argument("--delta", type=float, default=20.0)
    ap.add_argument("--npts", type=int, default=2000)
    ap.add_argument("--amin", type=float, default=-math.pi / 2)
    ap.add_argument("--amax", type=float, default=+math.pi / 2)
    ap.add_argument("--no-wrap-h", action="store_true")
    args = ap.parse_args()

    angles = get_sign_angles_cached(args.degree, delta=args.delta)

    a = np.linspace(float(args.amin), float(args.amax), int(args.npts))

    y_u10_im = np.empty_like(a, dtype=np.float64)
    y_u00_re = np.empty_like(a, dtype=np.float64)
    y_u00_re_wrapped = np.empty_like(a, dtype=np.float64)

    wrap_h = (not args.no_wrap_h)

    for i, ai in enumerate(a):
        U_raw = qsp_unitary(float(ai), angles, wrap_h=False)
        y_u10_im[i] = float(np.imag(U_raw[1, 0]))
        y_u00_re[i] = float(np.real(U_raw[0, 0]))

        U_wrapped = qsp_unitary(float(ai), angles, wrap_h=wrap_h)
        y_u00_re_wrapped[i] = float(np.real(U_wrapped[0, 0]))

    target_sin = sign(np.sin(a))
    target_a = sign(a)

    plt.figure()
    plt.plot(a, target_sin, label="sign(sin a)")
    plt.plot(a, target_a, label="sign(a)")
    plt.plot(a, y_u10_im, label="Im<U10> (raw)")
    plt.plot(a, y_u00_re, label="Re<U00> (raw)")
    plt.plot(a, y_u00_re_wrapped, label=f"Re<U00> (H-wrap={wrap_h})")
    plt.xlabel("a")
    plt.ylabel("response")
    plt.title(f"QSP sign response (degree={args.degree}, delta={args.delta})")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
