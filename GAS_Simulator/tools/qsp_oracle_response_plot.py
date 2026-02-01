import argparse
import math
import cmath
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("src")
from oracles.qsp_oracle import get_sign_angles_cached


def Rx(beta: float) -> np.ndarray:
    c = math.cos(beta / 2.0)
    s = math.sin(beta / 2.0)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)


def Rz(beta: float) -> np.ndarray:
    return np.array(
        [[cmath.exp(-1j * beta / 2.0), 0.0], [0.0, cmath.exp(1j * beta / 2.0)]],
        dtype=np.complex128,
    )

def X() -> np.ndarray:
    return np.array([[0, 1], [1, 0]], dtype=np.complex128)

def S_dagger() -> np.ndarray:
    return np.array([[1, 0], [0, -1j]], dtype=np.complex128)

def qsp_unitary(a: float, angles: list[float]) -> np.ndarray:
    """
    回路の縮約モデル。
    ancilla上の信号演算子が exp(-i a Z) であるとき、回路は
      U = Rx(-2φ0) Π_{k=1..d} [ Rz(2a) Rx(-2φk) ]
    に縮約される。
    """
    U = np.eye(2, dtype=np.complex128)
    U = Rx(-2.0 * angles[0]) @ U
    for phi in angles[1:]:
        U = Rz(2.0 * a) @ U
        U = Rx(-2.0 * phi) @ U

    U = S_dagger() @ U
    U = X() @ U
    return U


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--degrees", nargs="+", type=int, default=[9, 21, 39, 79])
    ap.add_argument("--num", type=int, default=2001)
    ap.add_argument("--amin", type=float, default=-1)
    ap.add_argument("--amax", type=float, default=1)
    ap.add_argument("--outfile", type=str, default="qsp_oracle_response.png")
    args = ap.parse_args()

    x_grid = np.linspace(args.amin, args.amax, args.num)
    theta_grid = np.arccos(np.clip(x_grid, -1.0, 1.0))


    plt.figure()
    plt.plot(x_grid, np.sign(x_grid), label="sign(x)")


    for d in args.degrees:
        angles = get_sign_angles_cached(d)
        y_re = []
        y_im = []
        y_abs = []
        y_leak = []
        y_arg = []
        y = []
        for th in theta_grid:
            U = qsp_unitary(float(th), angles)
            y.append(float(np.real(U[0, 0])))
        plt.plot(x_grid, y, label=f"Re<U00>, d={len(angles)-1}")

        # plt.plot(a_grid, y, label=f"Re<U00>, d={len(angles)-1}")
        # plt.plot(a_grid, y_re, label=f"Re<U00>, d={d}")
        # plt.plot(a_grid, y_im, label=f"Im<U00>, d={d}")
        # plt.plot(a_grid, y_abs, label=f"|U00|, d={d}")
        # plt.plot(a_grid, y_leak, label=f"Leak<U10>, d={d}")
        # plt.plot(a_grid, y_arg, label=f"Arg<U00>, d={d}")

    plt.ylim(-1.2, 1.2)
    plt.xlabel("x")
    plt.ylabel("response")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.outfile, dpi=200)
    print(args.outfile)


if __name__ == "__main__":
    main()
