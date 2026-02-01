import argparse
import os
import sys
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt


def _rx_minus2phi(phi: float) -> np.ndarray:
    # Rx(-2phi) = cos(phi) I + i sin(phi) X
    c = float(np.cos(phi))
    s = float(np.sin(phi))
    return np.array([[c, 1j * s], [1j * s, c]], dtype=np.complex128)


def _diag_wz_rz(theta: float, Delta: float) -> np.ndarray:
    # Wz contributes exp(-i theta) to |0>, exp(+i theta) to |1>
    # Rz(2Delta) contributes exp(-i Delta) to |0>, exp(+i Delta) to |1>
    # combined: diag(exp(-i(theta+Delta)), exp(+i(theta+Delta)))
    a0 = np.exp(-1j * (theta + Delta))
    a1 = np.exp(+1j * (theta + Delta))
    return np.array([[a0, 0.0j], [0.0j, a1]], dtype=np.complex128)


def _apply_sdg_x(U: np.ndarray) -> np.ndarray:
    # Post-multiply by nothing, rather left-multiply by (X Sdg) applied after oracle
    # If circuit is U then applying gates after it yields U' = (X Sdg) U
    Sdg = np.array([[1.0 + 0.0j, 0.0j], [0.0j, -1.0j]], dtype=np.complex128)
    X = np.array([[0.0j, 1.0 + 0.0j], [1.0 + 0.0j, 0.0j]], dtype=np.complex128)
    return (X @ Sdg) @ U


def qsp_effective_unitary_from_angles(
    angles: list[float],
    theta: float,
    Delta: float = 0.0,
) -> np.ndarray:
    """
    qsp_nocircuit.py の _apply_one_oracle と同じ順序で，
    ancillaに作用する2×2ユニタリを構成する。

    v <- Rx(-2*phi0) v
    for i=1..d:
      v <- diag(exp(-i(theta+Delta)), exp(+i(theta+Delta))) v
      v <- Rx(-2*phii) v
    """
    d = len(angles) - 1
    U = _rx_minus2phi(float(angles[0]))
    D = _diag_wz_rz(float(theta), float(Delta))
    for i in range(1, d + 1):
        U = _rx_minus2phi(float(angles[i])) @ (D @ U)
    return U


def _observable(U: np.ndarray, name: str) -> np.ndarray:
    if name == "re_u00":
        return np.real(U[0, 0])
    if name == "im_u10":
        return np.imag(U[1, 0])
    if name == "abs_u00":
        return np.abs(U[0, 0])
    if name == "abs_u10_sq":
        return np.abs(U[1, 0]) ** 2
    if name == "abs_u00_sq":
        return np.abs(U[0, 0]) ** 2
    raise ValueError(f"unknown observable: {name}")


def _sign_target(a: np.ndarray) -> np.ndarray:
    # sign(0)=0 として描く
    y = np.ones_like(a, dtype=np.float64)
    y[a < 0] = -1.0
    y[a == 0] = 0.0
    return y


@dataclass
class MarginStats:
    w_09: float
    w_099: float
    max_err_outside_01: float
    max_err_outside_005: float


def _margin_stats(a: np.ndarray, y: np.ndarray) -> MarginStats:
    # y should approximate sign(a)
    aa = np.abs(a)

    def _w_for(level: float) -> float:
        ok = (np.abs(y) >= level)
        if not np.any(ok):
            return float("nan")
        return float(np.min(aa[ok]))

    tgt = _sign_target(a)
    err = np.abs(y - tgt)

    def _max_err_outside(eps: float) -> float:
        mask = (aa >= eps)
        if not np.any(mask):
            return float("nan")
        return float(np.max(err[mask]))

    return MarginStats(
        w_09=_w_for(0.9),
        w_099=_w_for(0.99),
        max_err_outside_01=_max_err_outside(0.1),
        max_err_outside_005=_max_err_outside(0.05),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--degrees", type=int, nargs="+", default=[9, 19, 39, 79])
    ap.add_argument("--npts", type=int, default=400)
    ap.add_argument("--delta", type=float, default=0.0)
    ap.add_argument(
        "--observable",
        type=str,
        default="re_u00",
        choices=["re_u00", "im_u10", "abs_u00", "abs_u00_sq", "abs_u10_sq"],
    )
    ap.add_argument("--apply_sdg_x", action="store_true")
    ap.add_argument("--outdir", type=str, default="tools_out")
    ap.add_argument("--save_csv", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # tools/ からでも src/ をimportできるようにする
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.append(os.path.join(root, "src"))

    from oracles.qsp_oracle import get_sign_angles_cached

    a = np.linspace(-1.0, 1.0, int(args.npts), dtype=np.float64)
    theta = np.arccos(np.clip(a, -1.0, 1.0))  # [0, pi]
    Delta = float(args.delta)

    stats_rows = []
    plt.figure()
    for d in args.degrees:
        angles = list(get_sign_angles_cached(int(d)))
        Uvals = []
        for th in theta:
            U = qsp_effective_unitary_from_angles(angles, float(th), Delta=Delta)
            if args.apply_sdg_x:
                U = _apply_sdg_x(U)
            Uvals.append(_observable(U, args.observable))
        y = np.array(Uvals, dtype=np.float64)

        st = _margin_stats(a, y)
        stats_rows.append((int(d), len(angles), st.w_09, st.w_099, st.max_err_outside_01, st.max_err_outside_005))

        plt.plot(a, y, label=f"d={int(d)}")

        if args.save_csv:
            csv_path = os.path.join(args.outdir, f"resp_d{int(d)}_{args.observable}.csv")
            np.savetxt(
                csv_path,
                np.column_stack([a, y]),
                delimiter=",",
                header="a,y",
                comments="",
            )

    plt.plot(a, _sign_target(a), linestyle="--", label="sign(a)")
    plt.xlabel("a")
    plt.ylabel(args.observable + ("  with X*Sdg" if args.apply_sdg_x else ""))
    plt.title("QSP response vs a in [-1,1]")
    plt.legend()

    png_path = os.path.join(
        args.outdir,
        f"qsp_resp_{args.observable}" + ("_withXSdg" if args.apply_sdg_x else "") + ".png",
    )
    plt.savefig(png_path, dpi=200)

    print("=== QSP oracle response check (nocircuit-consistent 2x2) ===")
    print(f"observable={args.observable}")
    print(f"apply_sdg_x={bool(args.apply_sdg_x)}")
    print(f"Delta={Delta}")
    print(f"saved: {png_path}")
    print("")
    print("degree angles_len w(|y|>=0.9) w(|y|>=0.99) max_err(|a|>=0.1) max_err(|a|>=0.05)")
    for row in stats_rows:
        d, alen, w09, w099, e01, e005 = row
        print(f"{d:6d} {alen:10d} {w09:12.6g} {w099:13.6g} {e01:17.6g} {e005:18.6g}")


if __name__ == "__main__":
    main()
