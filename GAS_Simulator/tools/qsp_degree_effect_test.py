from __future__ import annotations

import argparse
import os
import sys
import numpy as np


# src を import 可能にする
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_THIS_DIR)
_SRC_DIR = os.path.join(_ROOT_DIR, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from engines.qsp_nocircuit import QSPNoCircuitEngine
from utils.math_tools import evaluate_obj


def _make_psi_ref_uniform_key_anc0(n_key: int) -> np.ndarray:
    """
    QSPのstate_prepが key にだけHをかけ，ancillaは|0>のままなので
    psi_ref は
      (1/sqrt(2^n)) * sum_z |z> ⊗ |0>
    になる
    """
    n = 1 << int(n_key)
    psi = np.zeros(2 * n, dtype=np.complex128)
    psi[:n] = 1.0 / np.sqrt(float(n))
    return psi


def _good_mask(obj_fun_str: str, n_key: int, var_type: str, threshold: float) -> np.ndarray:
    """
    good[z] = (f(z) < threshold)
    bitstringは x0が先頭になるように [::-1] を使う
    """
    n = 1 << int(n_key)
    good = np.zeros(n, dtype=bool)
    for idx in range(n):
        bitstr = format(idx, f"0{n_key}b")[::-1]
        val = evaluate_obj(obj_fun_str, bitstr, var_type=var_type)
        good[idx] = (float(val) < float(threshold))
    return good


def _prob_key_from_state(state: np.ndarray, n_key: int) -> np.ndarray:
    """
    ancillaを周辺化して p(z)=|psi(z,0)|^2+|psi(z,1)|^2 を返す
    """
    n = 1 << int(n_key)
    a0 = state[:n]
    a1 = state[n:]
    p = (np.abs(a0) ** 2 + np.abs(a1) ** 2).astype(np.float64)
    s = float(p.sum())
    if s > 0.0:
        p /= s
    return p


def run_one(
    n_key: int,
    obj_fun_str: str,
    var_type: str,
    degree: int,
    threshold: float,
    rotation_count: int,
    seed: int,
) -> dict:
    psi_ref = _make_psi_ref_uniform_key_anc0(n_key)

    eng = QSPNoCircuitEngine(
        n_key=int(n_key),
        obj_fun_str=str(obj_fun_str),
        qsp_degree=int(degree),
        var_type=str(var_type),
        psi_ref=psi_ref,
        seed=int(seed),
    )

    good = _good_mask(obj_fun_str, n_key, var_type, threshold)

    # 初期分布での良い集合確率
    p0 = _prob_key_from_state(psi_ref, n_key)
    P_good_0 = float(np.sum(p0[good]))

    # 1ステップ後（rotation_count回の oracle+diffuser）
    st1 = eng.run_grover_iterations(float(threshold), int(rotation_count), initial_state=psi_ref)
    p1 = _prob_key_from_state(st1, n_key)
    P_good_1 = float(np.sum(p1[good]))

    return {
        "degree": int(eng.qsp_degree),
        "angles_len": int(eng.angles.size),
        "P_good_0": P_good_0,
        "P_good_1": P_good_1,
        "delta": (P_good_1 - P_good_0),
        "min_p1": float(np.min(p1)),
        "max_p1": float(np.max(p1)),
        "rotation": int(rotation_count),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_key", type=int, default=4)
    ap.add_argument("--objective", type=str, default="x0 + x1 + x2 + x3")
    ap.add_argument("--var_type", type=str, default="spin", choices=["spin", "binary"])
    ap.add_argument("--threshold", type=float, default=0.0)
    ap.add_argument("--rotation", type=int, default=1)
    ap.add_argument("--rotation_max", type=int, default=None)   # これが追加された行
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--degrees", type=int, nargs="+", default=[9, 19, 39, 79])

    args = ap.parse_args()   # 必ずここが先

    print("=== QSP degree effect test (nocircuit) ===")
    print(f"n_key={args.n_key}, var_type={args.var_type}")
    print(f"objective={args.objective}")
    if args.rotation_max is None:
        print(f"threshold={args.threshold}, rotation={args.rotation}")
    else:
        print(f"threshold={args.threshold}, rotation=1..{args.rotation_max}")
    print(f"degrees={args.degrees}")
    print("")


    rows = []
    for d in args.degrees:
        try:
            if args.rotation_max is None:
                rows.append(
                    run_one(
                        n_key=args.n_key,
                        obj_fun_str=args.objective,
                        var_type=args.var_type,
                        degree=int(d),
                        threshold=float(args.threshold),
                        rotation_count=int(args.rotation),
                        seed=int(args.seed),
                    )
                )
            else:
                best = None
                for r in range(1, int(args.rotation_max) + 1):
                    out = run_one(
                        n_key=args.n_key,
                        obj_fun_str=args.objective,
                        var_type=args.var_type,
                        degree=int(d),
                        threshold=float(args.threshold),
                        rotation_count=int(r),
                        seed=int(args.seed),
                    )
                    out["rotation"] = int(r)
                    if (best is None) or (out["P_good_1"] > best["P_good_1"]):
                        best = out
                rows.append(best)
        except Exception as e:
            print(f"[degree={d}] FAILED: {type(e).__name__}: {e}")
            print("")
            continue


    if not rows:
        raise SystemExit("No successful degrees")

    header = ["degree", "angles_len", "rotation", "P_good_0", "P_good_1", "delta", "min_p1", "max_p1"]
    colw = {k: max(len(k), 12) for k in header}
    for r in rows:
        for k in header:
            colw[k] = max(colw[k], len(f"{r[k]:.6g}" if isinstance(r[k], float) else str(r[k])))

    def fmt_row(r: dict) -> str:
        out = []
        for k in header:
            v = r[k]
            if isinstance(v, float):
                s = f"{v:.6g}"
            else:
                s = str(v)
            out.append(s.rjust(colw[k]))
        return " ".join(out)

    print(" ".join(k.rjust(colw[k]) for k in header))
    for r in rows:
        print(fmt_row(r))

    deltas = [r["delta"] for r in rows]
    if max(deltas) - min(deltas) < 1e-6:
        print("")
        print("deltaがdegreeに依存していない")
        print("次はrotationを増やすかthresholdを変えて再実行して，1ステップの回転が適切か確認する必要がある")


if __name__ == "__main__":
    main()
