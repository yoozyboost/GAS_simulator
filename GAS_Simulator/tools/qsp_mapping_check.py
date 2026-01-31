import argparse
import math
import numpy as np

import sys
sys.path.append("src")
from utils.math_tools import str_to_sympy


def eval_cost_spin(polydict, svec):
    v = 0.0
    for ps, k in polydict.items():
        k = float(k)
        idx = np.nonzero(ps)[0]
        if idx.size == 0:
            v += k
        else:
            prod = 1.0
            for i in idx:
                prod *= float(svec[i])
            v += k * prod
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_key", type=int, required=True)
    ap.add_argument("--obj", type=str, required=True)
    ap.add_argument("--threshold", type=float, default=0.0)
    ap.add_argument("--near", type=float, default=0.05)
    args = ap.parse_args()

    expr = str_to_sympy(args.obj)
    polydict = expr.as_poly().as_dict()

    l1 = float(sum(abs(float(k)) for k in polydict.values()))
    if l1 == 0.0:
        l1 = 1.0
    scale = math.pi / (2.0 * l1)

    a_vals = []
    for b in range(1 << args.n_key):
        bits = [(b >> i) & 1 for i in range(args.n_key)]
        svec = np.array([1.0 if bit == 0 else -1.0 for bit in bits], dtype=np.float64)
        fx = eval_cost_spin(polydict, svec)
        a = scale * (fx - float(args.threshold))
        a_vals.append(a)

    a_vals = np.array(a_vals, dtype=np.float64)
    amin = float(a_vals.min())
    amax = float(a_vals.max())

    pivot = math.pi / 2.0
    near = float(args.near)
    frac = float(np.mean(np.abs(a_vals - pivot) <= near))

    print(f"l1_norm = {l1}")
    print(f"scale   = {scale}")
    print(f"a_min   = {amin}")
    print(f"a_max   = {amax}")
    print(f"fraction(|a-pi/2|<= {near}) = {frac}")


if __name__ == "__main__":
    main()
