from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# diagnostics/ から実行しても src/ が import できるようにする。
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_THIS_DIR)
_SRC_DIR = os.path.join(_ROOT_DIR, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from utils.math_tools import evaluate_obj
from oracles.qd_oracle import QDOracleBuilder
from oracles.qsp_oracle import QSPOracleBuilder
from state_prep import get_state_prep_method


def _int_to_bitstring_x0_first(i: int, n: int) -> str:
    return format(int(i), f"0{n}b")[::-1]


def _signed_twos_complement(bits_msb_to_lsb: List[int]) -> int:
    n = len(bits_msb_to_lsb)
    u = 0
    for b in bits_msb_to_lsb:
        u = (u << 1) | int(b)
    if n == 0:
        return 0
    if bits_msb_to_lsb[0] == 0:
        return int(u)
    return int(u - (1 << n))


def _extract_bits_msb_to_lsb_from_index(index: int, qubits_msb_to_lsb: List[int]) -> List[int]:
    return [((index >> q) & 1) for q in qubits_msb_to_lsb]


@dataclass
class PerXResult:
    x_int: int
    bitstring: str
    f_value: float
    good: int
    p_mark: float
    decoded_val: Optional[int] = None
    decoded_val_prob: Optional[float] = None


def _accuracy_from_scores(good: np.ndarray, score: np.ndarray, convention: str) -> float:
    """
    convention:
      - "mark_is_good": score>=0.5 を good と判定
      - "mark_is_bad" : score< 0.5 を good と判定
    """
    if convention == "mark_is_good":
        pred = (score >= 0.5).astype(int)
    elif convention == "mark_is_bad":
        pred = (score < 0.5).astype(int)
    else:
        raise ValueError(convention)
    return float(np.mean((pred == good).astype(float)))


def _print_summary(title: str, results: List[PerXResult]) -> None:
    good = np.array([r.good for r in results], dtype=int)
    score = np.array([r.p_mark for r in results], dtype=float)

    acc_good = _accuracy_from_scores(good, score, "mark_is_good")
    acc_bad = _accuracy_from_scores(good, score, "mark_is_bad")

    if acc_good >= acc_bad:
        best = "mark_is_good"
        best_acc = acc_good
    else:
        best = "mark_is_bad"
        best_acc = acc_bad

    print("")
    print(f"== {title} ==")
    print(f"accuracy(mark=good) = {acc_good:.6f}")
    print(f"accuracy(mark=bad ) = {acc_bad:.6f}")
    print(f"best_convention      = {best}")
    print(f"best_accuracy        = {best_acc:.6f}")


def run_qsp_oracle_check(
    n_key: int,
    obj: str,
    var_type: str,
    threshold: float,
    qsp_degree: int,
    show_rows: int,
) -> List[PerXResult]:
    """
    QSP oracle 単体の検証。
    各 x について |x>|0> に oracle を適用し、P(anc=1) を測る。
    """
    bld = QSPOracleBuilder()
    qc_oracle = bld.build_oracle(
        n_key=n_key,
        obj_fun_str=obj,
        threshold=float(threshold),
        qsp_degree=int(qsp_degree),
    )

    anc = n_key
    results: List[PerXResult] = []

    for x in range(1 << n_key):
        bitstr = _int_to_bitstring_x0_first(x, n_key)
        f = evaluate_obj(obj, bitstr, var_type=var_type)
        good = 1 if (f < threshold) else 0

        qc = QuantumCircuit(n_key + 1)
        # |x> を準備。bitstr は x0 が先頭なので qubit i が x_i。
        for i in range(n_key):
            if bitstr[i] == "1":
                qc.x(i)
        qc.compose(qc_oracle, inplace=True)

        sv = Statevector.from_instruction(qc)
        probs = np.abs(sv.data) ** 2

        p1 = 0.0
        for idx, p in enumerate(probs):
            if ((idx >> anc) & 1) == 1:
                p1 += float(p)

        results.append(
            PerXResult(
                x_int=x,
                bitstring=bitstr,
                f_value=float(f),
                good=int(good),
                p_mark=float(p1),
            )
        )

    results.sort(key=lambda r: r.x_int)

    _print_summary("QSP oracle semantics", results)

    if show_rows > 0:
        print("")
        print("x_int  bitstring  f(x)        good  P(anc=1|x)")
        for r in results[:show_rows]:
            print(f"{r.x_int:5d}  {r.bitstring}   {r.f_value:10.6f}   {r.good:4d}  {r.p_mark:12.8f}")

    return results


def run_qd_stateprep_sign_check(
    n_key: int,
    n_val: int,
    obj: str,
    var_type: str,
    threshold: float,
    initial_state: str,
    state_prep_params: Dict,
    show_rows: int,
    decode_value: bool,
) -> List[PerXResult]:
    """
    QD の state_prep(Ay) で、value 符号ビットが 1 になるかを検証。
    oracle 自体は Z を当てるだけなので、mark 集合は state_prep が作る符号で決まる。
    """
    bld = QDOracleBuilder()
    sp = get_state_prep_method(initial_state)

    qc_sp = bld.build_state_prep(
        n_key=n_key,
        obj_fun_str=obj,
        state_prep_method=sp,
        n_val=int(n_val),
        is_spin=(var_type == "spin"),
        threshold=float(threshold),
        **(state_prep_params or {}),
    )

    sv = Statevector.from_instruction(qc_sp)
    amps = sv.data
    probs = np.abs(amps) ** 2

    sign_qubit = n_key  # 実装コメント上の符号ビット
    key_mask = (1 << n_key) - 1

    # value bits の並びは実装コメントに合わせる。
    # v=0..n_val-1 が MSB->LSB のつもりで扱う。
    val_qubits_msb_to_lsb = [n_key + v for v in range(n_val)]

    # 条件付きで P(sign=1|x) を求める。
    p_key = np.zeros(1 << n_key, dtype=np.float64)
    p_sign1_key = np.zeros(1 << n_key, dtype=np.float64)

    # decode_value をする場合は、各 x について value の最頻ビット列も取る。
    # ここでは P(val=v | x) を全 v 走査して argmax を取る。
    if decode_value:
        p_val_key = np.zeros(((1 << n_key), (1 << n_val)), dtype=np.float64)
    else:
        p_val_key = None

    for idx, p in enumerate(probs):
        key = idx & key_mask
        p_key[key] += float(p)

        sign = (idx >> sign_qubit) & 1
        if sign == 1:
            p_sign1_key[key] += float(p)

        if decode_value:
            # value integer を val_qubits_msb_to_lsb で読む。
            bits = _extract_bits_msb_to_lsb_from_index(idx, val_qubits_msb_to_lsb)
            u = 0
            for b in bits:
                u = (u << 1) | int(b)
            p_val_key[key, u] += float(p)

    results: List[PerXResult] = []
    for x in range(1 << n_key):
        bitstr = _int_to_bitstring_x0_first(x, n_key)
        f = evaluate_obj(obj, bitstr, var_type=var_type)

        # QD は f(x) - threshold < 0 を negative とみなして符号ビットが立つ設計。
        good = 1 if (f < threshold) else 0

        if p_key[x] <= 0.0:
            p1 = 0.0
        else:
            p1 = float(p_sign1_key[x] / p_key[x])

        decoded_val = None
        decoded_prob = None
        if decode_value and p_val_key is not None and p_key[x] > 0.0:
            v_u = int(np.argmax(p_val_key[x]))
            bits = [(v_u >> (n_val - 1 - j)) & 1 for j in range(n_val)]
            decoded_val = _signed_twos_complement(bits)
            decoded_prob = float(p_val_key[x, v_u] / p_key[x])

        results.append(
            PerXResult(
                x_int=x,
                bitstring=bitstr,
                f_value=float(f),
                good=int(good),
                p_mark=float(p1),
                decoded_val=decoded_val,
                decoded_val_prob=decoded_prob,
            )
        )

    results.sort(key=lambda r: r.x_int)

    _print_summary("QD state_prep sign semantics", results)

    if show_rows > 0:
        print("")
        if decode_value:
            print("x_int  bitstring  f(x)        good  P(sign=1|x)   val_mode  P(val_mode|x)")
            for r in results[:show_rows]:
                dv = "None" if r.decoded_val is None else str(r.decoded_val)
                dp = 0.0 if r.decoded_val_prob is None else float(r.decoded_val_prob)
                print(f"{r.x_int:5d}  {r.bitstring}   {r.f_value:10.6f}   {r.good:4d}  {r.p_mark:12.8f}  {dv:>7s}  {dp:12.8f}")
        else:
            print("x_int  bitstring  f(x)        good  P(sign=1|x)")
            for r in results[:show_rows]:
                print(f"{r.x_int:5d}  {r.bitstring}   {r.f_value:10.6f}   {r.good:4d}  {r.p_mark:12.8f}")

    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", type=str, required=True, choices=["qd", "qsp"])
    ap.add_argument("--var_type", type=str, required=True, choices=["binary", "spin"])
    ap.add_argument("--n_key", type=int, required=True)
    ap.add_argument("--obj", type=str, required=True)
    ap.add_argument("--threshold", type=float, default=0.0)
    ap.add_argument("--show_rows", type=int, default=16)

    # QD
    ap.add_argument("--n_val", type=int, default=5)
    ap.add_argument("--initial_state", type=str, default="uniform")
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--decode_value", action="store_true")

    # QSP
    ap.add_argument("--qsp_degree", type=int, default=21)

    args = ap.parse_args()

    sp_params: Dict = {}
    if args.k is not None:
        sp_params["k"] = int(args.k)

    print("")
    print("=== check_oracle_semantics ===")
    print(f"method     = {args.method}")
    print(f"var_type   = {args.var_type}")
    print(f"n_key      = {args.n_key}")
    if args.method == "qd":
        print(f"n_val      = {args.n_val}")
        print(f"init_state = {args.initial_state}")
        print(f"sp_params  = {sp_params}")
    if args.method == "qsp":
        print(f"qsp_degree = {args.qsp_degree}")
    print(f"obj        = {args.obj}")
    print(f"threshold  = {args.threshold}")

    if args.method == "qsp":
        run_qsp_oracle_check(
            n_key=int(args.n_key),
            obj=str(args.obj),
            var_type=str(args.var_type),
            threshold=float(args.threshold),
            qsp_degree=int(args.qsp_degree),
            show_rows=int(args.show_rows),
        )
        return

    run_qd_stateprep_sign_check(
        n_key=int(args.n_key),
        n_val=int(args.n_val),
        obj=str(args.obj),
        var_type=str(args.var_type),
        threshold=float(args.threshold),
        initial_state=str(args.initial_state),
        state_prep_params=sp_params,
        show_rows=int(args.show_rows),
        decode_value=bool(args.decode_value),
    )


if __name__ == "__main__":
    main()
