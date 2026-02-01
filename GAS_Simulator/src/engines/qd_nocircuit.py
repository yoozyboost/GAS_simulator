import math
from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np
import sympy


def int_to_bin_le(i: int, n: int) -> str:
    """整数 i を、x0 が先頭（リトルエンディアン）となるビット列へ変換する。"""
    return np.binary_repr(i, width=n)[::-1]


def build_key_weights(n_key: int, init_state: str, state_prep_params: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """
    キーレジスタの初期分布 w_x を返す。
    w_x = |alpha_x|^2。
    bit列の約束は int_to_bin_le に一致。
    """
    if state_prep_params is None:
        state_prep_params = {}

    if init_state == "w":
        init_state = "w_state"

    N = 1 << n_key
    w = np.zeros(N, dtype=np.float64)

    if init_state in ("uniform", "hadamard"):
        w[:] = 1.0 / N
        return w

    if init_state == "w_state":
        if n_key == 0:
            w[0] = 1.0
            return w
        for i in range(n_key):
            w[1 << i] = 1.0 / n_key
        return w

    if init_state == "dicke":
        k = state_prep_params.get("k")
        if k is None:
            raise ValueError("initial_state='dicke' requires algorithm.state_prep_params.k.")
        for x in range(N):
            if x.bit_count() == int(k):
                w[x] = 1.0
        s = float(w.sum())
        if s == 0.0:
            raise ValueError(f"Dicke state has no basis states for n_key={n_key}, k={k}.")
        w /= s
        return w

    raise ValueError(f"Unknown initial_state: {init_state}")


def _sympy_expr(objfunstr: str):
    return sympy.expand(sympy.sympify(objfunstr))


def build_objfun_values_all_x(n_key: int, objfunstr: str, var_type: str) -> np.ndarray:
    """
    全状態 x (0..2^n_key-1) に対する目的関数値 f(x) を numpy 配列で返す。
    x_i の割り当ては、i のビット i を x_i に対応させる（LSB -> x0）。
    var_type:
      - 'binary': x_i ∈ {0,1}
      - 'spin'  : x_i ∈ {+1,-1} （0 -> +1, 1 -> -1）
    """
    expr = _sympy_expr(objfunstr)
    xs = sympy.symbols(f"x0:{n_key}")
    f = sympy.lambdify(xs, expr, "numpy")

    N = 1 << n_key
    bits = np.zeros((N, n_key), dtype=np.float64)
    for i in range(N):
        for j in range(n_key):
            b = (i >> j) & 1
            if var_type == "spin":
                bits[i, j] = 1.0 - 2.0 * float(b)
            else:
                bits[i, j] = float(b)

    vals = np.array(f(*[bits[:, j] for j in range(n_key)]), dtype=np.float64)
    return vals


def fejer_probs(a: np.ndarray, M: int) -> np.ndarray:
    """
    Fejér核（QFT離散化に起因する測定分布）に基づく確率表。
    入力 a は任意の実数配列で、0..M-1 の剰余を想定する。
    出力は shape=(len(a), M) で、各行が確率分布。
    """
    j = np.arange(M, dtype=np.float64)[None, :]
    x = a[:, None] - j
    num = np.sin(np.pi * x)
    den = np.sin(np.pi * x / M)

    ratio = np.empty_like(num)
    eps = 1e-12
    np.divide(num, den, out=ratio, where=(np.abs(den) >= eps))
    ratio[np.abs(den) < eps] = float(M)

    P = (ratio * ratio) / (M * M)
    P = np.clip(P, 0.0, 1.0)
    P_sum = P.sum(axis=1, keepdims=True)
    P /= P_sum
    return P


@dataclass
class QDNoCircuitEngine:
    """
    QD-GAS の回路なしシミュレーション。
    mode:
      - 'infinite' : 理想Grover（閾値で完全にマーキング）
      - 'fejer'    : n_val 有限（Fejér核に基づく確率的マーキング）
    """
    n_key: int
    n_val: int
    obj_values: np.ndarray
    weights: np.ndarray
    mode: str = "fejer"

    @staticmethod
    def build(n_key: int, n_val: int, objfunstr: str, var_type: str, init_state: str, state_prep_params: Optional[Dict[str, Any]] = None, mode: str = "fejer") -> "QDNoCircuitEngine":
        vals = build_objfun_values_all_x(n_key, objfunstr, var_type)
        w = build_key_weights(n_key, init_state, state_prep_params)
        return QDNoCircuitEngine(n_key=n_key, n_val=n_val, obj_values=vals, weights=w, mode=mode)

    def sample_key_int(self, threshold: float, rotation_count: int, rng: np.random.Generator) -> int:
        N = 1 << self.n_key
        w = self.weights

        if self.mode == "infinite":
            marked = (self.obj_values < float(threshold))
            p_mark = float(w[marked].sum())
            p_succ = _grover_success_prob(p_mark, rotation_count)

            if rng.random() <= p_succ:
                p = w * marked.astype(np.float64)
            else:
                p = w * (~marked).astype(np.float64)

            s = float(p.sum())
            if s <= 0.0:
                return int(rng.integers(N))
            return int(rng.choice(N, p=p / s))

        if self.mode == "fejer":
            M = 1 << int(self.n_val)
            half = M // 2

            d = self.obj_values - float(threshold)
            a = np.mod(d, M).astype(np.float64)

            P = fejer_probs(a, M)
            q_x = P[:, half:].sum(axis=1)  # 各 x が「マーキング側」に入る確率

            p_mark = float((w * q_x).sum())
            p_succ = _grover_success_prob(p_mark, rotation_count)

            if rng.random() <= p_succ:
                p = w * q_x
            else:
                p = w * (1.0 - q_x)

            s = float(p.sum())
            if s <= 0.0:
                return int(rng.integers(N))
            return int(rng.choice(N, p=p / s))

        raise ValueError(f"Unknown QDNoCircuitEngine.mode: {self.mode}")

    def sample_key_bitstring(self, threshold: float, rotation_count: int, rng: np.random.Generator) -> str:
        idx = self.sample_key_int(threshold, rotation_count, rng)
        return int_to_bin_le(idx, self.n_key)


def _grover_success_prob(p_mark: float, rotation_count: int) -> float:
    if p_mark <= 0.0:
        return 0.0
    if p_mark >= 1.0:
        return 1.0
    alpha = math.asin(math.sqrt(p_mark))
    return float(math.sin((2 * int(rotation_count) + 1) * alpha) ** 2)
