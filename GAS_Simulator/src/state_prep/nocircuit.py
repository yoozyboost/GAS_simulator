from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


def build_key_weights(
    n_key: int,
    initial_state: str,
    state_prep_params: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """キーレジスタの初期分布 w_x を返す。

    定義
      - 計算基底 |x> は整数 x の2進表現で表す。
      - 0番目のビット（LSB）は key の qubit0（x0）に対応する。
      - w_x = |\alpha_x|^2 とする。

    対応する initial_state
      - uniform, hadamard: |+>^{\otimes n_key}
      - w_state: |W_{n_key}> = (1/sqrt(n_key)) \sum_i |1_i>
      - dicke: |D^n_k> = (1/sqrt(C(n,k))) \sum_{|x|=k} |x>
    """

    if state_prep_params is None:
        state_prep_params = {}

    n_key = int(n_key)
    if n_key < 0:
        raise ValueError("n_key must be >= 0")

    N = 1 << n_key
    w = np.zeros(N, dtype=np.float64)

    name = str(initial_state)
    if name in ("uniform", "hadamard"):
        if N == 0:
            return w
        w[:] = 1.0 / float(N)
        return w

    if name == "w_state":
        if n_key == 0:
            w[0] = 1.0
            return w
        for i in range(n_key):
            w[1 << i] = 1.0 / float(n_key)
        return w

    if name == "dicke":
        k = state_prep_params.get("k", None)
        if k is None:
            raise ValueError("initial_state='dicke' requires algorithm.state_prep_params.k")
        k = int(k)
        if k < 0 or k > n_key:
            raise ValueError("dicke parameter k must satisfy 0 <= k <= n_key")

        for x in range(N):
            if int(x).bit_count() == k:
                w[x] = 1.0

        s = float(w.sum())
        if s <= 0.0:
            raise ValueError(f"Dicke state has no basis states for n_key={n_key}, k={k}")
        w /= s
        return w

    raise ValueError(f"Unknown initial_state: {initial_state}")


def build_key_statevector(
    n_key: int,
    initial_state: str,
    state_prep_params: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """キーレジスタの状態ベクトル \alpha を返す。

    本関数は、上記 build_key_weights の w に対して
      \alpha_x = sqrt(w_x)
    として実装する。

    備考
      - 位相はすべて 0 とする。
      - 対応する3状態（uniform, w_state, dicke）では、既存の state_prep 回路が
        生成する状態と一致する。
    """
    w = build_key_weights(n_key, initial_state, state_prep_params)
    alpha = np.sqrt(w).astype(np.complex128)
    return alpha


def build_psi_ref(
    n_key: int,
    initial_state: str,
    state_prep_params: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """QSP回路なしエンジン用の基準状態 |psi_ref> を作る。

    既存の qsp_nocircuit は、diffuser を
      D = 2|psi_ref><psi_ref| - I
    として実装する。
    現行の回路あり実装では、state_prep は ancilla を |0> のまま残す。
    したがって、ここでは
      |psi_ref> = |0>_anc \otimes \sum_x \alpha_x |x>
    を返す。

    出力
      - 長さ 2^(n_key+1) の statevector
      - 前半が anc=0 ブロック、後半が anc=1 ブロック
    """

    n_key = int(n_key)
    N = 1 << n_key
    psi = np.zeros(2 * N, dtype=np.complex128)
    psi[:N] = build_key_statevector(n_key, initial_state, state_prep_params)
    # 正規化（安全策）
    nrm = float(np.linalg.norm(psi))
    if nrm > 0.0:
        psi /= nrm
    return psi
