from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Any, List, Optional

import numpy as np
import sympy

from utils.math_tools import str_to_sympy
from oracles.qsp_oracle import get_sign_angles_cached


def _parity_prod(indices: np.ndarray, mask: int) -> np.ndarray:
    """
    prod_z = (-1)^{popcount(indices & mask)} を {+1,-1} で返す。
    indices は 0..2^n_key-1 の配列。
    mask の i ビットは key の qubit i（LSBがqubit0）に対応。
    """
    x = (indices.astype(np.uint64) & np.uint64(mask))
    x ^= x >> np.uint64(32)
    x ^= x >> np.uint64(16)
    x ^= x >> np.uint64(8)
    x ^= x >> np.uint64(4)
    x ^= x >> np.uint64(2)
    x ^= x >> np.uint64(1)
    parity = (x & np.uint64(1)).astype(np.int8)
    return (1 - 2 * parity).astype(np.float64)


@dataclass
class _ScaledPoly:
    scale_factor: float
    const_scaled: float
    terms: List[Tuple[int, float]]  # (mask, coeff_scaled)


class QSPNoCircuitEngine:
    """
    qsp_oracle.py と同一の規約で、回路なしに statevector を更新してサンプリングする。

    重要点
      - 係数のL1ノルム v = Σ|c| を用いて scale_factor = pi/(2v)
      - Delta = -scaled_threshold + const_scaled
      - Wz_plus は exp(-i * theta(x)) を key に、Wz_minus は exp(+i * theta(x)) を key に作用
      - ancilla は n_key 番目の qubit、statevector は [anc=0ブロック, anc=1ブロック] として扱う

    高速化
      - diffuser は 2|psi_ref><psi_ref| - I
      - psi_ref が anc=0 のみに支持を持つ形（本実装のstate_prepでは常にそう）なので
        inner product と更新を key 側だけで計算できる
      - key 側の支持が疎（w_state, dicke）なら支持部分のみで inner を計算し更新する
      - さらに支持上で振幅が一定なら、更新は定数シフトになる
    """

    def __init__(
        self,
        n_key: int,
        obj_fun_str: str,
        qsp_degree: int,
        var_type: str,
        psi_ref: np.ndarray,
        seed: int | None = None,
    ):
        self.n_key = int(n_key)
        self.obj_fun_str = str(obj_fun_str)
        self.qsp_degree = int(qsp_degree)
        self.var_type = str(var_type)

        self.psi_ref = np.asarray(psi_ref, dtype=np.complex128)
        if self.psi_ref.ndim != 1:
            raise ValueError("psi_ref must be 1D statevector.")
        if self.psi_ref.size != (1 << (self.n_key + 1)):
            raise ValueError("psi_ref size mismatch for n_key+1 qubits.")

        self.rng = np.random.default_rng(seed)

        requested_degree = self.qsp_degree
        if requested_degree % 2 == 0:
            requested_degree += 1

        # qsp_oracle.py の get_sign_angles_cached は、現状では degree に依存せず固定の角度列を返す。
        # 回路構築側は angles の長さで実効degreeが決まるため、ここもそれに合わせる。
        self.angles = np.asarray(get_sign_angles_cached(requested_degree), dtype=np.float64)
        if self.angles.size < 2:
            raise ValueError("angles is too short.")

        self.qsp_degree = int(self.angles.size - 1)


        self._poly = self._build_scaled_poly(self.obj_fun_str, self.n_key)

        self._N = 1 << self.n_key
        self._indices = np.arange(self._N, dtype=np.uint64)
        self._x_terms = self._precompute_theta(self._indices, self._poly.terms)
        self._N = 1 << self.n_key
        self._init_reflection_fastpath()


        self._init_reflection_fastpath()

    def _init_reflection_fastpath(self) -> None:
        """
        diffuser の高速パス用に、psi_ref の key 側支持を抽出する。
        psi_ref は一般には任意だが、現行state_prepでは anc=1 側はゼロ。
        """
        n = self._N
        a1 = self.psi_ref[n:]
        if float(np.linalg.norm(a1)) > 1e-12:
            self._ref_mode = "generic_full"
            self._phi_support = None
            self._phi_support_idx = None
            self._phi_const = None
            return

        phi = self.psi_ref[:n]
        idx = np.flatnonzero(np.abs(phi) > 1e-14).astype(np.int64)
        if idx.size == 0:
            self._ref_mode = "generic_full"
            self._phi_support = None
            self._phi_support_idx = None
            self._phi_const = None
            return

        phi_sup = phi[idx].copy()

        mags = np.abs(phi_sup)
        if float(mags.max() - mags.min()) <= 1e-13:
            phi0 = phi_sup[0]
            if abs(phi0) > 0:
                ph = phi_sup / phi0
                if float(np.max(np.abs(ph - 1.0))) <= 1e-12:
                    self._ref_mode = "anc0_support_const"
                    self._phi_support_idx = idx
                    self._phi_support = None
                    self._phi_const = phi0
                    return

        self._ref_mode = "anc0_support_vec"
        self._phi_support_idx = idx
        self._phi_support = phi_sup
        self._phi_const = None

    @staticmethod
    def _build_scaled_poly(obj_fun_str: str, n_key: int) -> _ScaledPoly:
        """
        qsp_oracle.py と同じ正規化：
          v = Σ |c_S|,
          scale_factor = pi/(2v),
          scaled_coeff = c_S * scale_factor,
          Delta = -scaled_threshold + (定数項のscaled_coeffの和)
        """
        expr = str_to_sympy(obj_fun_str)

        gens = [sympy.Symbol(f"x{i}") for i in range(int(n_key))]
        poly = sympy.Poly(expr, *gens, domain="RR")
        polydict: Dict[Tuple[int, ...], Any] = poly.as_dict()

        l1 = 0.0
        for c in polydict.values():
            l1 += abs(float(c))
        if l1 == 0.0:
            l1 = 1.0

        # scale_factor = float(np.pi) / (2.0 * l1)
        scale_factor = 1.0 / (2.0 * l1)


        const_scaled = 0.0
        terms: List[Tuple[int, float]] = []

        for ps, c in polydict.items():
            k = float(c) * scale_factor
            if all(int(e) == 0 for e in ps):
                const_scaled += k
                continue

            mask = 0
            for i, e in enumerate(ps):
                if int(e) != 0:
                    mask |= (1 << i)

            if mask == 0:
                const_scaled += k
            else:
                terms.append((mask, k))

        return _ScaledPoly(scale_factor=scale_factor, const_scaled=const_scaled, terms=terms)

    @staticmethod
    def _precompute_theta(indices: np.ndarray, terms: List[Tuple[int, float]]) -> np.ndarray:
        theta = np.zeros(indices.shape[0], dtype=np.float64)
        for mask, k in terms:
            prod = _parity_prod(indices, int(mask))
            # theta += float(k) * prod
        return theta

    @staticmethod
    def _apply_rx_minus2phi(a0: np.ndarray, a1: np.ndarray, phi: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rx(-2*phi) を ancilla 振幅 (a0,a1) に適用。
        Rx(-2φ) = [[cosφ, i sinφ],
                   [i sinφ, cosφ]]
        """
        c = float(np.cos(phi))
        s = float(np.sin(phi))
        new0 = c * a0 + 1j * s * a1
        new1 = 1j * s * a0 + c * a1
        return new0, new1

    def _apply_one_oracle(self, state: np.ndarray, threshold: float) -> np.ndarray:
        """
        qsp_oracle.py の QSP_Oracle を、状態ベクトルへ直接適用。
        ancilla=0 ブロックに Wz_plus、ancilla=1 ブロックに Wz_minus を適用する。
        """
        n = self._N
        a0 = state[:n].copy()
        a1 = state[n:].copy()

        scaled_threshold = float(threshold) * self._poly.scale_factor
        Delta = -scaled_threshold + float(self._poly.const_scaled)

        a0, a1 = self._apply_rx_minus2phi(a0, a1, float(self.angles[0]))

        # x = (f - threshold)/(2*l1) を作り、theta = arccos(x) を信号角として使う
        x = self._x_terms + (self._poly.const_scaled - float(threshold) * self._poly.scale_factor)
        x = np.clip(x, -1.0, 1.0)
        theta = np.arccos(x)

        phase_plus = np.exp(-1j * theta)
        phase_minus = np.conjugate(phase_plus)

        a0 *= phase_plus
        a1 *= phase_minus


        return np.concatenate([a0, a1], axis=0)

    def _init_reflection_fastpath(self) -> None:
        """
        diffuser D = 2|psi_ref><psi_ref| - I の高速パス初期化。

        現行の state_prep では ancilla=1 側がゼロであることが多いので、
        ancilla=0 側の key 振幅 phi の支持だけを保持して inner product を高速計算する。

        - ancilla=1 にも支持がある場合は generic_full に落とす。
        - ancilla=0 の支持が疎なら O(|support|)。
        - 支持上で振幅が定数なら inner は sum だけで計算できる。
        """
        n = 1 << self.n_key

        a1 = self.psi_ref[n:]
        if float(np.linalg.norm(a1)) > 1e-12:
            self._ref_mode = "generic_full"
            self._phi_support_idx = None
            self._phi_support = None
            self._phi_const = None
            return

        phi = self.psi_ref[:n]
        idx = np.flatnonzero(np.abs(phi) > 1e-14).astype(np.int64)
        if idx.size == 0:
            self._ref_mode = "generic_full"
            self._phi_support_idx = None
            self._phi_support = None
            self._phi_const = None
            return

        phi_sup = phi[idx].copy()
        mags = np.abs(phi_sup)

        if float(mags.max() - mags.min()) <= 1e-13:
            phi0 = phi_sup[0]
            if abs(phi0) > 0:
                ph = phi_sup / phi0
                if float(np.max(np.abs(ph - 1.0))) <= 1e-12:
                    self._ref_mode = "anc0_support_const"
                    self._phi_support_idx = idx
                    self._phi_support = None
                    self._phi_const = phi0
                    return

        self._ref_mode = "anc0_support_vec"
        self._phi_support_idx = idx
        self._phi_support = phi_sup
        self._phi_const = None

    def _apply_diffuser_about_psi(self, state: np.ndarray) -> np.ndarray:
        """
        D = 2|psi_ref><psi_ref| - I
        高速パスが使える場合は ancilla=0 側の支持だけで計算する。
        """
        n = 1 << self.n_key

        if getattr(self, "_ref_mode", "generic_full") == "generic_full":
            inner = np.vdot(self.psi_ref, state)
            return 2.0 * inner * self.psi_ref - state

        a0 = state[:n].copy()
        a1 = state[n:].copy()

        mode = self._ref_mode

        if mode == "anc0_support_const":
            idx = self._phi_support_idx
            phi0 = self._phi_const

            s = a0[idx].sum()
            inner = np.conjugate(phi0) * s

            a0 = -a0
            a1 = -a1
            shift = 2.0 * inner * phi0
            a0[idx] += shift
            return np.concatenate([a0, a1], axis=0)

        if mode == "anc0_support_vec":
            idx = self._phi_support_idx
            phi_sup = self._phi_support

            inner = np.vdot(phi_sup, a0[idx])

            a0 = -a0
            a1 = -a1
            a0[idx] += 2.0 * inner * phi_sup
            return np.concatenate([a0, a1], axis=0)

        inner = np.vdot(self.psi_ref, state)
        return 2.0 * inner * self.psi_ref - state


    def run_grover_iterations(
        self,
        threshold: float,
        rotation_count: int,
        initial_state: np.ndarray | None = None,
    ) -> np.ndarray:
        r = int(rotation_count)
        if r < 1:
            raise ValueError("rotation_count must be >= 1.")

        if initial_state is None:
            state = self.psi_ref.copy()
        else:
            state = np.asarray(initial_state, dtype=np.complex128).copy()

        for _ in range(r):
            state = self._apply_one_oracle(state, float(threshold))
            state = self._apply_diffuser_about_psi(state)

        return state

    def sample_key_bitstring(
        self,
        threshold: float,
        rotation_count: int,
        initial_state: np.ndarray | None = None,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[str, np.ndarray]:
        """
        返り値の bitstring は既存実装に合わせて x0 が左になるように反転して返す。
        rng を渡せば外部の乱数系列に揃えられる。
        """
        if rng is None:
            rng = self.rng

        state = self.run_grover_iterations(threshold, rotation_count, initial_state=initial_state)
        n = self._N
        a0 = state[:n]
        a1 = state[n:]
        p_key = (np.abs(a0) ** 2 + np.abs(a1) ** 2).astype(np.float64)
        s = float(p_key.sum())
        if s <= 0.0:
            p_key = np.ones_like(p_key) / float(p_key.size)
        else:
            p_key /= s

        if rng is None:
            rng = self.rng
        idx = int(rng.choice(n, p=p_key))

        bitstr = format(idx, f"0{self.n_key}b")[::-1]
        return bitstr, p_key
