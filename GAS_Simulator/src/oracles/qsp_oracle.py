import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from functools import lru_cache
from collections import defaultdict
from .base import OracleBuilder
from utils.math_tools import str_to_sympy
import sympy

# pyqsp を使う場合の PolySign 生成パラメータ
DEFAULT_SIGN_DELTA = 20.0

# 互換性維持のため、pyqsp が無い場合は次数79のみ従来値でフォールバック
_FALLBACK_SIGN_ANGLES_D79 = [
    -1.5716717067761834, 0.0011503675299671112, -0.0012692522360260305,
    0.0014241998918902432, -0.0016215775143417765, 0.0018683974610131937,
    -0.0021723565891240693, 0.0025418772055978422, -0.002986150663268239,
    0.0035151850524568484, -0.004139859264196577, 0.004871986814923401,
    -0.005724394292989432, 0.006711021206927903, -0.007847050514211862,
    0.009149082376604944, -0.010635368007211543, 0.012326126279190719,
    -0.014243973726967507, 0.016414509740118488, -0.018867114792015194,
    0.021636043136030292, -0.02476192690819823, 0.028293863271141362,
    -0.032292342517240114, 0.036833414627504, -0.04201472383380778,
    0.04796443843633669, -0.05485480857573721, 0.06292338614692139,
    -0.07250745369663902, 0.08410232412240748, -0.0984652585380521,
    0.11681265011463404, -0.14122445672643558, 0.17556042597168542,
    -0.22782696262360758, 0.3174958692386929, -0.5045208121324722,
    1.0294926736346954, 1.0294926736346928, -0.5045208121324694,
    0.31749586923869333, -0.2278269626236091, 0.17556042597169,
    -0.1412244567264403, 0.11681265011463576, -0.09846525853805235,
    0.08410232412240845, -0.07250745369663975, 0.06292338614692264,
    -0.0548548085757366, 0.047964438436336354, -0.04201472383380822,
    0.036833414627504085, -0.032292342517240225, 0.028293863271141806,
    -0.02476192690819823, 0.021636043136030764, -0.018867114792015305,
    0.01641450974011846, -0.014243973726967729, 0.012326126279190941,
    -0.010635368007211293, 0.009149082376604833, -0.007847050514211418,
    0.006711021206927625, -0.005724394292989127, 0.0048719868149231516,
    -0.004139859264195939, 0.0035151850524559602, -0.0029861506632673784,
    0.0025418772055974814, -0.0021723565891233754, 0.0018683974610124443,
    -0.0016215775143414435, 0.0014241998918894383, -0.00126925223602542,
    0.0011503675299666394, -0.0008753799812867835
]


@lru_cache(maxsize=32)
def get_sign_angles_cached(degree: int, delta: float = 20.0) -> list[float]:
    """
    符号関数のQSP位相角列を pyqsp で生成して返す。
    - 常に奇数次数へ丸める。
    - 返り値は長さ degree+1 の実数リスト。
    - 固定リスト（d=79固定）を廃止し、任意次数に対応する。
    """

    d = int(degree)
    if d % 2 == 0:
        d += 1

    try:
        from pyqsp.angle_sequence import QuantumSignalProcessingPhases
        # from pyqsp.poly import PolySign
    except Exception as e:
        raise ImportError(
            "pyqsp が必要です。pip install pyqsp を実行してから再試行してください。"
        ) from e

    # cheb_samples は degree より大きく取る。エイリアシング回避のため。:contentReference[oaicite:1]{index=1}
    # cheb_samples = max(4 * d, d + 1, 250)
    cheb_samples = 4 * d

    # 固定deltaでtanh近似すると遷移幅が次数で縮まらない。
    # 次数を上げたときに遷移幅も縮むように、内部的にはdeltaを次数に比例させる。
    d0 = 79.0
    delta_eff = float(delta) * (float(d) / d0)
    # tanh(delta x) を Chebyshev 近似して符号関数を作る。
    # ここでは Chebyshev 基底係数 c[0..d] を直接 pyqsp に渡す。
    x = np.cos(np.pi * (np.arange(cheb_samples) + 0.5) / cheb_samples)
    y = np.tanh(delta_eff * x)
    c = np.polynomial.chebyshev.chebfit(x, y, d)

    # 奇関数なので偶数次数項をゼロ化する。
    c[0::2] = 0.0

    # |p(x)|<=1 を満たすように安全側に正規化する。
    grid = np.linspace(-1.0, 1.0, 4001)
    maxabs = float(np.max(np.abs(np.polynomial.chebyshev.chebval(grid, c))))
    if maxabs > 0.0:
        c *= (0.999 / maxabs)

    coeff = np.asarray(c, dtype=np.float64).reshape(-1).tolist()


    # 位相角生成（Wz 規約。あなたの実装は RX(-2*phi) を使うのでWz側でそろえる）
        # 位相角生成は method="sym_qsp" を明示する。
    # デフォルト laurent では chebyshev_basis=True が許されず例外になるため。
        # 位相角生成は Chebyshev 基底係数ベクトルを渡す。
    # method は sym_qsp を固定し、signal_operator は Wz を明示する。
    try:
        phases = QuantumSignalProcessingPhases(
            coeff,
            method="sym_qsp",
            chebyshev_basis=True,
            signal_operator="Wz",
        )
    except TypeError as e:
        raise RuntimeError(
            "pyqsp が signal_operator='Wz' または chebyshev_basis=True を受け付けない。"
            "この場合は pyqsp の更新が必要。"
        ) from e

    # sym_qsp系だと (full_phi, reduced_phi, parity) を返す実装があるため吸収。:contentReference[oaicite:4]{index=4}
    if isinstance(phases, tuple):
        phases = phases[0]

    phases = np.asarray(phases, dtype=np.float64).reshape(-1)
    if phases.size != d + 1:
        raise ValueError(f"pyqsp returned {phases.size} phases for degree={d} (expected {d+1}).")

    return phases.tolist()


class QSPOracleBuilder(OracleBuilder):

    def build_oracle(self, n_key: int, **kwargs) -> QuantumCircuit:
        obj_fun_str = kwargs.get('obj_fun_str')
        threshold = kwargs.get('threshold', 0.0)
        degree = kwargs.get('qsp_degree', 21)
        blockencoding_method = str(kwargs.get('blockencoding_method', 'legacy'))

        if degree % 2 == 0:
            degree += 1

        # --- 修正箇所: 正規化ロジックを元のQSP_ilquantumに合わせる ---

        # 1. 目的関数のみをパースして係数を取得
        expr = str_to_sympy(obj_fun_str)
        gens = [sympy.Symbol(f"x{i}") for i in range(int(n_key))]
        polydict = sympy.Poly(expr, *gens, domain="RR").as_dict()

        # 2. L1ノルム (係数の絶対値和) を計算。閾値は含めない。
        l1_norm = sum(abs(k) for k in polydict.values())
        if l1_norm == 0: l1_norm = 1.0 # ゼロ除算回避

        # 3. スケーリング係数 (pi/2 を掛けるのが重要)
        # QSP_ilquantum: objfun * pi / (2 * v)
        scale_factor = np.pi / (2 * l1_norm)

        # 4. 各項の係数をスケーリング
        scaled_polydict = {ps: float(k) * scale_factor for ps, k in polydict.items()}

        # 5. 閾値 (Delta) の計算
        # 元コード: Delta = float((-delta / (objfun_max)) * (pi / 2))
        # ここでの threshold は GASの文脈での閾値 y_th。
        # f(x) < y_th  <=>  f(x) - y_th < 0
        # 定数項として -y_th を加えることに相当するため、符号はマイナス
        scaled_threshold = (threshold * scale_factor)

        # 6. 角度計算
        angles = get_sign_angles_cached(degree)

        # 回路構築へ (スケーリング済みの辞書と閾値を渡す)
        return self._construct_qsp_circuit(
            n_key,
            scaled_polydict,
            scaled_threshold,
            angles,
            blockencoding_method=blockencoding_method,
        )

    def build_state_prep(self, n_key: int, obj_fun_str: str, state_prep_method, **kwargs) -> QuantumCircuit:
        qc = QuantumCircuit(n_key + 1, name="QSP_StatePrep")
        qc_key = state_prep_method.build(n_key, **kwargs)
        qc.compose(qc_key, range(n_key), inplace=True)
        return qc

    @staticmethod
    def _make_K(z_indices: list[int], n_qubits: int) -> list[int]:
        """
        varify_circuit.py の make_K と同一のロジック。
        Z項のインデックス集合（term_indices）の「相対構造」から、
        制御付きX（ancilla→data）の適用先インデックス集合 K を生成する。
        """
        if not z_indices:
            return []

        sorted_indices = sorted(z_indices)
        min_index = sorted_indices[0]
        normalized_indices = [idx - min_index for idx in sorted_indices]
        max_normalized = normalized_indices[-1]
        required_bits = max_normalized.bit_length()
        bit_space_size = 1 << required_bits

        z_pattern_mask = 0
        for idx in normalized_indices:
            z_pattern_mask |= (1 << idx)

        block_size = bit_space_size >> 1
        a = z_pattern_mask & ((1 << block_size) - 1) if block_size > 0 else 0
        b = z_pattern_mask >> block_size

        control_bits = [0] * required_bits
        for i in range(required_bits):
            if a == b:
                control_bits[i] = 1
            else:
                control_bits[i] = 0
                a ^= b

            block_size >>= 1
            if block_size > 0:
                mask = (1 << block_size) - 1
                b = a & mask
                a >>= block_size
            else:
                b = 0

        x_pattern_generator = 1
        for i in range(required_bits):
            previous_generator = x_pattern_generator
            shift_amount = 1 << i
            x_pattern_generator <<= shift_amount
            if control_bits[required_bits - 1 - i] == 0:
                x_pattern_generator += previous_generator

        x_pattern_str = bin(x_pattern_generator)[2:].zfill(bit_space_size)
        if bit_space_size == 0:
            return []

        x_indices = [
            i for i in range(n_qubits)
            if x_pattern_str[i % bit_space_size] == '1'
        ]
        return x_indices

    @classmethod
    def _build_wz_signal_optimized_gate(cls, n_key: int, polydict: dict) -> Gate:
        """
        Wz 信号演算子（|0><0|⊗U + |1><1|⊗U†）を、
        制御付きK（ancilla→data の CX 群）による共役で構成する。

        量子ビットの並びは [ancilla] + [key0..key_{n_key-1}] とする。
        """
        grouped_terms: dict[tuple[int, ...], list[tuple[tuple[int, ...], float]]] = defaultdict(list)

        for (ps, k) in polydict.items():
            k = float(k)
            if abs(k) < 1e-12:
                continue

            term_indices = [i for i, power in enumerate(ps) if power > 0]
            if not term_indices:
                continue

            k_key = tuple(cls._make_K(term_indices, n_key))
            grouped_terms[k_key].append((tuple(term_indices), k))

        Wz = QuantumCircuit(n_key + 1, name="Wz_signal_opt")
        ancilla = 0

        for k_indices, terms in grouped_terms.items():
            if k_indices:
                for idx in k_indices:
                    Wz.cx(ancilla, int(idx) + 1)

            for term_indices, coeff in terms:
                angle = 2.0 * float(coeff)
                if len(term_indices) == 1:
                    Wz.rz(angle, int(term_indices[0]) + 1)
                else:
                    for i in range(len(term_indices) - 1):
                        Wz.cx(int(term_indices[i]) + 1, int(term_indices[i + 1]) + 1)
                    Wz.rz(angle, int(term_indices[-1]) + 1)
                    for i in range(len(term_indices) - 2, -1, -1):
                        Wz.cx(int(term_indices[i]) + 1, int(term_indices[i + 1]) + 1)

            if k_indices:
                for idx in k_indices:
                    Wz.cx(ancilla, int(idx) + 1)

        return Wz.to_gate()

    def _construct_qsp_circuit(self, n_key, polydict, scaled_threshold, angles, blockencoding_method: str = "legacy"):
        # polydictは既にスケーリング済み

        # 定数項の集約 (閾値分を引く)
        Delta = -np.pi / 2 - float(scaled_threshold)

        O = QuantumCircuit(n_key + 1, name="QSP_Oracle")

        use_optimized = (blockencoding_method.lower() in {"optimized", "control_k", "controlled_k", "k_grouped"})
        if use_optimized:
            Wz_signal_gate = self._build_wz_signal_optimized_gate(int(n_key), polydict)
        else:
            Wz_plus = QuantumCircuit(n_key)
            Wz_minus = QuantumCircuit(n_key)

        for (ps, k) in polydict.items():
            k = float(k)
            i_nonzero = np.nonzero(ps)[0]
            n_nonzero = len(i_nonzero)

            if n_nonzero == 0:
                # 目的関数自体の定数項を加算
                Delta += k
            else:
                if not use_optimized:
                    for i in range(n_nonzero - 1):
                        Wz_plus.cx(i_nonzero[i], i_nonzero[-1])
                        Wz_minus.cx(i_nonzero[i], i_nonzero[-1])

                    # 回転角: QSP_ilquantumでは 2*k (kは既にpi/2スケール済み)
                    Wz_plus.rz(2 * k, i_nonzero[-1])
                    Wz_minus.rz(-2 * k, i_nonzero[-1])

                    for i in range(n_nonzero - 1):
                        Wz_plus.cx(i_nonzero[n_nonzero - 2 - i], i_nonzero[-1])
                        Wz_minus.cx(i_nonzero[n_nonzero - 2 - i], i_nonzero[-1])

        if not use_optimized:
            controlled_Wz_plus = Wz_plus.to_gate().control(1)
            controlled_Wz_minus = Wz_minus.to_gate().control(1)

        ancilla = n_key

        O.rx(-2 * angles[0], ancilla)
        for i in range(1, len(angles)):
            if use_optimized:
                O.append(Wz_signal_gate, [ancilla] + list(range(n_key)))
            else:
                O.x(ancilla)
                O.append(controlled_Wz_plus, [ancilla] + list(range(n_key)))
                O.x(ancilla)
                O.append(controlled_Wz_minus, [ancilla] + list(range(n_key)))

            if abs(Delta) > 1e-9:
                O.rz(float(2 * Delta), ancilla)


            O.rx(-2 * angles[i], ancilla)
        # nocircuit と同じ「S† + X」を末尾に付加する
        O.sdg(ancilla)
        O.x(ancilla)

        return O
