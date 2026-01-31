# GAS_Simulator/src/core/gas_model.py

from __future__ import annotations

from typing import Any, Dict, Tuple, Optional

import numpy as np

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

from oracles import ORACLE_FACTORY
from state_prep import STATE_PREP_FACTORY

from state_prep.nocircuit import build_psi_ref

from engines.qd_nocircuit import QDNoCircuitEngine
from engines.qsp_nocircuit import QSPNoCircuitEngine


class GASModel:
    """
    GAS の「モデル」層。
    - 回路資産（回路図・コスト評価用）はデフォルトで構築する。
    - 収束評価（trialを回してbest曲線を見る）はデフォルトで回路なし backend を用いる。
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # --- problem ---
        prob = config.get("problem", {})
        self.n_key = int(prob.get("n_key", 2))
        self.n_val = int(prob.get("n_val", 5))  # QD用。QSPでは使わないが保持
        self.obj_fun_str = str(prob.get("objective_function", "x0 + x1"))
        self.var_type = str(prob.get("variable_type", "binary"))  # "binary" or "spin"

        # --- algorithm ---
        algo = config.get("algorithm", {})
        self.method = str(algo.get("method", "qd"))  # "qd" or "qsp"
        self.state_prep_name = str(algo.get("initial_state", "uniform"))
        self.state_prep_params = algo.get("state_prep_params", {}) or {}
        self.qsp_params = algo.get("qsp_params", {}) or {}

        # --- simulation ---
        simcfg = config.get("simulation", {})
        # backend は「回路を実際に実行する場合の方式」だが、過去設定との互換のため
        # backend="analytical" を指定された場合は convergence_backend="nocircuit" とみなし、
        # QD は qd_nocircuit_mode="infinite" をデフォルトに寄せる。
        self.backend = str(simcfg.get("backend", "statevector"))  # "statevector" or "qasm" or legacy "analytical"
        self.convergence_backend = str(simcfg.get("convergence_backend", "nocircuit"))
        self.max_iterations = int(simcfg.get("max_iterations", 30))
        self.seed = simcfg.get("seed", None)

        # 収束シミュレーション用の乱数。
        # runner.py 側でも rng を持つが、モデル単体利用とテストのためここでも保持する。
        self._rng = np.random.default_rng(self.seed)

        # QD 回路なしモード
        self.qd_nocircuit_mode = str(simcfg.get("qd_nocircuit_mode", "fejer"))

        # legacy analytical モードの吸収
        if self.backend == "analytical":
            self.convergence_backend = "nocircuit"
            if self.method == "qd" and ("qd_nocircuit_mode" not in simcfg):
                self.qd_nocircuit_mode = "infinite"
            # backend は回路実行系にのみ影響するので、ここでは statevector に寄せる
            self.backend = "statevector"

        # --- factories ---
        if self.method not in ORACLE_FACTORY:
            raise ValueError(f"Unknown method: {self.method}")
        if self.state_prep_name not in STATE_PREP_FACTORY:
            raise ValueError(f"Unknown initial_state: {self.state_prep_name}")

        self.oracle_builder = ORACLE_FACTORY[self.method]()
        self.state_prep_method = STATE_PREP_FACTORY[self.state_prep_name]()

        # --- circuit assets ---
        assets = config.get("circuit_assets", {}) or {}
        self.build_circuit_assets = bool(assets.get("build", True))

        self.qc_s: Optional[QuantumCircuit] = None
        self.qc_o: Optional[QuantumCircuit] = None
        self.qc_diff: Optional[QuantumCircuit] = None
        self.qc_grover: Optional[QuantumCircuit] = None
        self.qc_full_1step: Optional[QuantumCircuit] = None

        if self.build_circuit_assets:
            self._build_default_circuit_assets()

        # --- QD nocircuit engine ---
        self._qd_engine: Optional[QDNoCircuitEngine] = None
        if self.method == "qd":
            self._qd_engine = QDNoCircuitEngine.build(
                n_key=int(self.n_key),
                n_val=int(self.n_val),
                objfunstr=str(self.obj_fun_str),
                var_type=str(self.var_type),
                init_state=str(self.state_prep_name),
                state_prep_params=self.state_prep_params,
                mode=str(self.qd_nocircuit_mode),
            )

        # --- QSP nocircuit engine ---
        self._qsp_engine: Optional[QSPNoCircuitEngine] = None
        if self.method == "qsp":
            # diffuser 反射の基準状態 |psi_ref> を回路なしで構成する。
            # state_prep は現状 (uniform, w_state, dicke) のみをサポートし、
            # いずれも ancilla=0 側にのみ支持を持つ。
            psi_ref = build_psi_ref(
                n_key=int(self.n_key),
                initial_state=str(self.state_prep_name),
                state_prep_params=self.state_prep_params,
            )
            qsp_degree = int(self.qsp_params.get("qsp_degree", 9))

            self._qsp_engine = QSPNoCircuitEngine(
                n_key=self.n_key,
                obj_fun_str=self.obj_fun_str,
                qsp_degree=qsp_degree,
                var_type=self.var_type,
                psi_ref=psi_ref,
                seed=self.seed,
            )

    # -------------------------
    # circuit construction
    # -------------------------

    def _build_state_prep(self, threshold: float) -> QuantumCircuit:
        """
        oracle_builder 側の build_state_prep を呼び、キーワード引数で統一する。
        """
        kwargs: Dict[str, Any] = {}
        if self.method == "qd":
            kwargs["n_val"] = int(self.n_val)
            kwargs["is_spin"] = (self.var_type == "spin")
            kwargs["threshold"] = float(threshold)
            kwargs.update(self.state_prep_params)
            return self.oracle_builder.build_state_prep(
                n_key=int(self.n_key),
                obj_fun_str=self.obj_fun_str,
                state_prep_method=self.state_prep_method,
                **kwargs,
            )

        # qsp
        kwargs.update(self.state_prep_params)
        return self.oracle_builder.build_state_prep(
            n_key=int(self.n_key),
            obj_fun_str=self.obj_fun_str,
            state_prep_method=self.state_prep_method,
            **kwargs,
        )

    def _build_oracle(self, threshold: float) -> QuantumCircuit:
        if self.method == "qd":
            return self.oracle_builder.build_oracle(
                n_key=int(self.n_key),
                obj_fun_str=self.obj_fun_str,
                threshold=float(threshold),
                n_val=int(self.n_val),
                is_spin=(self.var_type == "spin"),
            )

        # qsp
        qsp_degree = int(self.qsp_params.get("qsp_degree", 9))
        return self.oracle_builder.build_oracle(
            n_key=int(self.n_key),
            obj_fun_str=self.obj_fun_str,
            threshold=float(threshold),
            qsp_degree=qsp_degree,
        )

    @staticmethod
    def _build_diffuser(n_qubits: int) -> QuantumCircuit:
        """
        標準 diffuser: 2|0><0| - I を実装する回路。
        Grover step 全体では S ... S^\dagger の共役で 2|psi><psi| - I になる。
        """
        n = int(n_qubits)
        qc = QuantumCircuit(n, name="Diffuser")
        if n == 1:
            qc.z(0)
            return qc

        qc.h(range(n))
        qc.x(range(n))
        qc.h(n - 1)
        qc.mcx(list(range(n - 1)), n - 1)
        qc.h(n - 1)
        qc.x(range(n))
        qc.h(range(n))
        return qc

    @staticmethod
    def _build_diffuser_ilquantum(n_controls: int) -> QuantumCircuit:
        # 総量子ビット数は n_controls + 1、最後が拡散用ancilla
        anc = n_controls
        qc = QuantumCircuit(n_controls + 1, name="Diffuser")
        qc.x(range(n_controls))
        qc.x(anc)
        qc.h(anc)
        qc.mcx(list(range(n_controls)), anc)
        qc.h(anc)
        qc.x(anc)
        qc.x(range(n_controls))
        return qc

    def construct_circuit(self, threshold: float, rotation_count: int) -> Tuple[QuantumCircuit, QuantumCircuit, QuantumCircuit, QuantumCircuit, QuantumCircuit]:
        qc_s = self._build_state_prep(threshold=float(threshold))
        qc_o = self._build_oracle(threshold=float(threshold))

        if self.method == "qd":
            n_controls = int(self.n_key + self.n_val)
        else:
            n_controls = int(self.n_key)

        qc_diff = self._build_diffuser_ilquantum(n_controls)
        n_qubits = n_controls + 1

        qc_g = QuantumCircuit(n_qubits, name="GroverStep")
        qc_g.compose(qc_o, inplace=True)
        qc_g.compose(qc_s.inverse(), inplace=True)
        qc_g.compose(qc_diff, inplace=True)
        qc_g.compose(qc_s, inplace=True)

        qc_full = QuantumCircuit(n_qubits, name="full_circuit_1step")
        qc_full.compose(qc_s, inplace=True)
        for _ in range(int(rotation_count)):
            qc_full.compose(qc_g, inplace=True)

        return qc_s, qc_o, qc_diff, qc_g, qc_full

    def _build_default_circuit_assets(self):
        qc_s, qc_o, qc_diff, qc_g, qc_full = self.construct_circuit(threshold=0.0, rotation_count=1)

        # 名前は既存の出力と合わせる（Aerが嫌うときはdecomposeして実行側で潰す）
        qc_s.name = "StatePrep"
        qc_o.name = "Oracle"
        qc_diff.name = "Diffuser"
        qc_g.name = "GroverStep"
        qc_full.name = "full_circuit_1step"

        self.qc_s = qc_s
        self.qc_o = qc_o
        self.qc_diff = qc_diff
        self.qc_grover = qc_g
        self.qc_full_1step = qc_full

    def get_components_for_visualization(self) -> Dict[str, QuantumCircuit]:
        out: Dict[str, QuantumCircuit] = {}
        if self.qc_s is not None:
            out["StatePrep"] = self.qc_s
        if self.qc_o is not None:
            out["Oracle"] = self.qc_o
        if self.qc_diff is not None:
            out["Diffuser"] = self.qc_diff
        if self.qc_grover is not None:
            out["GroverStep"] = self.qc_grover
        if self.qc_full_1step is not None:
            out["full_circuit_1step"] = self.qc_full_1step
        return out

    # -------------------------
    # execution backends
    # -------------------------

    def run_step(self, threshold: float, rotation_count: int) -> str:
        if self.convergence_backend == "circuit":
            return self.run_step_circuit(threshold, rotation_count)
        return self.run_step_nocircuit(threshold, rotation_count)

    def run_step_circuit(self, threshold: float, rotation_count: int) -> str:
        """
        回路ありで 1 step サンプリング。
        AerError 'unknown instruction: StatePrep' を避けるため、
        実行前に decompose を必ず挟む。
        """
        _, _, _, _, qc_full = self.construct_circuit(threshold=float(threshold), rotation_count=int(rotation_count))

        # Aerが嫌う named Instruction を落とす
        qc_full = qc_full.decompose(reps=10)

        n_key = self.n_key

        # statevector は Aer を使わず Statevector に寄せる方が安定
        if self.backend == "statevector":
            sv = Statevector.from_instruction(qc_full)
            probs = np.abs(sv.data) ** 2

            # QSP: ancilla を周辺化して key 分布にする
            if self.method == "qsp":
                n = 1 << n_key
                p_key = probs[:n] + probs[n: n + n]
                p_key = p_key / p_key.sum()
                idx = int(self._rng.choice(n, p=p_key))
                return format(idx, f"0{n_key}b")[::-1]

            # QD: key は下位 n_key ビット
            n = 1 << n_key
            p_key = np.zeros(n, dtype=np.float64)
            for basis_idx, p in enumerate(probs):
                key_idx = basis_idx & (n - 1)
                p_key[key_idx] += float(p)
            p_key = p_key / p_key.sum()
            idx = int(self._rng.choice(n, p=p_key))
            return format(idx, f"0{n_key}b")[::-1]

        # qasm backend
        qc_full.measure_all()
        sim = AerSimulator()
        qc_full = transpile(qc_full, sim, optimization_level=0)
        result = sim.run(qc_full, shots=1).result()
        counts = result.get_counts()
        bitstr = next(iter(counts.keys()))

        # qiskit counts は classical bit の順が回路依存なので、ここでは安全策として key 部分を右から取る
        # ancilla/val を含む場合でも、末尾 n_key を key として読む
        key_bits = bitstr.replace(" ", "")[-n_key:]
        return key_bits[::-1]

    def transpile_circuit(self, qc: QuantumCircuit) -> QuantumCircuit:
        """circuit_evaluation 設定に基づきトランスパイルした回路を返す。"""
        eval_config = self.config.get("circuit_evaluation") or {}
        opt_level = int(eval_config.get("optimization_level", 1))
        basis_gates = eval_config.get("basis_gates", None)

        # Aer が嫌う named Instruction を落とす
        qc2 = qc.decompose(reps=10)
        sim = AerSimulator()

        if basis_gates is None or basis_gates == "Default":
            return transpile(qc2, sim, optimization_level=opt_level)
        return transpile(qc2, sim, optimization_level=opt_level, basis_gates=list(basis_gates))

    def run_step_nocircuit(self, threshold: float, rotation_count: int, rng: Optional[np.random.Generator] = None) -> str:
        """
        回路なしで 1 step サンプリング。
        QSP は engines/qsp_nocircuit.py を使用する。
        """
        if rng is None:
            rng = self._rng

        if self.method == "qsp":
            if self._qsp_engine is None:
                raise RuntimeError("QSPNoCircuitEngine is not initialized.")
            bitstr, _ = self._qsp_engine.sample_key_bitstring(
                threshold=float(threshold),
                rotation_count=int(rotation_count),
                initial_state=None,
                rng=rng,
            )

            return bitstr

        # QD
        if self._qd_engine is None:
            raise RuntimeError("QDNoCircuitEngine is not initialized.")
        return self._qd_engine.sample_key_bitstring(
            threshold=float(threshold),
            rotation_count=int(rotation_count),
            rng=rng,
        )
