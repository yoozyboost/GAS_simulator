import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from core.gas_model import GASModel
from qiskit.quantum_info import Statevector


def _key_probs_from_statevector(qc, n_key: int) -> np.ndarray:
    sv = Statevector(qc)
    full_probs = sv.probabilities()
    key_probs = np.zeros(2**n_key, dtype=np.float64)
    for i, p in enumerate(full_probs):
        if p == 0.0:
            continue
        key_idx = i & ((1 << n_key) - 1)
        key_probs[key_idx] += float(p)
    key_probs /= key_probs.sum()
    return key_probs


def _tvd(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * float(np.sum(np.abs(p - q)))


def test_qd_fejer_nocircuit_matches_circuit_distribution_small():
    """
    小サイズで、回路なしFejérの経験分布が回路の厳密分布に概ね近いことを確認する。
    """
    cfg = {
        "experiment_name": "pytest_qd_match",
        "problem": {
            "n_key": 2,
            "n_val": 5,
            "objective_function": "x0 + 2*x1",
            "variable_type": "binary",
        },
        "algorithm": {
            "method": "qd",
            "initial_state": "uniform",
        },
        "simulation": {
            "backend": "statevector",
            "convergence_backend": "nocircuit",
            "qd_nocircuit_mode": "fejer",
            "max_iterations": 1,
            "seed": 0,
        },
        "execution": {"num_trials": 1},
        "circuit_assets": {"build": True},
    }

    model = GASModel(cfg)

    threshold = 1.0
    rotation_count = 1

    # 回路のキー分布（厳密）
    from qiskit import QuantumCircuit

    qc_s, qc_o, qc_diff, _ = model.construct_circuit(threshold, rotation_count)
    nq = qc_s.num_qubits
    qc_full = QuantumCircuit(nq)
    qc_full.append(qc_s, range(nq))
    for _ in range(rotation_count):
        qc_full.append(qc_o, range(nq))
        qc_full.append(qc_diff, range(nq))

    p_circ = _key_probs_from_statevector(qc_full, n_key=model.n_key)


    # 回路なしの経験分布
    rng = np.random.default_rng(0)
    n_shots = 4000
    counts = np.zeros(2**model.n_key, dtype=np.int64)
    for _ in range(n_shots):
        b = model.run_step_nocircuit(threshold, rotation_count, rng=rng)
        idx = int(b[::-1], 2)  # bはx0先頭のため、整数化は反転してMSB表現へ
        counts[idx] += 1
    p_nc = counts / float(n_shots)

    # 近似なので緩め
    assert _tvd(p_nc, p_circ) < 0.15
