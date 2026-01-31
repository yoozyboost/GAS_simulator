import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from core.gas_model import GASModel
from qiskit.quantum_info import Statevector


def _key_probs_from_statevector_qsp(qc, n_key: int) -> np.ndarray:
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


def test_qsp_nocircuit_matches_circuit_distribution_small():
    cfg = {
        "experiment_name": "pytest_qsp_match",
        "problem": {
            "n_key": 2,
            "objective_function": "x0 + x1 + x0*x1",
            "variable_type": "spin",
        },
        "algorithm": {
            "method": "qsp",
            "initial_state": "uniform",
            "qsp_params": {
                "qsp_degree": 9,
            },
        },
        "simulation": {
            "backend": "statevector",
            "convergence_backend": "nocircuit",
            "max_iterations": 1,
            "seed": 0,
        },
        "execution": {"num_trials": 1},
        "circuit_assets": {"build": True},
    }

    model = GASModel(cfg)

    threshold = 0.0
    rotation_count = 1

    from qiskit import QuantumCircuit

    qc_s, qc_o, qc_diff, _ = model.construct_circuit(threshold, rotation_count)
    nq = qc_s.num_qubits
    qc_full = QuantumCircuit(nq)
    qc_full.append(qc_s, range(nq))
    for _ in range(rotation_count):
        qc_full.append(qc_o, range(nq))
        qc_full.append(qc_diff, range(nq))

    p_circ = _key_probs_from_statevector_qsp(qc_full, n_key=model.n_key)


    rng = np.random.default_rng(0)
    n_shots = 2000
    counts = np.zeros(2**model.n_key, dtype=np.int64)
    for _ in range(n_shots):
        b = model.run_step_nocircuit(threshold, rotation_count, rng=rng)
        idx = int(b[::-1], 2)
        counts[idx] += 1
    p_nc = counts / float(n_shots)

    assert _tvd(p_nc, p_circ) < 0.08
