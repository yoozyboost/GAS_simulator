import sys
import os
from qiskit import QuantumCircuit

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from oracles import ORACLE_FACTORY
from state_prep import STATE_PREP_FACTORY


def test_qd_oracle_and_stateprep_binary():
    oracle_builder = ORACLE_FACTORY["qd"]()
    state_prep = STATE_PREP_FACTORY["uniform"]()

    n_key = 2
    n_val = 3

    qc_s = oracle_builder.build_state_prep(
        n_key=n_key,
        obj_fun_str="x0 + x1",
        state_prep_method=state_prep,
        n_val=n_val,
        is_spin=False,
        threshold=0.0,
    )
    qc_o = oracle_builder.build_oracle(
        n_key=n_key,
        n_val=n_val,
        is_spin=False,
        threshold=0.0,
    )

    assert qc_s.num_qubits == n_key + n_val
    assert qc_o.num_qubits == n_key + n_val


def test_qd_oracle_and_stateprep_spin():
    oracle_builder = ORACLE_FACTORY["qd"]()
    state_prep = STATE_PREP_FACTORY["uniform"]()

    n_key = 2
    n_val = 3

    qc_s = oracle_builder.build_state_prep(
        n_key=n_key,
        obj_fun_str="x0 + x1",
        state_prep_method=state_prep,
        n_val=n_val,
        is_spin=True,
        threshold=0.0,
    )
    qc_o = oracle_builder.build_oracle(
        n_key=n_key,
        n_val=n_val,
        is_spin=True,
        threshold=0.0,
    )

    assert qc_s.num_qubits == n_key + n_val
    assert qc_o.num_qubits == n_key + n_val


def test_qsp_oracle_and_stateprep():
    oracle_builder = ORACLE_FACTORY["qsp"]()
    state_prep = STATE_PREP_FACTORY["uniform"]()

    n_key = 2

    qc_s = oracle_builder.build_state_prep(
        n_key=n_key,
        obj_fun_str="x0 + x1",
        state_prep_method=state_prep,
    )
    qc_o = oracle_builder.build_oracle(
        n_key=n_key,
        obj_fun_str="x0 + x1",
        threshold=0.5,
        qsp_degree=9,
    )

    assert qc_s.num_qubits == n_key + 1
    assert qc_o.num_qubits == n_key + 1
