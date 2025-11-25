import sys
import os
import pytest
from qiskit import QuantumCircuit

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from oracles import ORACLE_FACTORY

def test_qd_oracle_binary():
    builder = ORACLE_FACTORY["qd"]()
    # 2変数 x0 + x1, 値レジスタ3qubit
    qc = builder.build(n_key=2, obj_fun_str="x0 + x1", n_val=3, is_spin=False)
    
    # Check circuit size: n_key(2) + n_val(3) = 5
    assert qc.num_qubits == 5
    print("\nQD(Binary) Oracle Created successfully.")

def test_qd_oracle_spin():
    builder = ORACLE_FACTORY["qd"]()
    # 2変数, Spin
    qc = builder.build(n_key=2, obj_fun_str="x0 + x1", n_val=3, is_spin=True)
    assert qc.num_qubits == 5
    print("QD(Spin) Oracle Created successfully.")

def test_qsp_oracle():
    builder = ORACLE_FACTORY["qsp"]()
    # 2変数, 閾値 0.5
    qc = builder.build(n_key=2, obj_fun_str="x0 + x1", threshold=0.5, qsp_degree=10)
    
    # Check circuit size: n_key(2) + ancilla(1) = 3
    assert qc.num_qubits == 3
    print("QSP Oracle Created successfully.")

if __name__ == "__main__":
    try:
        test_qd_oracle_binary()
        test_qd_oracle_spin()
        test_qsp_oracle()
        print("All Oracle tests passed!")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()