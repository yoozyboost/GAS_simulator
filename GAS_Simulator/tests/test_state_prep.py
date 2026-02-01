import sys
import os
import pytest
from qiskit.quantum_info import Statevector
from math import sqrt, isclose

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from state_prep import STATE_PREP_FACTORY

def test_uniform_state():
    n = 3
    builder = STATE_PREP_FACTORY["uniform"]()
    qc = builder.build(n)
    sv = Statevector(qc)
    # すべての振幅が 1/sqrt(8)
    for amp in sv.data:
        assert isclose(abs(amp), 1/sqrt(2**n))

def test_w_state():
    n = 3
    builder = STATE_PREP_FACTORY["w_state"]()
    qc = builder.build(n)
    sv = Statevector(qc).data
    # |001>, |010>, |100> の係数が 1/sqrt(3)
    target_indices = [1, 2, 4]
    for idx in target_indices:
        assert isclose(sv[idx].real, 1/sqrt(n), abs_tol=1e-5)

def test_dicke_state():
    n = 4
    k = 2
    builder = STATE_PREP_FACTORY["dicke"]()
    qc = builder.build(n, k=k)
    sv = Statevector(qc).data
    # nCk = 4C2 = 6通りの状態が等確率
    expected_amp = sqrt(1/6)

    # ハミング重みが2のインデックスを探してチェック
    count = 0
    for i, amp in enumerate(sv):
        if bin(i).count('1') == k:
            assert isclose(amp.real, expected_amp, abs_tol=1e-5)
            count += 1
    assert count == 6

if __name__ == "__main__":
    test_uniform_state()
    test_w_state()
    test_dicke_state()
    print("All State Prep tests passed!")
