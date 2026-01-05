from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import Permutation, RYGate
from math import sqrt, acos, comb
from .base import StatePrepBuilder

class DickeStateBuilder(StatePrepBuilder):
    def build(self, n_qubits: int, **kwargs) -> QuantumCircuit:
        """
        Dicke状態 |D^n_k> を生成
        kwargs['k']: Hamming weight (1の数)
        """
        k = kwargs.get('k')
        if k is None:
            raise ValueError("Dicke state requires parameter 'k' (hamming weight).")
        
        qc = self._construct_dicke_state(n_qubits, k)
        qc.name = f"Dicke(n={n_qubits},k={k})"
        return qc

    def _construct_dicke_state(self, n: int, k: int) -> QuantumCircuit:
        if not k <= n:
            raise ValueError("k must be <= n")
        
        # 最適化: k > n/2 の場合は反転させる
        if 2 * k > n and k != n:
            qc = self._construct_dicke_state(n, n - k)
            qc.x(range(n))
            return qc

        qc = QuantumCircuit(n)
        qc.x(range(k))

        dsr = self._construct_dicke_state_recursion(n, k)
        qc.compose(dsr, inplace=True)
        return qc

    def _construct_dicke_state_recursion(self, n: int, k: int) -> QuantumCircuit:
        qc = QuantumCircuit(n)
        if n == k:
            return qc

        m = min(n - k, n // 2)
        wbd = self._construct_weight_distribution_block(n, m, k)
        qc.compose(wbd, inplace=True)

        if n - m <= k:
            dsu = self._construct_dicke_state_unitary(n - m, n - m)
            qc.compose(dsu, reversed(range(n - m)), inplace=True)
        else:
            ds = self._construct_dicke_state_recursion(n - m, k)
            qc.compose(ds, range(n - m), inplace=True)

        if m <= k:
            dsu = self._construct_dicke_state_unitary(m, m)
            qc.compose(dsu, reversed(range(n - m, n)), inplace=True)
        else:
            ds = self._construct_dicke_state_recursion(m, k)
            qc.compose(ds, range(n - m, n), inplace=True)

        return qc

    def _construct_weight_distribution_block(self, n: int, m: int, k: int) -> QuantumCircuit:
        qr1 = QuantumRegister(k)
        qr2 = QuantumRegister(min(k, m))
        qc = QuantumCircuit(qr1)
        
        if n - m - k > 0:
            qc.add_register(QuantumRegister(n - m - k))
        qc.add_register(qr2)
        if m - k > 0:
            qc.add_register(QuantumRegister(m - k))

        # step (1)
        for i in range(k - 1):
            qc.cx(qr1[i + 1], qr1[i])
        qc.barrier()

        # step (2)
        for l in range(1, k + 1):
            x_vals = [comb(m, i) * comb(n - m, l - i) for i in range(l + 1)]
            s = sum(x_vals)
            for i in range(min(l, qr2.size)):
                angle = self._convert_angle(sqrt(x_vals[i] / s))
                if i == 0:
                    qc.cry(angle, qr1[l - 1], qr2[i])
                else:
                    qc.append(RYGate(angle).control(2), [qr1[l - 1], qr2[i - 1], qr2[i]])
                s -= x_vals[i]
            qc.barrier()
        qc.barrier()

        # step (3)
        for i in reversed(range(k - 1)):
            qc.cx(qr1[i + 1], qr1[i])
        qc.barrier()

        # step (4)
        if qr2.size >= k:
            qc.cx(qr2[k - 1], qr1[k - 1])
        for l in reversed(range(min(k - 1, qr2.size))):
            qc.append(
                Permutation(k - l, [i + 1 for i in range(k - l - 1)] + [0]).control(1),
                [qr2[l]] + qr1[l:k][:] # list slice fix for qiskit registers
            )
            qc.cx(qr2[l], qr1[k - 1])
        qc.barrier()
        return qc

    def _construct_dicke_state_unitary(self, n: int, k: int) -> QuantumCircuit:
        qc = QuantumCircuit(n)
        if n == 1 and k == 1:
            return qc
        
        if n == k:
            scs = self._construct_split_cyclic_shift_unitary(n, k - 1)
            dsu = self._construct_dicke_state_unitary(n - 1, k - 1)
            qc.compose(scs, inplace=True)
            qc.compose(dsu, range(n - 1), inplace=True)
            return qc

        scs = self._construct_split_cyclic_shift_unitary(n, k)
        dsu = self._construct_dicke_state_unitary(n - 1, k)
        qc.compose(scs, inplace=True)
        qc.compose(dsu, range(n - 1), inplace=True)
        return qc

    def _construct_split_cyclic_shift_unitary(self, n: int, k: int) -> QuantumCircuit:
        qc = QuantumCircuit(n)
        qc.cx(n - 2, n - 1)
        qc.cry(self._convert_angle(sqrt(1 / n)), n - 1, n - 2)
        qc.cx(n - 2, n - 1)

        for l in range(2, k + 1):
            qc.cx(n - 1 - l, n - 1)
            qc.append(
                RYGate(self._convert_angle(sqrt(l / n))).control(2), 
                [n - 1, n - l, n - l - 1]
            )
            qc.cx(n - 1 - l, n - 1)
        qc.barrier()
        return qc

    def _convert_angle(self, val: float) -> float:
        # acosのドメインエラー回避のためのクリッピング
        val = max(-1.0, min(1.0, val))
        return 2 * acos(val)