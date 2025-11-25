from qiskit import QuantumCircuit
from math import ceil, sqrt, atan
from .base import StatePrepBuilder

class WStateBuilder(StatePrepBuilder):
    def build(self, n_qubits: int, **kwargs) -> QuantumCircuit:
        """
        W状態 |W_n> = 1/sqrt(n) * (|10...0> + ... + |0...01>) を生成
        """
        return self._construct_w_state(n_qubits)

    def _construct_w_state(self, n: int) -> QuantumCircuit:
        # 元の wstate.py のロジック
        qc = QuantumCircuit(n, name="WState")
        if n == 0:
            return qc
        
        qc.x(0)
        count = 1
        queue = [(n // 2, n, 0)]

        while len(queue):
            a, b, control = queue.pop(0)
            if b == 1:
                continue

            theta = 2 * atan(sqrt((b - a) / a))
            qc.cry(theta, control, count)
            qc.cx(count, control)

            queue.append(((b // 2) // 2, b // 2, control))
            queue.append((ceil(b / 2) // 2, ceil(b / 2), count))
            count += 1

        return qc