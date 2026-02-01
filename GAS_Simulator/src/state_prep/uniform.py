from qiskit import QuantumCircuit
from .base import StatePrepBuilder

class UniformStateBuilder(StatePrepBuilder):
    def build(self, n_qubits: int, **kwargs) -> QuantumCircuit:
        qc = QuantumCircuit(n_qubits, name="Uniform")
        qc.h(range(n_qubits))
        return qc
