from abc import ABC, abstractmethod
from qiskit import QuantumCircuit

class StatePrepBuilder(ABC):
    @abstractmethod
    def build(self, n_qubits, **kwargs) -> QuantumCircuit:
        pass
