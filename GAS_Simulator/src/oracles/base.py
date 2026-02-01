from abc import ABC, abstractmethod
from qiskit import QuantumCircuit

class OracleBuilder(ABC):
    """
    GASの主要コンポーネントを提供するビルダー
    手法(QD/QSP)によって、目的関数の情報が「状態準備」に入るか「オラクル」に入るかが異なるため、
    両方をこのクラスで管理する。
    """

    @abstractmethod
    def build_oracle(self, n_key: int, **kwargs) -> QuantumCircuit:
        """
        Groverの反復でマーキングを行うオラクル回路 (O) を返す。
        QD: 符号反転を行うZゲート (目的関数情報は持たない)
        QSP: 目的関数をエンコードしたブロックエンコーディング回路
        """
        pass

    @abstractmethod
    def build_state_prep(self, n_key: int, obj_fun_str: str, state_prep_method, **kwargs) -> QuantumCircuit:
        """
        Groverの反復で使用する状態準備回路 (A) を返す。
        QD: 初期化 + 位相エンコード + IQFT (ここに目的関数情報が入る)
        QSP: 単なる初期状態準備 (Hadamardなど)

        :param state_prep_method: 初期状態生成クラス (Uniform, W, Dickeなど) のインスタンス
        """
        pass
