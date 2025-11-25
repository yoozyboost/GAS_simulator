import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT
from .base import OracleBuilder
from utils.math_tools import str_to_sympy

class QDOracleBuilder(OracleBuilder):
    
    def build_oracle(self, n_key: int, **kwargs) -> QuantumCircuit:
        """
        QD-GASのオラクル: 値レジスタの先頭(符号ビット)にZゲートをかけるだけ
        """
        n_val = kwargs.get('n_val', 5)
        
        # 回路サイズは n_key + n_val
        qc = QuantumCircuit(n_key + n_val, name="QD_Oracle")
        
        # 値レジスタは n_key から始まる。2の補数表現のMSB(符号ビット)は n_key 番目
        # ilquantumの実装: O.z(n_key)
        qc.z(n_key)
        
        return qc

    def build_state_prep(self, n_key: int, obj_fun_str: str, state_prep_method, **kwargs) -> QuantumCircuit:
        """
        QD-GASの状態準備 (Ay): 初期化 + 位相回転 + IQFT
        ilquantum.py の constructAy に相当
        """
        n_val = kwargs.get('n_val', 5)
        is_spin = kwargs.get('is_spin', False)
        threshold = kwargs.get('threshold', 0.0) # QDでは閾値シフトをここで行う

        # 式のパース: f(x) - threshold
        expr = str_to_sympy(f"({obj_fun_str}) - ({threshold})")
        polydict = expr.as_poly().as_dict()

        # 1. 基本的な重ね合わせ状態の作成 (Key + Val)
        # ilquantumでは constructAy の冒頭で `A.h(range(n_key + n_val))` としている
        # Key側は W状態などを許容し、Val側は必ずHとする
        
        qc = QuantumCircuit(n_key + n_val, name="QD_StatePrep")
        
        # Key部分の初期化 (Uniform, W, Dickeなど)
        qc_key_init = state_prep_method.build(n_key)
        qc.compose(qc_key_init, range(n_key), inplace=True)
        
        # Value部分の初期化 (常にHadamard)
        qc.h(range(n_key, n_key + n_val))
        
        qc.barrier()

        # 2. 位相回転 (Phase encoding)
        if not is_spin:
            # Binary Variables
            for (ps, k) in polydict.items():
                k = float(k)
                i_nonzero = np.nonzero(ps)[0]
                n_nonzero = len(i_nonzero)
                
                if n_nonzero == 0: # 定数項
                    for v in range(n_val):
                        qc.p(k * np.pi / 2**(n_val - 1 - v), n_key + v)
                else:
                    for v in range(n_val):
                        qc.mcp(k * np.pi / 2**(n_val - 1 - v), list(i_nonzero), n_key + v)
        else:
            # Spin Variables
            for (ps, k) in polydict.items():
                k = float(k)
                i_nonzero = np.nonzero(ps)[0]
                n_nonzero = len(i_nonzero)

                if n_nonzero == 0:
                    for v in range(n_val):
                        qc.rz(k * np.pi / 2**(n_val - 1 - v), n_key + v)
                else:
                    for v in range(n_val):
                        # Parity check -> Rz -> Uncompute
                        for i in i_nonzero:
                            qc.cx(i, n_key + v)
                        qc.rz(k * np.pi / 2**(n_val - 1 - v), n_key + v)
                        for i in i_nonzero:
                            qc.cx(i, n_key + v)
        
        qc.barrier()

        # 3. Inverse QFT
        iqft = QFT(n_val, do_swaps=False).inverse().reverse_bits()
        qc.append(iqft, range(n_key, n_key + n_val))

        return qc