import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from oracles import ORACLE_FACTORY
from state_prep import STATE_PREP_FACTORY
from utils.math_tools import evaluate_obj, calc_obj_fun_distribution

class GASModel:
    def __init__(self, config):
        self.config = config
        self.n_key = config['problem']['n_key']
        self.obj_fun_str = config['problem']['objective_function']
        self.var_type = config['problem'].get('variable_type', 'binary')
        self.method = config['algorithm']['method'] 
        self.init_state_name = config['algorithm'].get('initial_state', 'uniform')
        self.backend_type = config['simulation']['backend']
        self.n_val = config['problem'].get('n_val', 5) if self.method == 'qd' else 0
        self.analytical_data = None

    def _prepare_analytical_data(self):
        if self.analytical_data is None:
            vals, cum_counts, memo = calc_obj_fun_distribution(
                self.n_key, self.obj_fun_str, self.var_type
            )
            self.analytical_data = {
                'values': vals,
                'cum_counts': cum_counts,
                'memo': memo,
                'total_count': cum_counts[-1]
            }

    def construct_circuit(self, threshold: float, rotation_count: int) -> QuantumCircuit:
        
        # 1. コンポーネントのビルド
        oracle_builder = ORACLE_FACTORY[self.method]()
        state_prep_method = STATE_PREP_FACTORY[self.init_state_name]()
        
        # kwargsの準備 (QD/QSP共通で渡すもの、個別で使うもの)
        build_kwargs = {
            'n_val': self.n_val,
            'is_spin': (self.var_type == 'spin'),
            'threshold': threshold,
            'obj_fun_str': self.obj_fun_str # QSPのOracle構築には必要
        }
        if 'qsp_params' in self.config['algorithm']:
            build_kwargs.update(self.config['algorithm']['qsp_params'])

        # --- A (State Prep) の構築 ---
        # 修正: build_state_prep は obj_fun_str を引数で取るため、kwargsからは除外して渡す
        prep_kwargs = {k: v for k, v in build_kwargs.items() if k != 'obj_fun_str'}
        qc_s = oracle_builder.build_state_prep(self.n_key, self.obj_fun_str, state_prep_method, **prep_kwargs)
        
        # --- O (Oracle) の構築 ---
        # build_oracle は kwargs から obj_fun_str を取る設計(QSP)なので、そのまま渡す
        qc_o = oracle_builder.build_oracle(self.n_key, **build_kwargs)
        
        # --- D (Diffuser) の構築 ---
        # D = U_s (2|0><0| - I) U_s^dagger
        n_qubits = qc_s.num_qubits
        qc_diff = QuantumCircuit(n_qubits, name="Diffuser")
        
        # A^dagger
        qc_diff.compose(qc_s.inverse(), inplace=True)
        
        # Reflection about |0...0>
        qc_diff.x(range(n_qubits))
        qc_diff.h(n_qubits - 1)
        qc_diff.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc_diff.h(n_qubits - 1)
        qc_diff.x(range(n_qubits))
        
        # A
        qc_diff.compose(qc_s, inplace=True)
        
        # --- 回路結合 ---
        total_qubits = max(qc_s.num_qubits, qc_o.num_qubits)
        total_qc = QuantumCircuit(total_qubits)
        
        # 初期状態: A |0>
        total_qc.compose(qc_s, inplace=True)
        
        # Grover Operator: G = A D A^dagger O
        grover_op = QuantumCircuit(total_qubits, name="G")
        grover_op.compose(qc_o, inplace=True)
        grover_op.compose(qc_diff, inplace=True)
        
        for _ in range(rotation_count):
            total_qc.compose(grover_op, inplace=True)
            
        return total_qc

    def get_components_for_visualization(self):
        """可視化用コンポーネント取得"""
        threshold = 0.0
        oracle_builder = ORACLE_FACTORY[self.method]()
        state_prep_method = STATE_PREP_FACTORY[self.init_state_name]()
        
        build_kwargs = {
            'n_val': self.n_val,
            'is_spin': (self.var_type == 'spin'),
            'threshold': threshold,
            'obj_fun_str': self.obj_fun_str
        }
        if 'qsp_params' in self.config['algorithm']:
            build_kwargs.update(self.config['algorithm']['qsp_params'])

        # 修正: ここでも kwargs から obj_fun_str を除外
        prep_kwargs = {k: v for k, v in build_kwargs.items() if k != 'obj_fun_str'}
        qc_s = oracle_builder.build_state_prep(self.n_key, self.obj_fun_str, state_prep_method, **prep_kwargs)
        
        # Oracle構築にはそのまま渡す
        qc_o = oracle_builder.build_oracle(self.n_key, **build_kwargs)
        
        qc_grover = self.construct_circuit(threshold, rotation_count=1)
        
        return {
            "state_prep": qc_s,
            "oracle": qc_o,
            "full_circuit_1step": qc_grover
        }

    def run_step_circuit(self, threshold: float, rotation_count: int):
        qc = self.construct_circuit(threshold, rotation_count)
        
        if self.backend_type == "statevector":
            qc.remove_final_measurements()
            sv = Statevector(qc)
            
            full_probs = sv.probabilities()
            key_probs = np.zeros(2**self.n_key)
            
            for i, p in enumerate(full_probs):
                if p == 0: continue
                key_idx = i & ((1 << self.n_key) - 1)
                key_probs[key_idx] += p
                
            key_probs /= np.sum(key_probs)
            idx = np.random.choice(len(key_probs), p=key_probs)
            bitstring = format(idx, f'0{self.n_key}b')
            return bitstring[::-1]
            
        elif self.backend_type == "qasm":
            return self._run_qasm(qc)

    def get_circuit_metrics(self, threshold: float, rotation_count: int):
        qc = self.construct_circuit(threshold, rotation_count)
        decomposed = qc.decompose()
        return {
            'depth': decomposed.depth(),
            'gate_count': dict(decomposed.count_ops()),
            'qubits': qc.num_qubits
        }

    def _run_qasm(self, qc):
        from qiskit import transpile
        from qiskit_aer import Aer
        qc.measure_all()
        backend = Aer.get_backend('qasm_simulator')
        job = backend.run(transpile(qc, backend), shots=self.config['simulation'].get('shots', 1024))
        counts = job.result().get_counts()
        key_counts = {}
        for k, v in counts.items():
            key_part = k[-self.n_key:] 
            key_counts[key_part] = key_counts.get(key_part, 0) + v
        measured_state = max(key_counts, key=key_counts.get)
        return measured_state[::-1]
    
    def run_step_analytical(self, threshold: float, rotation_count: int):
        self._prepare_analytical_data()
        
        vals = self.analytical_data['values']
        cum_counts = self.analytical_data['cum_counts']
        total_N = self.analytical_data['total_count']
        
        import bisect
        idx = bisect.bisect_left(vals, threshold)
        
        if idx == 0:
            M = 0 
        else:
            M = cum_counts[idx - 1]
            
        theta = np.arcsin(np.sqrt(M / total_N))
        prob_success = np.sin((2 * rotation_count + 1) * theta) ** 2
        
        if np.random.random() < prob_success and M > 0:
            rand_idx = np.random.randint(1, M + 1)
            val_idx = bisect.bisect_left(cum_counts, rand_idx)
            sampled_val = vals[val_idx]
        else:
            if total_N == M:
                 rand_idx = np.random.randint(1, M + 1)
                 val_idx = bisect.bisect_left(cum_counts, rand_idx)
                 sampled_val = vals[val_idx]
            else:
                rand_idx = np.random.randint(M + 1, total_N + 1)
                val_idx = bisect.bisect_left(cum_counts, rand_idx)
                sampled_val = vals[val_idx]
                
        candidates = self.analytical_data['memo'][sampled_val]
        chosen_int = np.random.choice(candidates)
        bitstring = format(chosen_int, f'0{self.n_key}b')
        return bitstring[::-1]