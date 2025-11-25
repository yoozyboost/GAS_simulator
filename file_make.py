import os
from pathlib import Path

def create_project_structure():
    # プロジェクトのルートディレクトリ名
    root_dir = "GAS_Simulator"
    
    # 作成するディレクトリ構成
    directories = [
        "config",
        "results",
        "tests",
        "src",
        "src/core",         # GASのメインロジック
        "src/oracles",      # オラクル (QD, QSP)
        "src/state_prep",   # 初期状態 (Uniform, W, Dicke)
        "src/utils",        # 数学ツール, 可視化
    ]

    # 作成するファイルとその初期内容（コメント等）
    files = {
        "requirements.txt": 
            "qiskit\nqiskit-aer\nnumpy\nscipy\nsympy\nmatplotlib\npyyaml\ntqdm\npytest\n# pyqsp  # QSPの角度計算に使用予定\n",

        "main.py": 
            """import sys
import os
# srcディレクトリをパスに追加してモジュール検索できるようにする
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.runner import GASRunner

if __name__ == "__main__":
    # ここに引数処理などを記述予定
    print("GAS Simulator Initialized.")
    # runner = GASRunner("config/exp_test.yaml")
    # runner.run()
""",

        "config/exp_test.yaml": 
            """experiment_name: "test_run_001"
problem:
  n_key: 4
  objective_function: "x0 + x1 - x2 - x3"
  variable_type: "binary" 

algorithm:
  method: "qd" # or "qsp"
  initial_state: "uniform" # "w_state", "dicke"

simulation:
  backend: "statevector"
  shots: 1024
""",

        "src/__init__.py": "",
        "src/runner.py": 
            """class GASRunner:
    def __init__(self, config_path):
        self.config_path = config_path
        pass

    def run(self):
        print(f"Running simulation with {self.config_path}")
""",

        # --- CORE ---
        "src/core/__init__.py": "",
        "src/core/gas_model.py": 
            """# GASのアルゴリズム骨子（Grover Operatorの反復など）をここに記述
class GASModel:
    pass
""",

        # --- ORACLES ---
        "src/oracles/__init__.py": "",
        "src/oracles/base.py": 
            """from abc import ABC, abstractmethod
from qiskit import QuantumCircuit

class OracleBuilder(ABC):
    @abstractmethod
    def build(self, n_key, obj_fun_str, **kwargs) -> QuantumCircuit:
        pass
""",
        "src/oracles/qd_oracle.py": "# 従来のQD-GASオラクル実装（ilquantum.py由来）\n",
        "src/oracles/qsp_oracle.py": "# QSP-GASオラクル実装（QSP_ilquantum.py由来）\n",

        # --- STATE PREP ---
        "src/state_prep/__init__.py": "",
        "src/state_prep/base.py": 
            """from abc import ABC, abstractmethod
from qiskit import QuantumCircuit

class StatePrepBuilder(ABC):
    @abstractmethod
    def build(self, n_qubits, **kwargs) -> QuantumCircuit:
        pass
""",
        "src/state_prep/uniform.py": "# 一様重ね合わせ (Hadamard layer)\n",
        "src/state_prep/w_state.py": "# W状態 (wstate.py由来)\n",
        "src/state_prep/dicke_state.py": "# Dicke状態 (dicke.py由来)\n",

        # --- UTILS ---
        "src/utils/__init__.py": "",
        "src/utils/math_tools.py": "# Sympy解析、変数変換、評価関数など\n",
        "src/utils/visualization.py": "# プロット、回路図描画など\n",
    }

    base_path = Path(root_dir)

    # ディレクトリ作成
    for d in directories:
        dir_path = base_path / d
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {dir_path}")

    # ファイル作成
    for filename, content in files.items():
        file_path = base_path / filename
        if not file_path.exists():
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"Created file: {file_path}")
        else:
            print(f"Skipped (already exists): {file_path}")

    print("\nProject structure generation complete!")
    print(f"Root directory: {os.path.abspath(root_dir)}")

if __name__ == "__main__":
    create_project_structure()