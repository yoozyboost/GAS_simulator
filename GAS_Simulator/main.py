import sys
import os

# srcディレクトリをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from batch_runner import BatchGASRunner

def main():
    # ==========================================
    # ここに実行・比較したい設定ファイル一覧を記述
    # ==========================================
    config_list = [
        # "config/qd_spin.yaml",   # シナリオ1
        "config/qsp_spin_circuit_uniform_legacy-blockencoding.yaml",  # シナリオ2
        "config/qsp_spin_circuit_uniform_optimized-blockencoding.yaml", 
        # "config/qsp_spin_circuit_dicke.yaml",  # シナリオ2
        # "config/qsp_spin_nocircuit.yaml",  # シナリオ3
        # "config/exp_another_test.yaml",
        # "config/analytical.yaml"  # Analyticalモードのベンチマーク
        # "config/qd_binary_circuit.yaml",
        # "config/qd_spin.yaml"
        # "config/qd_binary_nocircuit_uni/form.yaml",
        # "config/qd_binary_nocircuit_dicke.yaml",
    ]

    # バッチ実行クラスを呼び出す
    batch_runner = BatchGASRunner(config_list)
    batch_runner.run_all()

if __name__ == "__main__":
    main()