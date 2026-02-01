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

        #QSPで従来のブロック埋め込み手法と効率的なブロック埋め込み手法を用いたときの比較
        "config/qsp_spin_circuit_uniform_legacy-blockencoding.yaml", 
        "config/qsp_spin_circuit_uniform_optimized-blockencoding.yaml",

        #QDの回路なしシミュレーション　一様重ね合わせとDicke状態を初期状態としたときの比較
        # "config/qd_binary_nocircuit_uniform.yaml",
        # "config/qd_binary_nocircuit_dicke.yaml",
    ]

    # バッチ実行クラスを呼び出す
    batch_runner = BatchGASRunner(config_list)
    batch_runner.run_all()

if __name__ == "__main__":
    main()
