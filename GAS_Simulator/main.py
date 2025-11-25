import sys
import os
# srcディレクトリをパスに追加してモジュール検索できるようにする
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.runner import GASRunner

if __name__ == "__main__":
    runner = GASRunner("config/exp_test.yaml") # コメントアウト
    # runner = GASRunner("config/exp_qsp.yaml")    # 新しい設定を使用
    runner.run()
