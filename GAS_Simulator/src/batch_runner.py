import os
import traceback
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

from runner import GASRunner


class BatchGASRunner:
    def __init__(self, config_files):
        # main.py から渡される形式に合わせる
        if isinstance(config_files, str):
            config_files = [config_files]
        self.config_files = list(config_files)

        # バッチ実行用の親ディレクトリ作成
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.batch_dir = os.path.join("results", f"BatchRun_{timestamp}")
        os.makedirs(self.batch_dir, exist_ok=True)

        self.results = []

    def run_all(self):
        print(f"=== Starting Batch Execution: {len(self.config_files)} files ===")
        print(f"Results will be saved to: {self.batch_dir}\n")

        for cfg_path in self.config_files:
            try:
                runner = GASRunner(cfg_path, parent_dir=self.batch_dir)
                res_data = runner.run()

                if res_data is None or not isinstance(res_data, dict):
                    raise RuntimeError("runner.run() did not return a dict result")

                # 必須キーが欠けても落とさない
                if "name" not in res_data:
                    res_data["name"] = os.path.basename(cfg_path)

                self.results.append(res_data)
                print(f"-> Finished: {res_data['name']}\n")

            except Exception as e:
                print(f"-> Failed: {cfg_path}")
                print(str(e))
                traceback.print_exc()
                print("")

        if not self.results:
            print("No successful runs. Skipping comparison plots.")
            return

        self.plot_comparison_convergence()
        self.plot_comparison_cdf()

    def _as_float_array(self, x, default_len=None):
        """
        list / np.ndarray / None を受けて、np.ndarray(dtype=float) を返す。
        None の場合は default_len があれば zeros を返す。
        """
        if x is None:
            if default_len is None:
                return np.asarray([], dtype=float)
            return np.zeros(int(default_len), dtype=float)
        a = np.asarray(x, dtype=float)
        return a

    def plot_comparison_convergence(self):
        """収束履歴の比較プロット（入力揺れに強い実装）"""
        plt.figure(figsize=(10, 6))

        # 色はtab10を循環
        colors = plt.cm.tab10(np.linspace(0, 1, 10))

        global_opt_drawn = False

        for i, res in enumerate(self.results):
            conv = res.get("convergence", {})
            if not isinstance(conv, dict):
                continue

            # 旧/新の揺れを吸収
            x_raw = conv.get("iterations", None)
            y_raw = conv.get("mean", None)
            if y_raw is None:
                y_raw = conv.get("best", None)  # フォールバック

            std_raw = conv.get("std", None)

            # x が無ければ y の長さから作る
            if y_raw is None:
                continue

            y = self._as_float_array(y_raw)
            if x_raw is None:
                x = np.arange(len(y), dtype=float)
            else:
                x = self._as_float_array(x_raw)

            # std が無ければ 0
            std = self._as_float_array(std_raw, default_len=len(y))

            # 長さ合わせ
            L = min(len(x), len(y), len(std))
            if L == 0:
                continue
            x = x[:L]
            y = y[:L]
            std = std[:L]

            label = res.get("name", f"exp_{i}")
            color = colors[i % len(colors)]

            plt.plot(x, y, label=label, color=color, linewidth=2)
            # plt.fill_between(x, y - std, y + std, color=color, alpha=0.1)

            # global_opt は1本だけ描く
            if not global_opt_drawn:
                g = conv.get("global_opt", None)
                if g is None:
                    g = res.get("global_optimum", None)
                if g is not None:
                    try:
                        g = float(g)
                        plt.axhline(y=g, color="black", linestyle=":", label="Global Opt")
                        global_opt_drawn = True
                    except Exception:
                        pass

        plt.xlabel("Iteration")
        plt.ylabel("Objective Function Value")
        plt.title("Convergence Comparison")
        plt.legend()
        plt.grid(True, alpha=0.3)

        save_path = os.path.join(self.batch_dir, "comparison_convergence.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved comparison plot: {save_path}")

    def plot_comparison_cdf(self):
        """クエリCDFの比較プロット（入力揺れに強い実装）"""
        plt.figure(figsize=(10, 6))

        plotted_any = False

        for res in self.results:
            cdf = res.get("cdf", {})
            if not isinstance(cdf, dict):
                continue

            sorted_q_raw = cdf.get("sorted_queries", None)
            total_trials = cdf.get("total_trials", None)

            if sorted_q_raw is None or total_trials is None:
                continue

            try:
                total_trials = int(total_trials)
            except Exception:
                continue

            if total_trials <= 0:
                continue

            sorted_q = self._as_float_array(sorted_q_raw)
            n_success = len(sorted_q)
            if n_success == 0:
                continue

            y_vals = np.arange(1, n_success + 1, dtype=float) / float(total_trials)

            plt.step(sorted_q, y_vals, where="post", label=f"{res.get('name','exp')} ({n_success}/{total_trials})")
            plotted_any = True

        if not plotted_any:
            plt.close()
            print("No CDF data found. Skipping comparison CDF.")
            return

        plt.xlabel("Cumulative Queries")
        plt.ylabel("Success Probability")
        plt.title("Query Complexity CDF Comparison")
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.05)

        save_path = os.path.join(self.batch_dir, "comparison_cdf.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved comparison CDF: {save_path}")
