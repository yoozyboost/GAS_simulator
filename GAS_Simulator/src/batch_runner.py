import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from runner import GASRunner

class BatchGASRunner:
    def __init__(self, config_files):
        self.config_files = config_files
        
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
                # runnerにバッチディレクトリを渡して、その中に個別フォルダを作らせる
                runner = GASRunner(cfg_path, parent_dir=self.batch_dir)
                res_data = runner.run()
                self.results.append(res_data)
                print(f"-> Finished: {res_data['name']}\n")
            except Exception as e:
                print(f"-> Failed: {cfg_path}")
                print(e)
                import traceback
                traceback.print_exc()

        # 全部終わったら比較グラフを作成
        if self.results:
            self.plot_comparison_convergence()
            self.plot_comparison_cdf()

    def plot_comparison_convergence(self):
        """収束履歴の比較プロット"""
        plt.figure(figsize=(10, 6))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.results)))
        
        for i, res in enumerate(self.results):
            conv = res['convergence']
            x = conv['iterations']
            y = conv['mean']
            std = conv['std']
            label = res['name']
            color = colors[i]
            
            plt.plot(x, y, label=label, color=color, linewidth=2)
            plt.fill_between(x, y - std, y + std, color=color, alpha=0.1)
            
            # Global Optラインは最初の1つだけ描画するか、全部同じなら1本引く
            if i == 0:
                plt.axhline(y=conv['global_opt'], color='black', linestyle=':', label='Global Opt')

        plt.xlabel('Iteration')
        plt.ylabel('Objective Function Value')
        plt.title('Convergence Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        save_path = os.path.join(self.batch_dir, 'comparison_convergence.png')
        plt.savefig(save_path)
        plt.close()
        print(f"Saved comparison plot: {save_path}")

    def plot_comparison_cdf(self):
        """クエリCDFの比較プロット"""
        plt.figure(figsize=(10, 6))
        
        for res in self.results:
            cdf = res['cdf']
            sorted_q = cdf['sorted_queries']
            n_success = len(sorted_q)
            total = cdf['total_trials']
            
            if n_success == 0:
                continue
                
            y_vals = np.arange(1, n_success + 1) / total
            
            # Step plot
            plt.step(sorted_q, y_vals, where='post', label=f"{res['name']} ({n_success}/{total})")

        plt.xlabel('Cumulative Queries')
        plt.ylabel('Success Probability')
        plt.title('Query Complexity CDF Comparison')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.05)
        
        save_path = os.path.join(self.batch_dir, 'comparison_cdf.png')
        plt.savefig(save_path)
        plt.close()
        print(f"Saved comparison CDF: {save_path}")