import os
import yaml
import csv
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from math import ceil, sqrt, isclose

from core.gas_model import GASModel
from utils.math_tools import evaluate_obj, calc_obj_fun_distribution
from utils.visualization import save_circuit_diagrams, save_circuit_metrics

class GASRunner:
    def __init__(self, config_path: str, parent_dir: str = None):
        self.config_path = config_path
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
            
        exp_name = self.config.get('experiment_name', 'exp')
        
        # 親ディレクトリが指定されていればその下に、なければ日時フォルダを作成
        if parent_dir:
            self.result_dir = os.path.join(parent_dir, exp_name)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.result_dir = os.path.join("results", f"{timestamp}_{exp_name}")
            
        os.makedirs(self.result_dir, exist_ok=True)
        
        with open(os.path.join(self.result_dir, "settings.yaml"), 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f)
            
        self.model = GASModel(self.config)

    def run(self):
        print(f"Starting Experiment: {self.config['experiment_name']}")
        
        if self.model.backend_type != 'analytical':
            # print("Generating circuit visualizations...") # ログ抑制したい場合はコメントアウト
            save_circuit_diagrams(self.model, self.result_dir)
            save_circuit_metrics(self.model, self.result_dir)

        # 真の最小値計算
        vals, _, _ = calc_obj_fun_distribution(self.model.n_key, self.model.obj_fun_str, self.model.var_type)
        global_opt_y = vals[0]

        num_trials = self.config.get('execution', {}).get('num_trials', 1)
        max_iter = self.config['simulation'].get('max_iterations', 30)

        all_histories_best = []
        all_histories_sample = []
        queries_to_success = []
        success_count = 0

        print(f"Running {num_trials} trials for {self.config['experiment_name']}...")

        for i in range(num_trials):
            h_best, h_sample, q_success = self._run_single_trial(i, max_iter, global_opt_y)
            all_histories_best.append(h_best)
            all_histories_sample.append(h_sample)
            if q_success is not None:
                queries_to_success.append(q_success)
                success_count += 1
        
        # 個別シナリオの保存も行う
        self._save_convergence_statistics(all_histories_best, all_histories_sample, max_iter, global_opt_y)
        self._save_query_cdf(queries_to_success, num_trials)
        
        # 収束データ
        data_best = np.array(all_histories_best)
        mean_best = np.mean(data_best, axis=0)
        std_best = np.std(data_best, axis=0)
        
        # CDFデータ
        sorted_queries = np.sort(queries_to_success)
        
        result_data = {
            "name": self.config['experiment_name'],
            "convergence": {
                "iterations": np.arange(len(mean_best)),
                "mean": mean_best,
                "std": std_best,
                "global_opt": global_opt_y
            },
            "cdf": {
                "sorted_queries": sorted_queries,
                "success_count": success_count,
                "total_trials": num_trials
            }
        }
        return result_data

    def _run_single_trial(self, trial_id, max_iter, global_opt_y):
        n_key = self.model.n_key
        current_x = "".join([str(np.random.randint(0, 2)) for _ in range(n_key)])
        current_y = evaluate_obj(self.model.obj_fun_str, current_x, self.model.var_type)
        threshold = current_y
        m = 1
        no_improvement_count = 0
        history_best = [current_y]
        history_sample = [current_y]
        cumulative_queries = 0
        time_to_solution = None
        if isclose(current_y, global_opt_y, abs_tol=1e-9): time_to_solution = 0
        for it in range(1, max_iter + 1):
            rotation_count = int(ceil(np.random.uniform(0, m)))
            cumulative_queries += rotation_count
            if self.model.backend_type == 'analytical':
                new_x = self.model.run_step_analytical(threshold, rotation_count)
            else:
                new_x = self.model.run_step_circuit(threshold, rotation_count)
            new_y = evaluate_obj(self.model.obj_fun_str, new_x, self.model.var_type)
            history_sample.append(new_y)
            if time_to_solution is None and isclose(new_y, global_opt_y, abs_tol=1e-9):
                time_to_solution = cumulative_queries
            if new_y < current_y:
                current_x, current_y, threshold, m, no_improvement_count = new_x, new_y, new_y, 1, 0
            else:
                m = min(m * 1.14, sqrt(2**n_key))
                no_improvement_count += 1
            history_best.append(current_y)
            if no_improvement_count > 10: break
        final_best = history_best[-1]
        if len(history_best) <= max_iter:
            padding = max_iter - len(history_best) + 1
            history_best.extend([final_best] * padding)
            history_sample.extend([final_best] * padding)
        return history_best, history_sample, time_to_solution

    def _save_convergence_statistics(self, all_histories_best, all_histories_sample, max_iter, global_opt_y):
        data_best = np.array(all_histories_best)
        data_sample = np.array(all_histories_sample)
        mean_best = np.mean(data_best, axis=0)
        std_best = np.std(data_best, axis=0)
        mean_sample = np.mean(data_sample, axis=0)
        std_sample = np.std(data_sample, axis=0)
        iterations = np.arange(len(mean_best))
        csv_path = os.path.join(self.result_dir, "convergence_stats.csv")
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Iteration", "Mean_Best_Value", "Std_Best", "Mean_Sampled_Value", "Std_Sample"])
            for i in range(len(iterations)):
                writer.writerow([iterations[i], mean_best[i], std_best[i], mean_sample[i], std_sample[i]])
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, mean_best, label='Avg Best Value', color='blue')
        plt.fill_between(iterations, mean_best - std_best, mean_best + std_best, color='blue', alpha=0.2)
        plt.plot(iterations, mean_sample, label='Avg Sampled Value', color='green', linestyle='--', alpha=0.5)
        plt.axhline(y=global_opt_y, color='red', linestyle='--', label='True Minimum')
        plt.xlabel('Iteration')
        plt.ylabel('Objective Value')
        plt.title(f'Convergence: {self.config["experiment_name"]}')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.result_dir, 'convergence_avg.png'))
        plt.close()

    def _save_query_cdf(self, queries_to_success, total_trials):
        sorted_queries = np.sort(queries_to_success)
        n_success = len(sorted_queries)
        if n_success == 0: return
        y_vals = np.arange(1, n_success + 1) / total_trials
        csv_path = os.path.join(self.result_dir, "query_cdf.csv")
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Queries", "Success_Probability"])
            for q, p in zip(sorted_queries, y_vals):
                writer.writerow([q, p])
        plt.figure(figsize=(10, 6))
        plt.step(sorted_queries, y_vals, where='post')
        plt.xlabel('Cumulative Queries')
        plt.ylabel('Probability')
        plt.title(f'Query CDF: {self.config["experiment_name"]}')
        plt.grid(True)
        plt.savefig(os.path.join(self.result_dir, 'query_cdf.png'))
        plt.close()