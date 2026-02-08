import os
import yaml
import csv
import shutil
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
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        exp_name = self.config.get("experiment_name", "exp")

        # convergence_backend を省略したときのデフォルトを runner 側でも保証する
        sim_cfg = self.config.setdefault("simulation", {})
        if "convergence_backend" not in sim_cfg:
            sim_cfg["convergence_backend"] = "nocircuit"

        base_dir = parent_dir if parent_dir else "results"

        # 実験ディレクトリ（テストがここを見る）
        self.exp_dir = os.path.join(base_dir, exp_name)
        os.makedirs(self.exp_dir, exist_ok=True)

        # 実行ディレクトリ（履歴用）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.result_dir = os.path.join(self.exp_dir, "runs", timestamp)
        os.makedirs(self.result_dir, exist_ok=True)

        # settings.yaml は実験ディレクトリ直下と run ディレクトリの両方に残す
        with open(os.path.join(self.exp_dir, "settings.yaml"), "w", encoding="utf-8") as f:
            yaml.dump(self.config, f, allow_unicode=True)

        with open(os.path.join(self.result_dir, "settings.yaml"), "w", encoding="utf-8") as f:
            yaml.dump(self.config, f, allow_unicode=True)

        # 乱数
        seed = sim_cfg.get("seed", None)
        self.rng = np.random.default_rng(seed)

        self.model = GASModel(self.config)

    def run(self):
        print(f"Starting Experiment: {self.config['experiment_name']}")

        # 回路資産（図とメトリクス）は convergence_backend と無関係に作る
        build_assets = self.config.get("circuit_assets", {}).get("build", True)
        if build_assets:
            save_circuit_diagrams(self.model, self.result_dir)
            save_circuit_metrics(self.model, self.result_dir)

        # convergence_backend 優先で分岐（後方互換も確保）
        conv_backend = getattr(self.model, "convergence_backend", None)
        if conv_backend is None:
            conv_backend = self.config.get("simulation", {}).get("convergence_backend", None)

        # 真の最小値計算
        # - classical_exhaustive の場合は trial 側で 2^n 全列挙するため、ここでは重複計算を避ける
        global_opt_y = None
        if conv_backend != "classical_exhaustive":
            vals, _, _ = calc_obj_fun_distribution(self.model.n_key, self.model.obj_fun_str, self.model.var_type)
            global_opt_y = float(vals[0])

        num_trials = int(self.config.get("execution", {}).get("num_trials", 1))
        max_iter = int(self.config.get("simulation", {}).get("max_iterations", 30))

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

        # classical_exhaustive の場合、最終bestが真の最小値に一致する
        if global_opt_y is None:
            try:
                global_opt_y = float(min(h[-1] for h in all_histories_best if len(h) > 0))
            except Exception:
                global_opt_y = float("nan")

        # 統計保存
        self._save_convergence_statistics(all_histories_best, all_histories_sample, max_iter, global_opt_y)
        self._save_query_cdf(queries_to_success, num_trials)

        # 収束データ（batch_runner が参照する形式）
        data_best = np.array(all_histories_best, dtype=float)
        mean_best = np.mean(data_best, axis=0)
        std_best = np.std(data_best, axis=0, ddof=0)

        # CDFデータ
        sorted_queries = np.sort(np.array(queries_to_success, dtype=float)) if len(queries_to_success) > 0 else np.array([])

        # 実験ディレクトリ直下に「最新run」の成果物をコピー（テスト互換）
        self._copy_latest_artifacts_to_exp_dir()

        result_data = {
            "name": self.config["experiment_name"],
            "convergence_backend": getattr(
                self.model,
                "convergence_backend",
                self.config.get("simulation", {}).get("convergence_backend", None),
            ),
            "convergence": {
                "iterations": list(range(len(mean_best))),
                "mean": mean_best.tolist(),
                "std": std_best.tolist(),
                "global_opt": float(global_opt_y),
            },
            "cdf": {
                "sorted_queries": sorted_queries.tolist(),
                "success_count": int(success_count),
                "total_trials": int(num_trials),
            },
            "exp_dir": self.exp_dir,
            "result_dir": self.result_dir,
        }

        # サマリ表示（既存ログを維持）
        final_best_mean = float(mean_best[-1]) if len(mean_best) > 0 else float("nan")
        total_trials = num_trials
        p_success = float(success_count / total_trials) if total_trials > 0 else 0.0
        avg_queries = float(np.mean(sorted_queries)) if len(sorted_queries) > 0 else float("inf")

        print("=== Result Summary ===")
        cb = result_data["convergence_backend"]
        print(f"Convergence backend: {cb}")
        print(f"Final best (mean): {final_best_mean:.6f}")
        print(f"Global optimum:     {global_opt_y:.6f}")
        print(f"Success prob:       {p_success:.3f}")
        print(f"Avg queries:        {avg_queries}")

        return result_data

    def _run_single_trial(self, trial_id, max_iter, global_opt_y):
        n_key = self.model.n_key

        # convergence_backend 優先で分岐（後方互換も確保）
        conv_backend = getattr(self.model, "convergence_backend", None)
        if conv_backend is None:
            conv_backend = self.config.get("simulation", {}).get("convergence_backend", None)

        # ---- Classical Exhaustive Search baseline ----
        # 目的関数を 2^n 全列挙して最小値を求める。
        # クエリ数は「目的関数評価回数」と定義する。
        if conv_backend == "classical_exhaustive":
            return self._run_single_trial_classical_exhaustive(max_iter=max_iter, rng=self.rng)

        current_x = "".join([str(self.rng.integers(0, 2)) for _ in range(n_key)])
        current_y = float(evaluate_obj(self.model.obj_fun_str, current_x, self.model.var_type))

        threshold = current_y
        m = 1.0
        no_improvement_count = 0

        history_best = [current_y]
        history_sample = [current_y]

        cumulative_queries = 0
        time_to_solution = None
        if isclose(current_y, global_opt_y, abs_tol=1e-9):
            time_to_solution = 0

        for it in range(1, max_iter + 1):
            rotation_count = int(ceil(self.rng.uniform(0, m)))
            cumulative_queries += rotation_count

            if conv_backend == "nocircuit":
                new_x = self.model.run_step_nocircuit(threshold, rotation_count, rng=self.rng)
            else:
                new_x = self.model.run_step_circuit(threshold, rotation_count)

            new_y = float(evaluate_obj(self.model.obj_fun_str, new_x, self.model.var_type))

            history_sample.append(new_y)

            if time_to_solution is None and isclose(new_y, global_opt_y, abs_tol=1e-9):
                time_to_solution = cumulative_queries

            if new_y < current_y:
                current_x = new_x
                current_y = new_y
                threshold = new_y
                m = 1.0
                # no_improvement_count = 0
            else:
                m = min(m * 1.14, sqrt(2 ** n_key))
                # no_improvement_count += 1

            history_best.append(current_y)

            # if no_improvement_count > 10:
            #     break

        final_best = history_best[-1]

        # 長さを max_iter+1 にそろえる
        if len(history_best) <= max_iter:
            padding = max_iter - len(history_best) + 1
            history_best.extend([final_best] * padding)
            history_sample.extend([final_best] * padding)

        return history_best, history_sample, time_to_solution

    def _run_single_trial_classical_exhaustive(self, max_iter: int, rng: np.random.Generator):
        """古典的全探索（2^nの目的関数評価）を1trialとして実行する。

        列挙順は trial ごとに乱択化する。
        収束曲線は max_iter+1 点で出力する。
        iteration=0 は 1回目の評価後の best。
        iteration=t (t>=1) は追加のブロック評価後の best。
        """
        n_key = int(self.model.n_key)
        total_evals = int(2 ** n_key)

        # trial ごとに列挙順を乱択化する
        order = rng.permutation(total_evals)

        # 1回目の評価を iteration=0 に割り当てる
        i0 = int(order[0])
        bitstring = format(i0, f"0{n_key}b")[::-1]
        y0 = float(evaluate_obj(self.model.obj_fun_str, bitstring, self.model.var_type))

        best_y = y0
        last_best_update_eval = 1  # 1-indexed

        history_best = [best_y]
        history_sample = [y0]

        if max_iter <= 0 or total_evals == 1:
            return history_best, history_sample, 1

        remaining = total_evals - 1
        block = int(ceil(remaining / max_iter))

        eval_done = 1
        pos = 1

        for _ in range(1, max_iter + 1):
            if eval_done >= total_evals:
                history_best.append(best_y)
                history_sample.append(best_y)
                continue

            n_this = min(block, total_evals - eval_done)
            last_y = None

            for _j in range(n_this):
                i = int(order[pos])
                pos += 1

                bitstring = format(i, f"0{n_key}b")[::-1]
                y = float(evaluate_obj(self.model.obj_fun_str, bitstring, self.model.var_type))
                last_y = y

                eval_done += 1

                if y < best_y:
                    best_y = y
                    last_best_update_eval = eval_done

            history_best.append(best_y)
            history_sample.append(float(last_y) if last_y is not None else best_y)

        return history_best, history_sample, int(last_best_update_eval)

    def _save_convergence_statistics(self, all_histories_best, all_histories_sample, max_iter, global_opt_y):
        data_best = np.array(all_histories_best, dtype=float)
        data_sample = np.array(all_histories_sample, dtype=float)

        mean_best = np.mean(data_best, axis=0)
        std_best = np.std(data_best, axis=0, ddof=0)
        mean_sample = np.mean(data_sample, axis=0)
        std_sample = np.std(data_sample, axis=0, ddof=0)

        iterations = np.arange(len(mean_best))

        csv_path = os.path.join(self.result_dir, "convergence_stats.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Iteration", "Mean_Best_Value", "Std_Best", "Mean_Sampled_Value", "Std_Sample"])
            for i in range(len(iterations)):
                writer.writerow([int(iterations[i]), float(mean_best[i]), float(std_best[i]), float(mean_sample[i]), float(std_sample[i])])

        plt.figure(figsize=(10, 6))
        plt.plot(iterations, mean_best, label="Avg Best Value")
        # plt.fill_between(iterations, mean_best - std_best, mean_best + std_best, alpha=0.2)
        plt.plot(iterations, mean_sample, label="Avg Sampled Value", linestyle="--", alpha=0.5)
        plt.axhline(y=global_opt_y, linestyle="--", label="True Minimum")
        plt.xlabel("Iteration")
        plt.ylabel("Objective Value")
        plt.title(f'Convergence: {self.config["experiment_name"]}')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.result_dir, "convergence_avg.png"))
        plt.close()

    def _save_query_cdf(self, queries_to_success, total_trials):
        sorted_queries = np.sort(np.array(queries_to_success, dtype=float))
        n_success = len(sorted_queries)
        if n_success == 0:
            return

        y_vals = np.arange(1, n_success + 1) / total_trials

        csv_path = os.path.join(self.result_dir, "query_cdf.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Queries", "Success_Probability"])
            for q, p in zip(sorted_queries, y_vals):
                writer.writerow([float(q), float(p)])

        plt.figure(figsize=(10, 6))
        plt.step(sorted_queries, y_vals, where="post")
        plt.xlabel("Cumulative Queries")
        plt.ylabel("Probability")
        plt.title(f'Query CDF: {self.config["experiment_name"]}')
        plt.grid(True)
        plt.savefig(os.path.join(self.result_dir, "query_cdf.png"))
        plt.close()

    def _copy_latest_artifacts_to_exp_dir(self):
        latest_map = {
            "circuit_StatePrep.png": "circuit_state_prep.png",
            "circuit_Oracle.png": "circuit_oracle.png",
            "circuit_Diffuser.png": "circuit_diffuser.png",
            "circuit_GroverStep.png": "circuit_grover_step.png",
            "circuit_full_circuit_1step.png": "circuit_full_circuit_1step.png",
            "circuit_metrics.txt": "circuit_metrics.txt",
            "convergence_avg.png": "convergence_avg.png",
            "query_cdf.png": "query_cdf.png",
            "query_cdf.csv": "query_cdf.csv",
            "convergence_stats.csv": "convergence_stats.csv",
        }

        for src_name, dst_name in latest_map.items():
            src_path = os.path.join(self.result_dir, src_name)
            dst_path = os.path.join(self.exp_dir, dst_name)
            if os.path.exists(src_path):
                shutil.copyfile(src_path, dst_path)
