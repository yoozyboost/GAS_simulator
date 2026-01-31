import sys
import os
import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from runner import GASRunner
from core import gas_model as gas_model_module


def _write_yaml(path, d):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(d, f, allow_unicode=True)


def test_default_convergence_is_nocircuit(monkeypatch, tmp_path):
    """
    backendがstatevectorでも、convergence_backend未指定なら
    収束評価は回路なしが選ばれることを期待する。
    """

    # run_step_circuit が呼ばれたら即失敗にする
    def _boom(*args, **kwargs):
        raise RuntimeError("run_step_circuit was called but should not be called in default nocircuit mode")

    monkeypatch.setattr(gas_model_module.GASModel, "run_step_circuit", _boom, raising=True)

    cfg = {
        "experiment_name": "pytest_default_nocircuit",
        "problem": {
            "n_key": 2,
            "n_val": 4,
            "objective_function": "x0 + x1",
            "variable_type": "binary",
        },
        "algorithm": {
            "method": "qd",
            "initial_state": "uniform",
        },
        "simulation": {
            "backend": "statevector",
            "max_iterations": 3,
            # convergence_backend をあえて書かない
        },
        "execution": {
            "num_trials": 1,
        },
        # 回路資産はデフォルトで作る想定
        "circuit_assets": {
            "build": True,
        },
    }

    cfg_path = tmp_path / "config.yaml"
    _write_yaml(cfg_path, cfg)

    runner = GASRunner(str(cfg_path), parent_dir=str(tmp_path))
    runner.run()

    # 回路資産が出ていることの最低限チェック
    out_dir = tmp_path / "pytest_default_nocircuit"
    assert (out_dir / "settings.yaml").exists()
    assert (out_dir / "circuit_state_prep.png").exists()
    assert (out_dir / "circuit_oracle.png").exists()
    assert (out_dir / "circuit_full_circuit_1step.png").exists()
    assert (out_dir / "circuit_metrics.txt").exists()
    assert (out_dir / "convergence_avg.png").exists()
