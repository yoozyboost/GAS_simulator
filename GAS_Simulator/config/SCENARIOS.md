
# Scenario YAML Samples

このディレクトリには、GAS_Simulator の実行用YAMLサンプルを置きます。

## 基本構造

- experiment_name: 実験名（結果フォルダ名などに使われる）
- problem:
  - n_key: キーレジスタのビット数
  - n_val: バリューレジスタのビット数（QDのFejérモードで必要な場合がある）
  - objective_function: sympyで解釈可能な式。変数は x0, x1, ... を使う
  - variable_type: binary または spin
- algorithm:
  - method: qd, qsp, classical_exhaustive など
  - initial_state: uniform, hadamard, dicke, w_state
  - state_prep_params: initial_state が dicke の場合に k を指定
  - qsp_params: method=qsp の場合に qsp_degree, threshold などを指定
- simulation:
  - backend: statevector, qasm, analytical
  - convergence_backend: circuit, nocircuit
  - qd_nocircuit_mode: fejer または infinite（QDの回路なし）
  - max_iterations: 反復回数
  - seed: 乱数種
- execution:
  - num_trials: 試行回数（平均性能を見る場合に増やす）
- circuit_assets:
  - build: true にすると回路図やコスト見積り用の生成を有効化
