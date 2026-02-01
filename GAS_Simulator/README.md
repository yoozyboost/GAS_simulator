
# GAS_Simulator

Grover Adaptive Search（GAS）のシミュレーション用Python実装です。QD-GAS と QSP-GAS の両方に対応し、回路あり（Qiskit）と回路なし（数理モデル）の比較実験ができます。

## Requirements

- Python 3.10 以上
- requirements.txt に依存関係を記載

## Quick Start

1. 依存関係の導入

   pip install -r requirements.txt

2. YAMLを指定して実行

   python main.py --config config/qd_binary_nocircuit_uniform.yaml

## YAML Samples

config/ に実行例を複数置いています。スキーマの要点は config/SCENARIOS.md を参照してください。
