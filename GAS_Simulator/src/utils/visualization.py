import os
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit

def save_circuit_diagrams(model, save_dir: str):
    """
    モデルから回路コンポーネントを取得し、画像として保存する
    """
    components = model.get_components_for_visualization()
    
    for name, qc in components.items():
        try:
            # 分解して基本ゲートのみにすると見づらくなる場合があるので、
            # まずはハイレベルな構造を描画する
            # style={'fold': 20} などで折り返し幅を調整可能
            fig = qc.draw(output='mpl', filename=None)
            
            save_path = os.path.join(save_dir, f"circuit_{name}.png")
            fig.savefig(save_path, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved circuit diagram: {save_path}")
            
        except Exception as e:
            print(f"[Warning] Failed to draw circuit '{name}': {e}")
            print("Make sure 'matplotlib' and 'pylatexenc' (optional) are installed.")

def save_circuit_metrics(model, save_dir: str):
    """
    回路の深さやゲート構成比率をテキストファイルに保存する
    circuit_evaluation設定があればトランスパイル後の、なければ論理回路のメトリクスを出力
    """
    components = model.get_components_for_visualization()
    raw_qc = components["full_circuit_1step"]
    
    # --- 修正: 設定の有無で分岐 ---
    if 'circuit_evaluation' in model.config:
        # トランスパイルモード
        qc = model.transpile_circuit(raw_qc)
        mode_str = "Transpiled (Physical/Optimized)"
        
        eval_config = model.config.get('circuit_evaluation') or {}
        opt_level = eval_config.get('optimization_level', 1)
        basis_gates = eval_config.get('basis_gates', "Default")
    else:
        # 生回路モード
        qc = raw_qc
        mode_str = "Raw (Logical Structure)"
        opt_level = "N/A"
        basis_gates = "N/A"
    # ---------------------------
    
    depth = qc.depth()
    count_ops = qc.count_ops()
    n_qubits = qc.num_qubits
    
    report_path = os.path.join(save_dir, "circuit_metrics.txt")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== GAS Circuit Metrics (1 Grover Step) ===\n")
        f.write(f"Evaluation Mode: {mode_str}\n")
        
        if mode_str.startswith("Transpiled"):
            f.write(f"Configuration:\n")
            f.write(f"  Optimization Level: {opt_level}\n")
            f.write(f"  Basis Gates: {basis_gates}\n")
        
        f.write(f"\nResults:\n")
        f.write(f"  Qubits: {n_qubits}\n")
        f.write(f"  Depth: {depth}\n")
        f.write("\nGate Counts:\n")
        # 見やすいようにゲート名でソートして出力
        for gate in sorted(count_ops.keys()):
            f.write(f"  {gate}: {count_ops[gate]}\n")
            
    print(f"Saved metrics report: {report_path}")