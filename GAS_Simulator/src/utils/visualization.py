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
    """
    components = model.get_components_for_visualization()
    qc = components["full_circuit_1step"]
    
    # トランスパイル（基本ゲートへの分解）後のメトリクスを見るのが一般的
    # ここでは簡易的にdecomposeを使用
    decomposed_qc = qc.decompose()
    
    depth = decomposed_qc.depth()
    count_ops = decomposed_qc.count_ops()
    n_qubits = qc.num_qubits
    
    report_path = os.path.join(save_dir, "circuit_metrics.txt")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== GAS Circuit Metrics (1 Grover Step) ===\n")
        f.write(f"Qubits: {n_qubits}\n")
        f.write(f"Depth (decomposed): {depth}\n")
        f.write("\nGate Counts:\n")
        for gate, count in count_ops.items():
            f.write(f"  {gate}: {count}\n")
            
    print(f"Saved metrics report: {report_path}")