import numpy as np
from qiskit import QuantumCircuit
from functools import lru_cache
from .base import OracleBuilder
from utils.math_tools import str_to_sympy

# キャッシュ関数 (変更なし)
@lru_cache(maxsize=32)
def get_sign_angles_cached(degree: int):
    # try:
    #     import pyqsp
    #     from pyqsp.angle_sequence import QuantumSignalProcessingPhases
    #     import numpy as np 
        
    #     pg = pyqsp.poly.PolySign()
    #     p_out = pg.generate(degree=degree, delta=20) 
        
    #     if hasattr(p_out, 'coef'):
    #         p_coefs = p_out.coef
    #     else:
    #         p_coefs = p_out

    #     poly = np.polynomial.chebyshev.Chebyshev(p_coefs)
    #     phases = QuantumSignalProcessingPhases(poly, signal_operator="Wx")
        
    #     return list(phases)
    # except ImportError:
    #     print("[Warning] pyqsp not found. Using hardcoded fallback angles (d=39).")
    # return [-1.5728442741198856, 0.002716507314116434, -0.0034944203899851534, 0.004697402122637806, -0.0064598077284445665, 0.008941402808402499, -0.012331122328782751, 0.016852481636580485, -0.0227724071879436, 0.030416793061699643, -0.040198846282709866, 0.05267158710919273, -0.06862703554209859, 0.08929047702801984, -0.1167244477254279, 0.15474770866795162, -0.21130946252330474, 0.30582300746404245, -0.49819065573041055, 1.028344015660705, 1.0283440156607058, -0.4981906557304101, 0.3058230074640449, -0.21130946252330665, 0.15474770866795184, -0.11672444772542859, 0.08929047702802013, -0.06862703554209802, 0.052671587109192716, -0.04019884628270955, 0.030416793061698838, -0.02277240718794235, 0.01685248163657954, -0.012331122328781446, 0.00894140280840186, -0.0064598077284444, 0.004697402122637917, -0.0034944203899849313, 0.0027165073141163787, -0.0020479473249887387]
    return [-1.5716717067761834, 0.0011503675299671112, -0.0012692522360260305, 0.0014241998918902432, -0.0016215775143417765, 0.0018683974610131937, -0.0021723565891240693, 0.0025418772055978422, -0.002986150663268239, 0.0035151850524568484, -0.004139859264196577, 0.004871986814923401, -0.005724394292989432, 0.006711021206927903, -0.007847050514211862, 0.009149082376604944, -0.010635368007211543, 0.012326126279190719, -0.014243973726967507, 0.016414509740118488, -0.018867114792015194, 0.021636043136030292, -0.02476192690819823, 0.028293863271141362, -0.032292342517240114, 0.036833414627504, -0.04201472383380778, 0.04796443843633669, -0.05485480857573721, 0.06292338614692139, -0.07250745369663902, 0.08410232412240748, -0.0984652585380521, 0.11681265011463404, -0.14122445672643558, 0.17556042597168542, -0.22782696262360758, 0.3174958692386929, -0.5045208121324722, 1.0294926736346954, 1.0294926736346928, -0.5045208121324694, 0.31749586923869333, -0.2278269626236091, 0.17556042597169, -0.1412244567264403, 0.11681265011463576, -0.09846525853805235, 0.08410232412240845, -0.07250745369663975, 0.06292338614692264, -0.0548548085757366, 0.047964438436336354, -0.04201472383380822, 0.036833414627504085, -0.032292342517240225, 0.028293863271141806, -0.02476192690819823, 0.021636043136030764, -0.018867114792015305, 0.01641450974011846, -0.014243973726967729, 0.012326126279190941, -0.010635368007211293, 0.009149082376604833, -0.007847050514211418, 0.006711021206927625, -0.005724394292989127, 0.0048719868149231516, -0.004139859264195939, 0.0035151850524559602, -0.0029861506632673784, 0.0025418772055974814, -0.0021723565891233754, 0.0018683974610124443, -0.0016215775143414435, 0.0014241998918894383, -0.00126925223602542, 0.0011503675299666394, -0.0008753799812867835]

class QSPOracleBuilder(OracleBuilder):
    
    def build_oracle(self, n_key: int, **kwargs) -> QuantumCircuit:
        obj_fun_str = kwargs.get('obj_fun_str')
        threshold = kwargs.get('threshold', 0.0)
        degree = kwargs.get('qsp_degree', 21)

        if degree % 2 == 0:
            degree += 1

        # --- 修正箇所: 正規化ロジックを元のQSP_ilquantumに合わせる ---
        
        # 1. 目的関数のみをパースして係数を取得
        expr = str_to_sympy(obj_fun_str)
        polydict = expr.as_poly().as_dict()
        
        # 2. L1ノルム (係数の絶対値和) を計算。閾値は含めない。
        l1_norm = sum(abs(k) for k in polydict.values())
        if l1_norm == 0: l1_norm = 1.0 # ゼロ除算回避

        # 3. スケーリング係数 (pi/2 を掛けるのが重要)
        # QSP_ilquantum: objfun * pi / (2 * v)
        scale_factor = np.pi / (2 * l1_norm)

        # 4. 各項の係数をスケーリング
        scaled_polydict = {ps: float(k) * scale_factor for ps, k in polydict.items()}
        
        # 5. 閾値 (Delta) の計算
        # 元コード: Delta = float((-delta / (objfun_max)) * (pi / 2))
        # ここでの threshold は GASの文脈での閾値 y_th。
        # f(x) < y_th  <=>  f(x) - y_th < 0
        # 定数項として -y_th を加えることに相当するため、符号はマイナス
        scaled_threshold = (threshold * scale_factor)
        
        # 6. 角度計算
        angles = get_sign_angles_cached(degree)

        # 回路構築へ (スケーリング済みの辞書と閾値を渡す)
        return self._construct_qsp_circuit(n_key, scaled_polydict, scaled_threshold, angles)

    def build_state_prep(self, n_key: int, obj_fun_str: str, state_prep_method, **kwargs) -> QuantumCircuit:
        qc = QuantumCircuit(n_key + 1, name="QSP_StatePrep")
        qc_key = state_prep_method.build(n_key)
        qc.compose(qc_key, range(n_key), inplace=True)
        return qc

    def _construct_qsp_circuit(self, n_key, polydict, scaled_threshold, angles):
        # polydictは既にスケーリング済み
        
        # 定数項の集約 (閾値分を引く)
        Delta = -scaled_threshold
        
        O = QuantumCircuit(n_key + 1, name="QSP_Oracle")
        Wz_plus = QuantumCircuit(n_key)
        Wz_minus = QuantumCircuit(n_key)

        for (ps, k) in polydict.items():
            k = float(k)
            i_nonzero = np.nonzero(ps)[0]
            n_nonzero = len(i_nonzero)
            
            if n_nonzero == 0:
                # 目的関数自体の定数項を加算
                Delta += k
            else:
                for i in range(n_nonzero - 1):
                    Wz_plus.cx(i_nonzero[i], i_nonzero[-1])
                    Wz_minus.cx(i_nonzero[i], i_nonzero[-1])
                
                # 回転角: QSP_ilquantumでは 2*k (kは既にpi/2スケール済み)
                Wz_plus.rz(2 * k, i_nonzero[-1])
                Wz_minus.rz(-2 * k, i_nonzero[-1])
                
                for i in range(n_nonzero - 1):
                    Wz_plus.cx(i_nonzero[n_nonzero - 2 - i], i_nonzero[-1])
                    Wz_minus.cx(i_nonzero[n_nonzero - 2 - i], i_nonzero[-1])

        controlled_Wz_plus = Wz_plus.to_gate().control(1)
        controlled_Wz_minus = Wz_minus.to_gate().control(1)
        ancilla = n_key
        
        O.rx(-2 * angles[0], ancilla)
        for i in range(1, len(angles)):
            O.x(ancilla)
            O.append(controlled_Wz_plus, [ancilla] + list(range(n_key)))
            O.x(ancilla)
            O.append(controlled_Wz_minus, [ancilla] + list(range(n_key)))
            
            if abs(Delta) > 1e-9:
                O.rz(2 * Delta, ancilla)
            
            O.rx(-2 * angles[i], ancilla)
            
        return O