import sympy
import numpy as np
from functools import lru_cache
import re
from collections import defaultdict

def str_to_sympy(obj_fun_str: str):
    """文字列をSymPy式に変換"""
    return sympy.expand(sympy.sympify(obj_fun_str))

def get_num_vars(obj_fun_str: str) -> int:
    """式に含まれる変数の最大インデックスを取得 (例: x0 + x3 -> 4)"""
    expr = str_to_sympy(obj_fun_str)
    variables = expr.free_symbols
    max_idx = -1
    for var in variables:
        # x{d} の形式から d を抽出
        match = re.search(r'x(\d+)', str(var))
        if match:
            idx = int(match.group(1))
            if idx > max_idx:
                max_idx = idx
    return max_idx + 1

@lru_cache(maxsize=10000)
def evaluate_obj(obj_fun_str: str, bitstring: str, var_type: str = 'binary') -> float:
    """
    ビット列に対する目的関数の値を計算
    
    Args:
        obj_fun_str: 目的関数 (例: "x0 + x1")
        bitstring: ビット列 (例: "101")。左がx0に対応
        var_type: "binary" (0/1) or "spin" (+1/-1)
            - binary: そのまま代入
            - spin: bit '0' -> +1, bit '1' -> -1 として代入 (一般的なIsing定義)
    """
    n_vars = len(bitstring)
    expr = str_to_sympy(obj_fun_str)
    
    subs_dict = {}
    for i in range(n_vars):
        var_sym = sympy.Symbol(f'x{i}')
        if var_type == 'binary':
            val = int(bitstring[i])
        elif var_type == 'spin':
            # 通常、量子計算の測定値 '0'は|0>状態(+1)、'1'は|1>状態(-1)に対応
            val = 1 if bitstring[i] == '0' else -1
        else:
            raise ValueError(f"Unknown var_type: {var_type}")
        
        subs_dict[var_sym] = val

    return float(expr.subs(subs_dict))

def convert_binary_to_spin_eq(obj_fun_str: str) -> str:
    """
    Binary変数(0, 1)で記述された目的関数を、Spin変数(1, -1)の式に変換する。
    変換式: x_i = (1 - z_i) / 2
    
    Args:
        obj_fun_str: "x0 + x1" (xは0,1)
    Returns:
        変換後の式文字列 (変数名はxのまま、値の解釈が変わる)
    """
    expr = str_to_sympy(obj_fun_str)
    n_vars = get_num_vars(obj_fun_str)
    
    subs_dict = {}
    for i in range(n_vars):
        x_sym = sympy.Symbol(f'x{i}')
        # ここでは数式上の変換を行う。
        # 新しい変数 z_i を便宜上 x_i として扱う（ライブラリの互換性維持のため）
        # x = (1 - z) / 2
        subs_dict[x_sym] = (1 - x_sym) / 2
        
    new_expr = expr.subs(subs_dict)
    return str(sympy.expand(new_expr))

def normalize_for_qsp(obj_fun_str: str):
    """
    QSPのために目的関数を正規化する。
    係数の絶対値和で割り、定義域を[-1, 1]等に収める処理。
    """
    expr = str_to_sympy(obj_fun_str)
    polydict = expr.as_poly().as_dict()
    
    # L1ノルム（係数の絶対値和）を計算
    l1_norm = sum(abs(k) for k in polydict.values())
    
    if l1_norm == 0:
        return obj_fun_str, 0.0

    # スケーリング
    # QSPでは通常 |f(x)| <= 1 である必要があるため正規化
    # 係数を pi などを考慮して調整する場合もあるが、ここでは単純正規化
    normalized_expr = expr / l1_norm
    
    return str(sympy.expand(normalized_expr)), float(l1_norm)

def reversed_bits_table(n_qubits: int):
    """QiskitのLittle Endianに対応するためのビット反転テーブル"""
    table = np.zeros(2**n_qubits, dtype=object)
    format_str = f'0{n_qubits}b'
    for i in range(2**n_qubits):
        # bit列にして反転
        b_str = format(i, format_str)
        table[i] = b_str[::-1]
    return table

def calc_obj_fun_distribution(n_key: int, obj_fun_str: str, var_type: str = 'binary'):
    """
    全探索を行い、目的関数の値の分布と、各値に対応する解のリストを作成する。
    数理シミュレーション(woCircuit)で使用。
    """
    dist = defaultdict(int)
    memo = defaultdict(list)
    
    # 全状態探索 (2^n_key)
    for i in range(2 ** n_key):
        # 整数iをビット列に変換 (例: 5 -> "101")
        bitstring = format(i, f'0{n_key}b')
        val = evaluate_obj(obj_fun_str, bitstring, var_type)
        
        dist[val] += 1
        memo[val].append(i) # 整数値で保持

    sorted_items = sorted(dist.items())
    
    values = []
    cumulative_counts = []
    current_count = 0
    
    for val, count in sorted_items:
        values.append(val)
        current_count += count
        cumulative_counts.append(current_count)
        
    return values, cumulative_counts, dict(memo)