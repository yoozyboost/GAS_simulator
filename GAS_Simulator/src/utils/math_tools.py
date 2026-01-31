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
    bitstring: x0 x1 ... の順で与える（x0が先頭）
    var_type:
      - 'binary': 0/1
      - 'spin'  : 0 -> +1, 1 -> -1
    """
    expr = str_to_sympy(obj_fun_str)

    n_vars = get_num_vars(obj_fun_str)
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
    Binary変数(0,1)で書かれた式を、Spin変数(+1,-1)で評価できる形に変換する。
    変換は変数名を変えずに行う。
      binary: b_i ∈ {0,1}
      spin  : x_i ∈ {+1,-1} かつ b_i = (1 - x_i)/2
    よって式中の x_i を (1 - x_i)/2 に置換する。
    """
    expr = str_to_sympy(obj_fun_str)
    variables = sorted(list(expr.free_symbols), key=lambda v: int(str(v)[1:]))

    subs_map = {}
    for v in variables:
        subs_map[v] = (1 - v) / 2

    spin_expr = sympy.expand(expr.subs(subs_map))
    return str(spin_expr)

def get_all_bitstrings(n: int):
    """
    長さnのすべてのビット列を辞書で返す。
    key: int, value: bitstring（x0が先頭になるように反転済み）
    """
    table = {}
    format_str = f'0{n}b'
    for i in range(2**n):
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
        # 整数iをビット列に変換（x0が先頭になるよう反転）
        bitstring = format(i, f'0{n_key}b')[::-1]
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

def normalize_for_qsp(obj_fun_str: str, n_key: int = None):
    """
    QSP向けに目的関数を L1 ノルムで正規化し、(正規化後文字列, l1_norm) を返す。
    既存テスト互換のため、第二戻り値は scale ではなく l1_norm を返す。

    正規化後の式は expr / l1_norm。
    """
    if n_key is None:
        n_key = get_num_vars(obj_fun_str)

    xs = sympy.symbols(f'x0:{n_key}')
    expr = sympy.expand(sympy.sympify(obj_fun_str))

    poly = sympy.Poly(expr, *xs, domain='RR')
    coeffs = poly.as_dict()

    l1_norm = 0.0
    for _, c in coeffs.items():
        l1_norm += abs(float(c))

    if l1_norm == 0.0:
        return obj_fun_str, 0.0

    expr_scaled = sympy.expand(expr / float(l1_norm))
    return str(expr_scaled), float(l1_norm)
