import sys
import os
import pytest

# srcパスを通す
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from utils.math_tools import (
    get_num_vars,
    evaluate_obj,
    convert_binary_to_spin_eq,
    normalize_for_qsp
)

def test_get_num_vars():
    assert get_num_vars("x0 + x1") == 2
    assert get_num_vars("x0 + x3") == 4 # x0, x1, x2, x3 -> 4変数
    assert get_num_vars("5") == 0

def test_evaluate_binary():
    # x0=1, x1=0 -> 1 + 0 = 1
    assert evaluate_obj("x0 + x1", "10", "binary") == 1
    # x0=1, x1=1 -> 1 + 1 = 2
    assert evaluate_obj("x0 + x1", "11", "binary") == 2

def test_evaluate_spin():
    # spin: '0' -> +1, '1' -> -1
    # obj = x0 + x1
    # "00" -> (+1) + (+1) = 2
    assert evaluate_obj("x0 + x1", "00", "spin") == 2.0
    # "10" -> (-1) + (+1) = 0
    assert evaluate_obj("x0 + x1", "10", "spin") == 0.0

def test_conversion():
    # Binary: x0 (0 or 1)
    # Spin変換: x0 -> (1 - x0)/2  (※変数名はx0のまま)
    # 検算: x0(spin)=+1(bit '0') のとき -> (1 - 1)/2 = 0
    #       x0(spin)=-1(bit '1') のとき -> (1 - (-1))/2 = 1

    expr = "x0"
    conv_expr = convert_binary_to_spin_eq(expr)
    # sympyの形式によって "1/2 - x0/2" や "-x0/2 + 1/2" になる
    # 値で確認
    assert evaluate_obj(conv_expr, "0", "spin") == 0 # 元のBinaryの0に相当
    assert evaluate_obj(conv_expr, "1", "spin") == 1 # 元のBinaryの1に相当

def test_normalize():
    # 係数和: |2| + |-3| = 5
    # 正規化後: (2/5)x0 - (3/5)x1
    expr = "2*x0 - 3*x1"
    norm_expr, scale = normalize_for_qsp(expr)

    assert scale == 5.0
    # 簡易チェック
    assert evaluate_obj(norm_expr, "10", "binary") == (2*1 - 3*0)/5

if __name__ == "__main__":
    # 手動実行用
    try:
        test_get_num_vars()
        test_evaluate_binary()
        test_evaluate_spin()
        test_conversion()
        test_normalize()
        print("All Math Tools tests passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")
