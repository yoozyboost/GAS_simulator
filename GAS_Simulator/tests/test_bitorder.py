import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from utils.math_tools import calc_obj_fun_distribution


def test_calc_obj_fun_distribution_uses_lsb_as_x0():
    """
    x0が先頭かつLSBとして扱われることを前提にする。
    目的関数 f(x)=x0 なら、x0=1の状態は整数 {1,3,5,...} に対応する。
    n_key=2 の場合は {1,3}。
    """
    values, cum_counts, memo = calc_obj_fun_distribution(n_key=2, obj_fun_str="x0", var_type="binary")

    # 値1.0の候補集合が {1,3} であること
    # memoのキーはfloatになることがあるので 1.0 を使う
    assert 1.0 in memo
    assert set(memo[1.0]) == {1, 3}
