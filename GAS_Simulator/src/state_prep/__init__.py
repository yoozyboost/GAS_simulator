from .uniform import UniformStateBuilder
from .w_state import WStateBuilder
from .dicke_state import DickeStateBuilder

# 文字列キーでクラスを取得できるファクトリ
STATE_PREP_FACTORY = {
    "uniform": UniformStateBuilder,
    "hadamard": UniformStateBuilder, # alias
    "w_state": WStateBuilder,
    "w": WStateBuilder,              # alias
    "dicke": DickeStateBuilder
}


def get_state_prep_method(name: str):
    """STATE_PREP_FACTORYからビルダを生成して返すヘルパ"""
    if name not in STATE_PREP_FACTORY:
        raise KeyError(f"Unknown initial_state: {name}")
    return STATE_PREP_FACTORY[name]()
