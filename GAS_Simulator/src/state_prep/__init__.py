from .uniform import UniformStateBuilder
from .w_state import WStateBuilder
from .dicke_state import DickeStateBuilder

# 文字列キーでクラスを取得できるファクトリ
STATE_PREP_FACTORY = {
    "uniform": UniformStateBuilder,
    "hadamard": UniformStateBuilder, # alias
    "w_state": WStateBuilder,
    "dicke": DickeStateBuilder
}