from .qd_oracle import QDOracleBuilder
from .qsp_oracle import QSPOracleBuilder

ORACLE_FACTORY = {
    "qd": QDOracleBuilder,
    "qsp": QSPOracleBuilder
}

def get_oracle_builder(method: str):
    """ORACLE_FACTORYからビルダを生成して返すヘルパ"""
    if method not in ORACLE_FACTORY:
        raise KeyError(f"Unknown method: {method}")
    return ORACLE_FACTORY[method]()
