from .qd_oracle import QDOracleBuilder
from .qsp_oracle import QSPOracleBuilder

ORACLE_FACTORY = {
    "qd": QDOracleBuilder,
    "qsp": QSPOracleBuilder
}