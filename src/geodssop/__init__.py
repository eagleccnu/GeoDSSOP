"""GeoDSSOP-PDB public Python package."""

from geodssop.config import METHOD_ID, METHOD_NAME, VERSION
from geodssop.inference import PredictionResult, predict
from geodssop.model import GeoDSSOPConfig, GeoDSSOPModel

__version__ = VERSION

__all__ = [
    "GeoDSSOPConfig",
    "GeoDSSOPModel",
    "METHOD_ID",
    "METHOD_NAME",
    "PredictionResult",
    "VERSION",
    "predict",
]
