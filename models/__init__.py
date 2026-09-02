"""Top-level public surface for the models package.

Streamlit (or any other caller) only needs::

    from models import predict_lin, predict_clas
"""

from .config import PipelineConfig
from .predict import predict_lin, predict_clas

__all__ = ["PipelineConfig", "predict_lin", "predict_clas"]
