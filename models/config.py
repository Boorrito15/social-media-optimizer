"""Configuration, default paths and a frozen dataclass for the model pipeline.

The :class:`PipelineConfig` dataclass is the single source of truth for every
hyperparameter that historically lived as a magic number in the notebook.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(__file__).resolve().parent

DEFAULT_PROCESSED_CSV = (
    REPO_ROOT / "processed.csv"
    if (REPO_ROOT / "processed.csv").exists()
    else REPO_ROOT / "data" / "processed" / "processed.csv"
)

STATE_PATH = MODELS_DIR / "pipeline_state.joblib"
LINREG_PATH = MODELS_DIR / "linreg.joblib"
CLAS_PATH = MODELS_DIR / "clas.joblib"

PIPELINE_STATE_VERSION = 1
MODEL_BUNDLE_VERSION = 1


@dataclass(frozen=True)
class PipelineConfig:
    """Single source of truth for every pipeline hyperparameter.

    All fields have safe notebook-compatible defaults so a bare
    ``PipelineConfig()`` reproduces the pre-refactor behaviour 1:1.
    """

    train_test_random_seed: int = 42
    test_size: float = 0.10
    n_iqr_bins: int = 2
    bottom_iqr_percentile: float = 0.1
    top_iqr_percentile: float = 0.9
    min_vocab_frequency: int = 1


def truthy(name: str) -> bool:
    """Return True if env var ``name`` is set to a truthy string."""
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def processed_csv_path() -> Path:
    """Path to ``processed.csv``. Honors ``PROCESSED_CSV`` env override."""
    override = os.environ.get("PROCESSED_CSV")
    return Path(override) if override else DEFAULT_PROCESSED_CSV

