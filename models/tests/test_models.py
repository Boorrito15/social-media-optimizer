"""Smoke tests for ``models.predict_lin`` / ``models.predict_clas``.

These are integration tests: they require ``models/linreg.joblib``,
``models/clas.joblib`` and ``models/pipeline_state.joblib`` to already exist
on disk. Run ``python -m models.train`` once first.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from models import predict_clas, predict_lin
from models.config import CLAS_PATH, DEFAULT_PROCESSED_CSV, LINREG_PATH, MODELS_DIR, STATE_PATH

pytestmark = pytest.mark.skipif(
    not (STATE_PATH.exists() and LINREG_PATH.exists() and CLAS_PATH.exists()),
    reason=(
        "models artefacts missing; run `python -m models.train` first. "
        "Looked for: pipeline_state.joblib, linreg.joblib, clas.joblib "
        f"under {MODELS_DIR}"
    ),
)

df = pd.read_csv(DEFAULT_PROCESSED_CSV).iloc[:3].reset_index(drop=True)


def test_predict_lin_shape_and_types() -> None:
    out = predict_lin(df)
    assert isinstance(out, list)
    assert len(out) == len(df)
    for row in out:
        assert set(row) == {"engagement", "views"}
        assert isinstance(row["engagement"], float)
        assert isinstance(row["views"], float)


def test_predict_clas_label_format() -> None:
    out = predict_clas(df)
    assert isinstance(out, list)
    assert len(out) == len(df)
    for row in out:
        assert isinstance(row["views"], str) and row["views"].startswith("views_")
        assert isinstance(row["engagement"], str) and row["engagement"].startswith("engagement_")
