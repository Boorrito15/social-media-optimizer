"""Public ``predict_lin`` / ``predict_clas`` entry points.

These accept a DataFrame shaped like ``processed.csv`` and return a list of
prediction dicts.
"""

from __future__ import annotations

import joblib
import pandas as pd

from .config import (
    CLAS_PATH,
    LINREG_PATH,
    STATE_PATH,
    PipelineConfig,
)
from .shared.build_features import preprocess


def _build_X(df_proc: pd.DataFrame, bundle: dict):
    X = df_proc.drop(columns=["views", "engagement"])
    NUMERIC = bundle["NUMERIC_COLS"]
    binaries = [c for c in bundle["binary_cols"] if c in X.columns]
    Xn = bundle["scaler_X"].transform(X[NUMERIC])
    return pd.concat(
        [pd.DataFrame(Xn, columns=NUMERIC, index=X.index), X[binaries]], axis=1
    )


def predict_lin(
    df: pd.DataFrame, config: PipelineConfig | None = None
) -> list[dict]:
    """Predict ``engagement`` and ``views`` for each row of ``df``.

    Returns a list of ``{"engagement": <float>, "views": <float>}`` dicts in
    *raw* (unscaled) units, ordered to match ``df`` row order.
    """
    config = config or PipelineConfig()
    bundle = joblib.load(LINREG_PATH)
    state = joblib.load(STATE_PATH)
    df_proc = preprocess(df, state)
    X_final = _build_X(df_proc, bundle)

    preds_scaled = bundle["model"].predict(X_final)
    preds_raw = bundle["scaler_y"].inverse_transform(preds_scaled)
    return [
        {"engagement": float(p[0]), "views": float(p[1])} for p in preds_raw
    ]


def _split_label(label: str) -> tuple[str, str]:
    """Split ``"views_<i>_engagement_<j>"`` into ``("views_<i>", "engagement_<j>")``."""
    left, right = label.split("_engagement_")
    return left, f"engagement_{right}"


def predict_clas(
    df: pd.DataFrame, config: PipelineConfig | None = None
) -> list[dict]:
    """Predict per-row ``engagement`` and ``views`` bin labels.

    Returns a list of ``{"engagement": "engagement_<i>", "views": "views_<j>"}``
    dicts, ordered to match ``df`` row order.
    """
    config = config or PipelineConfig()
    bundle = joblib.load(CLAS_PATH)
    state = joblib.load(STATE_PATH)
    df_proc = preprocess(df, state)
    X_final = _build_X(df_proc, bundle)

    codes = bundle["model"].predict(X_final).astype(int)
    labels = bundle["classify_labels"]
    out: list[dict] = []
    for c in codes:
        views_label, eng_label = _split_label(labels[int(c)])
        out.append({"views": views_label, "engagement": eng_label})
    return out
