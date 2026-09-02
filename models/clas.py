"""Classification SVM model. Train, serialise, predict.

Output scheme
-------------
* Labels per row look like ``"views_<i>_engagement_<j>"`` (zero-indexed bins).
* y is a 1-D vector of integer class codes; ``SVC`` consumes integer labels
  directly. We do NOT use ``to_categorical`` because ``SVC`` requires integer
  class codes, not one-hot.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .config import (
    CLAS_PATH,
    MODEL_BUNDLE_VERSION,
    PipelineConfig,
    processed_csv_path,
)
from .shared.build_features import preprocess
from .shared.build_state import build_state
from .shared.filter import iqr_slice, validity_mask
from .shared.split import split_indices

SVM_PARAMS = dict(C=1.0, degree=2, gamma="scale", coef0=0.0)

NUMERIC_COLS_TEMPLATE = [
    "duration_seconds",
    "description_cluster",
    "title_cluster",
    "n_mentions",
    "n_hashtags",
    "n_emojis",
    "hashtag_count",
    "mention_count",
    "emoji_count",
]


def _iqr_binned(series: pd.Series, n_bins: int, bottom: float, top: float) -> np.ndarray:
    """Bin a 1-D series into ``n_bins`` integer bins using an IQR-style edge schedule.

    Reproduces the original notebook's ``_iqr_binned`` behaviour:
        lo = quantile(bottom); hi = quantile(top)
        edges = linspace(lo - 1.5*(hi-lo), hi + 1.5*(hi-lo), n_bins + 1)
        bin = clip(digitize(values, edges) - 1, 0, n_bins - 1)
    """
    lo = series.quantile(bottom)
    hi = series.quantile(top)
    iqr = hi - lo
    edges = np.linspace(lo - 1.5 * iqr, hi + 1.5 * iqr, n_bins + 1)
    return np.clip(np.digitize(series.to_numpy(), edges) - 1, 0, n_bins - 1).astype(int)


def _build_xy(df_processed: pd.DataFrame, config: PipelineConfig, idx_tr, idx_te):
    views_bin_tr = _iqr_binned(
        df_processed.iloc[idx_tr]["views"],
        config.n_iqr_bins,
        config.bottom_iqr_percentile,
        config.top_iqr_percentile,
    )
    eng_bin_tr = _iqr_binned(
        df_processed.iloc[idx_tr]["engagement"],
        config.n_iqr_bins,
        config.bottom_iqr_percentile,
        config.top_iqr_percentile,
    )
    cats_tr = pd.Categorical(
        [f"views_{v}_engagement_{e}" for v, e in zip(views_bin_tr, eng_bin_tr)]
    )
    y_tr = cats_tr.codes
    classify_labels = list(cats_tr.categories)

    views_bin_te = _iqr_binned(
        df_processed.iloc[idx_te]["views"],
        config.n_iqr_bins,
        config.bottom_iqr_percentile,
        config.top_iqr_percentile,
    )
    eng_bin_te = _iqr_binned(
        df_processed.iloc[idx_te]["engagement"],
        config.n_iqr_bins,
        config.bottom_iqr_percentile,
        config.top_iqr_percentile,
    )
    cats_te = pd.Categorical(
        [f"views_{v}_engagement_{e}" for v, e in zip(views_bin_te, eng_bin_te)],
        categories=cats_tr.categories,
    )
    y_te = cats_te.codes

    X_all = df_processed.drop(columns=["views", "engagement"])
    NUMERIC_COLS = [c for c in NUMERIC_COLS_TEMPLATE if c in X_all.columns]
    binary_cols = [
        c
        for c in X_all.columns
        if c not in set(NUMERIC_COLS) and c not in {"views", "engagement"}
    ]

    scaler_X = StandardScaler().fit(X_all.iloc[idx_tr][NUMERIC_COLS])

    def build(idx):
        Xn = scaler_X.transform(X_all.iloc[idx][NUMERIC_COLS])
        Xb = X_all.iloc[idx][binary_cols]
        return pd.concat(
            [
                pd.DataFrame(Xn, columns=NUMERIC_COLS, index=X_all.iloc[idx].index),
                Xb,
            ],
            axis=1,
        )

    X_tr = build(idx_tr)
    X_te = build(idx_te)

    return X_tr, X_te, y_tr, y_te, NUMERIC_COLS, binary_cols, scaler_X, classify_labels


def train(config: PipelineConfig, state: dict | None = None) -> dict:
    """Train the classification SVM from scratch and serialise the bundle."""
    state = state or build_state(config, force=True)
    df = pd.read_csv(processed_csv_path())
    df = preprocess(df, state)
    df = validity_mask(df)
    df = iqr_slice(df, config.bottom_iqr_percentile, config.top_iqr_percentile)
    df = df.reset_index(drop=True)

    idx_tr, idx_te = split_indices(
        len(df), config.test_size, config.train_test_random_seed
    )

    X_tr, X_te, y_tr, y_te, NUMERIC, binaries, scaler_X, classify_labels = _build_xy(
        df, config, idx_tr, idx_te
    )

    model = SVC(kernel="poly")
    model.set_params(**SVM_PARAMS)
    model.fit(X_tr, y_tr)

    preds = model.predict(X_te)
    acc = float(accuracy_score(y_te, preds))
    print(f"[clas] test accuracy: {acc:.4f}")

    bundle = {
        "version": MODEL_BUNDLE_VERSION,
        "config": config,
        "model": model,
        "scaler_X": scaler_X,
        "classify_labels": classify_labels,
        "NUMERIC_COLS": NUMERIC,
        "binary_cols": binaries,
    }
    joblib.dump(bundle, CLAS_PATH)
    return bundle


def train_or_load(
    config: PipelineConfig | None = None,
    retrain: bool = False,
) -> dict:
    """Return the bundle. Loads from disk if available, else trains."""
    config = config or PipelineConfig()
    if not retrain and CLAS_PATH.exists():
        return joblib.load(CLAS_PATH)
    return train(config)
