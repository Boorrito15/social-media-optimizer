"""Linear (regression) SVM model. Train, serialise, predict.

Output scheme
-------------
* ``bundle`` written to ``models/linreg.joblib`` ::
      {
          "version": 1,
          "config": PipelineConfig(...),
          "model": MultiOutputRegressor(SVR(C=1, degree=2, gamma='scale', coef0=0)),
          "scaler_X": StandardScaler,
          "scaler_y": RobustScaler,
          "NUMERIC_COLS": [...],
          "binary_cols": [...],
      }
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVR

from .config import (
    LINREG_PATH,
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


def _build_xy(df_processed: pd.DataFrame, idx_tr: np.ndarray, idx_te: np.ndarray):
    X_all = df_processed.drop(columns=["views", "engagement"])
    y_all = df_processed[["engagement", "views"]]

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

    scaler_y = RobustScaler().fit(y_all.iloc[idx_tr])
    y_tr = scaler_y.transform(y_all.iloc[idx_tr])
    y_te = scaler_y.transform(y_all.iloc[idx_te])

    return X_tr, X_te, y_tr, y_te, NUMERIC_COLS, binary_cols, scaler_X, scaler_y


def train(config: PipelineConfig, state: dict | None = None) -> dict:
    """Train the regression SVM from scratch and serialise the bundle."""
    state = state or build_state(config, force=True)
    df = pd.read_csv(processed_csv_path())
    df = preprocess(df, state)
    df = validity_mask(df)
    df = iqr_slice(df, config.bottom_iqr_percentile, config.top_iqr_percentile)
    df = df.reset_index(drop=True)

    idx_tr, idx_te = split_indices(
        len(df), config.test_size, config.train_test_random_seed
    )

    X_tr, X_te, y_tr, y_te, NUMERIC, binaries, scaler_X, scaler_y = _build_xy(
        df, idx_tr, idx_te
    )

    svr = SVR(kernel="poly")
    svr.set_params(**SVM_PARAMS)
    model = MultiOutputRegressor(svr)
    model.fit(X_tr, y_tr)

    mae = float(np.mean(np.abs(y_te - model.predict(X_te))))
    print(f"[linreg] test MAE (scaled units): {mae:.4f}")

    bundle = {
        "version": MODEL_BUNDLE_VERSION,
        "config": config,
        "model": model,
        "scaler_X": scaler_X,
        "scaler_y": scaler_y,
        "NUMERIC_COLS": NUMERIC,
        "binary_cols": binaries,
    }
    joblib.dump(bundle, LINREG_PATH)
    return bundle


def train_or_load(
    config: PipelineConfig | None = None,
    retrain: bool = False,
) -> dict:
    """Return the bundle. Loads from disk if available, else trains."""
    config = config or PipelineConfig()
    if not retrain and LINREG_PATH.exists():
        return joblib.load(LINREG_PATH)
    return train(config)
