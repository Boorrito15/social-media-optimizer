"""Train and export the models / artifacts served by the API and Streamlit app.

Usage:
    .venv/bin/python -m src.ml.train

The notebook relies on Keras/TensorFlow which is not part of this runtime.
This script reproduces the analysis intent with reproducible scikit-learn
models on a compact, faithful feature set built by FeaturePipeline.
It trains:
  * two "high / low" classifiers  (views, engagement)
  * two continuous regressors      (log1p views, log1p engagement)
  * a semantic "similar videos" index (sentence-transformers embeddings + peers)
and writes artifacts to data/models/bundle.joblib
"""

from __future__ import annotations

import math
import os

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split

from src.ml.features import FeaturePipeline, _parse_json_list

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "processed.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "data", "models")
BUNDLE_PATH = os.path.join(MODELS_DIR, "bundle.joblib")
SIM_PATH = os.path.join(MODELS_DIR, "similar.joblib")

BIN_Q = 0.5  # high = above median


def _extract_description(text) -> str:
    if not isinstance(text, str):
        return ""
    import json

    try:
        d = json.loads(text)
    except Exception:
        return ""
    if not isinstance(d, dict):
        return ""
    return str(d.get("play_by_play") or "").strip()


def _load() -> pd.DataFrame:
    df = pd.read_csv(DATA_CSV)
    df = df[df["views"].notna() & df["engagement"].notna()]
    df = df[df["views"] > 0]
    df = df[df["engagement"] > 0]
    df = df[df["content"].notna()]
    df["description"] = df["description_json"].map(_extract_description)
    df = df.reset_index(drop=True)
    return df


def _fit_binary_clf(X, y, evX, evy):
    clf = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=8,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        eval_metric="logloss",
        early_stopping_rounds=25,
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )
    clf.fit(X, y, eval_set=[(evX, evy)], verbose=False)
    return clf


def _fit_reg(X, y, evX, evy):
    reg = xgb.XGBRegressor(
        n_estimators=600,
        max_depth=8,
        learning_rate=0.06,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        eval_metric="mae",
        early_stopping_rounds=30,
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )
    reg.fit(X, y, eval_set=[(evX, evy)], verbose=False)
    return reg


def main() -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)
    df = _load()
    print(f"rows: {len(df)}")

    # Semantics for similarity / exploration
    embedder = _load_embedder()
    combos = df["content"].fillna("").tolist()
    embeddings = embedder.encode(combos, batch_size=128, show_progress_bar=True)
    embeddings = embeddings.astype(np.float32)

    # --- Feature pipeline ---------------------------------------------------
    # Compact pipeline: only features the app can populate from a free-text
    # description (+ inferred metadata), so serving and training share the
    # same feature space and the model actually discriminates user input.
    pipe = FeaturePipeline(compact=True)
    pipe.fit(df)
    X = pipe.transform(df)
    print(f"n_features: {X.shape[1]}")

    views_bin = (df["views"] >= df["views"].median()).astype(int).to_numpy()
    eng_bin = (df["engagement"] >= df["engagement"].median()).astype(int).to_numpy()

    # --- Train / eval split (for reported metrics) --------------------------
    split = train_test_split(
        X, views_bin, eng_bin,
        df[["views", "engagement"]].to_numpy(),
        test_size=0.2, random_state=42,
    )
    X_tr, X_te, v_tr, v_te, e_tr, e_te, ynum_tr, ynum_te = split

    clf_views = _fit_binary_clf(X_tr, v_tr, X_te, v_te)
    clf_eng = _fit_binary_clf(X_tr, e_tr, X_te, e_te)

    reg_views = _fit_reg(X_tr, np.log1p(ynum_tr[:, 0]), X_te, np.log1p(ynum_te[:, 0]))
    reg_eng = _fit_reg(X_tr, np.log1p(ynum_tr[:, 1]), X_te, np.log1p(ynum_te[:, 1]))

    # Report accuracy vs majority baseline
    for name, clf, ytr, yte in (
        ("views", clf_views, v_tr, v_te),
        ("engagement", clf_eng, e_tr, e_te),
    ):
        ytr = np.asarray(ytr)
        yte = np.asarray(yte)
        base = float(np.mean(ytr))
        acc = float(clf.score(X_te, yte))
        print(f"[{name}] majority={base:.3f} accuracy={acc:.3f}")

    views_75 = float(df["views"].quantile(0.75))
    eng_75 = float(df["engagement"].quantile(0.75))
    views_25 = float(df["views"].quantile(0.25))
    eng_25 = float(df["engagement"].quantile(0.25))

    # --- Realistic per-bin stats --------------------------------------------
    # Ground the displayed "estimate" in the actual historical performance of
    # posts the model puts in each bucket, instead of a noisy absolute
    # regression prediction on unseen inputs.
    pred_views = clf_views.predict(X)
    pred_eng = clf_eng.predict(X)
    cond: dict = {}
    for key, pred, metric in [
        ("views_high", pred_views, "views"),
        ("views_low", 1 - pred_views, "views"),
        ("eng_high", pred_eng, "engagement"),
        ("eng_low", 1 - pred_eng, "engagement"),
    ]:
        mask = pred.astype(bool)
        sub = df[mask][metric]
        cond[key] = {
            "median": float(sub.median()) if len(sub) else None,
            "p25": float(sub.quantile(0.25)) if len(sub) else None,
            "p75": float(sub.quantile(0.75)) if len(sub) else None,
            "n": int(len(sub)),
        }

    # --- Representative peers for the Explore tab ---------------------------
    peers = _representative(df, embeddings, n=40)

    bundle = {
        "pipe": pipe,
        "clf_views": clf_views,
        "clf_eng": clf_eng,
        "reg_views": reg_views,
        "reg_eng": reg_eng,
        "views_median": float(df["views"].median()),
        "eng_median": float(df["engagement"].median()),
        "views_75": views_75,
        "eng_75": eng_75,
        "views_25": views_25,
        "eng_25": eng_25,
        "conditional_stats": cond,
        "feature_names": list(_feature_names_total(pipe, n_cat=len(pipe.cat_columns))),
        "metrics": {
            "views_accuracy": float(clf_views.score(X_te, v_te)),
            "engagement_accuracy": float(clf_eng.score(X_te, e_te)),
            "views_majority": float(np.mean(np.asarray(v_tr))),
            "engagement_majority": float(np.mean(np.asarray(e_tr))),
        },
    }
    joblib.dump(bundle, BUNDLE_PATH)
    print(f"wrote {BUNDLE_PATH}")

    similar = {
        "embeddings": embeddings,
        "rows": df.to_dict("records"),
        "peers": peers,
    }
    joblib.dump(similar, SIM_PATH)
    print(f"wrote {SIM_PATH}")


def _load_embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("all-MiniLM-L6-v2")


def _representative(df: pd.DataFrame, embeddings: np.ndarray, n: int) -> list:
    """Pick n diverse representative posts for the Explore tab."""
    if len(df) <= n:
        return df.to_dict("records")
    centroids = embeddings[np.linspace(0, len(embeddings) - 1, n).round().astype(int)]
    dists = cdist(centroids, embeddings)
    idx = np.argmin(dists, axis=1)
    idx = list(dict.fromkeys(idx.tolist()))[:n]
    return df.iloc[idx].to_dict("records")


def _feature_names_total(pipe: FeaturePipeline, n_cat: int) -> list:
    names = list(pipe.cat_columns) + list(pipe.json_columns) + list(pipe.base_columns)
    if not pipe.compact:
        names += (
            [f"hashtag_{t}" for t in pipe.hashtag_vocab]
            + [f"mention_{t}" for t in pipe.mention_vocab]
            + [f"emoji_{t}" for t in pipe.emoji_vocab]
        )
    return names


if __name__ == "__main__":
    main()
