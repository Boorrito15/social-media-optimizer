"""Train and export the Keras 2-head classification model.

Replicates the approach from notebooks/rob.ipynb (2-head section, cell 116+)
but:
  * Saves via model.save('.keras') instead of broken joblib.dump()
  * Saves the fitted FeaturePipeline alongside for reproducible inference
  * Uses the FULL feature space (non-compact) so the Keras model has the
    same ~6,400-dim input the notebook achieved 82-84% test accuracy on.

Usage:
    python -m src.ml.train_keras
"""

from __future__ import annotations

import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# TensorFlow/Keras
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import keras
from keras import layers, Sequential, Model, Input

from src.ml.features import FeaturePipeline, _parse_json_list

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "processed.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "data", "models")
BUNDLE_PATH = os.path.join(MODELS_DIR, "bundle_keras.joblib")
MODEL_PATH = os.path.join(MODELS_DIR, "keras_model.keras")
N_BINS = 2


def _extract_description(text) -> str:
    if not isinstance(text, str):
        return ""
    try:
        d = json.loads(text)
    except Exception:
        return ""
    if not isinstance(d, dict):
        return ""
    return str(d.get("play_by_play") or "").strip()


def _load_and_filter() -> pd.DataFrame:
    df = pd.read_csv(DATA_CSV)
    df = df[df["views"].notna() & df["engagement"].notna()]
    df = df[df["views"] > 0]
    df = df[df["engagement"] > 0]
    df = df[df["content"].notna()]
    df["description"] = df["description_json"].map(_extract_description)
    df = df.reset_index(drop=True)
    return df


def _iqr_binned(series: pd.Series, n: int = N_BINS) -> np.ndarray:
    """Bin a series into n buckets using IQR-based edges (as notebook does)."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    edges = np.linspace(q1 - 1.5 * iqr, q3 + 1.5 * iqr, n + 1)
    return np.clip(np.digitize(series.to_numpy(), edges) - 1, 0, n - 1).astype(int)


def _build_model(input_dim: int) -> Model:
    """Build the 2-head Keras model — matches notebook cell 116."""
    shared_input = Input(shape=(input_dim,), name="input")

    x = layers.Dense(1611, activation="relu")(shared_input)
    x = layers.Dropout(rate=0.2)(x)
    x = layers.Dense(3200, activation="relu")(x)
    x = layers.Dropout(rate=0.2)(x)
    x = layers.Dense(1611, activation="relu")(x)
    x = layers.Dropout(rate=0.2)(x)

    views_head = layers.Dense(N_BINS, activation="softmax", name="views")(x)
    engagement_head = layers.Dense(N_BINS, activation="softmax", name="engagement")(x)

    model = Model(inputs=shared_input, outputs=[views_head, engagement_head])
    model.compile(
        loss={"views": "categorical_crossentropy", "engagement": "categorical_crossentropy"},
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        metrics={"views": "accuracy", "engagement": "accuracy"},
    )
    return model


def main() -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)

    # 1. Load data
    df = _load_and_filter()
    print(f"rows: {len(df)}")

    # 2. Fit FeaturePipeline in FULL (non-compact) mode
    pipe = FeaturePipeline(compact=False)
    pipe.fit(df)

    # 3. Transform — use dense array for Keras
    X_sparse = pipe.transform(df)
    X = X_sparse.toarray().astype(np.float32)
    n_features = X.shape[1]
    print(f"n_features: {n_features}")

    # 4. Create binned targets (exactly as notebook does)
    views_bin = _iqr_binned(df["views"])
    engagement_bin = _iqr_binned(df["engagement"])

    y_views = keras.utils.to_categorical(views_bin, num_classes=N_BINS)
    y_engagement = keras.utils.to_categorical(engagement_bin, num_classes=N_BINS)

    # 5. Train/eval split
    split = train_test_split(
        X, y_views, y_engagement,
        test_size=0.2, random_state=42,
    )
    X_tr, X_te, yv_tr, yv_te, ye_tr, ye_te = split

    # Standardize numeric columns
    NUMERIC_COLS = [c for c in [
        "duration_seconds", "description_cluster", "title_cluster",
        "n_mentions", "n_hashtags", "n_emojis",
        "hashtag_count", "mention_count", "emoji_count",
    ] if c in pipe.base_columns or c in df.columns]
    NUMERIC_COLS = [c for c in NUMERIC_COLS if c in pipe.base_columns]

    # Fit a scaler on the numeric parts only
    numeric_indices = [i for i, c in enumerate(pipe.base_columns) if c in NUMERIC_COLS]
    binary_indices = [i for i in range(X.shape[1]) if i not in numeric_indices]

    scaler_X = StandardScaler()
    if numeric_indices:
        X_tr_num = scaler_X.fit_transform(X_tr[:, numeric_indices])
        X_te_num = scaler_X.transform(X_te[:, numeric_indices])
        X_tr_scaled = np.zeros_like(X_tr)
        X_te_scaled = np.zeros_like(X_te)
        X_tr_scaled[:, numeric_indices] = X_tr_num
        X_te_scaled[:, numeric_indices] = X_te_num
        if binary_indices:
            X_tr_scaled[:, binary_indices] = X_tr[:, binary_indices]
            X_te_scaled[:, binary_indices] = X_te[:, binary_indices]
    else:
        X_tr_scaled = X_tr
        X_te_scaled = X_te

    # 6. Build & train model
    model = _build_model(n_features)
    model.summary()

    early_stopping = keras.callbacks.EarlyStopping(
        patience=20, restore_best_weights=True, monitor="val_loss"
    )
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        factor=0.5, patience=3, monitor="val_loss"
    )

    history = model.fit(
        X_tr_scaled,
        {"views": yv_tr, "engagement": ye_tr},
        batch_size=32,
        epochs=1000,
        validation_split=0.2,
        callbacks=[early_stopping, reduce_lr],
        verbose=1,
    )

    # 7. Evaluate on test set
    results = model.evaluate(
        X_te_scaled,
        {"views": yv_te, "engagement": ye_te},
        verbose=0,
    )

    # Parse multi-output metrics
    # results order: total_loss, views_loss, engagement_loss, views_accuracy, engagement_accuracy
    test_views_acc = float(results[3])
    test_eng_acc = float(results[4])

    # Baseline: majority class
    v_majority = int(pd.Series(views_bin).mode().iloc[0])
    e_majority = int(pd.Series(engagement_bin).mode().iloc[0])
    truth_views = yv_te.argmax(axis=1)
    truth_eng = ye_te.argmax(axis=1)
    baseline_v = float(np.mean(truth_views == v_majority))
    baseline_e = float(np.mean(truth_eng == e_majority))

    print(f"\n--- Results ---")
    print(f"Views  accuracy:      {test_views_acc:.4f}  (baseline: {baseline_v:.4f})")
    print(f"Engagement accuracy:  {test_eng_acc:.4f}  (baseline: {baseline_e:.4f})")

    # 8. Compute per-bucket conditional stats (for anchoring estimates)
    pred_views = model.predict(X_te_scaled, verbose=0)[0].argmax(axis=1)
    pred_eng = model.predict(X_te_scaled, verbose=0)[1].argmax(axis=1)
    pred_all_views = model.predict(X, verbose=0)[0].argmax(axis=1)
    pred_all_eng = model.predict(X, verbose=0)[1].argmax(axis=1)

    cond: dict = {}
    for key, pred_arr, metric in [
        ("views_high", pred_all_views, "views"),
        ("views_low", 1 - pred_all_views, "views"),
        ("eng_high", pred_all_eng, "engagement"),
        ("eng_low", 1 - pred_all_eng, "engagement"),
    ]:
        mask = pred_arr.astype(bool)
        sub = df[mask][metric]
        cond[key] = {
            "median": float(sub.median()) if len(sub) else None,
            "p25": float(sub.quantile(0.25)) if len(sub) else None,
            "p75": float(sub.quantile(0.75)) if len(sub) else None,
            "n": int(len(sub)),
        }

    # 9. Save model + pipeline + metadata as a bundle
    # Save Keras model in proper format
    model.save(MODEL_PATH)
    print(f"wrote {MODEL_PATH}")

    bundle = {
        "pipe": pipe,
        "scaler_X": scaler_X,
        "numeric_cols": NUMERIC_COLS,
        "views_median": float(df["views"].median()),
        "eng_median": float(df["engagement"].median()),
        "views_75": float(df["views"].quantile(0.75)),
        "eng_75": float(df["engagement"].quantile(0.75)),
        "views_25": float(df["views"].quantile(0.25)),
        "eng_25": float(df["engagement"].quantile(0.25)),
        "conditional_stats": cond,
        "metrics": {
            "views_accuracy": test_views_acc,
            "engagement_accuracy": test_eng_acc,
            "views_majority": float(np.mean(views_bin == v_majority)),
            "engagement_majority": float(np.mean(engagement_bin == e_majority)),
            "baseline_views": baseline_v,
            "baseline_engagement": baseline_e,
        },
        "model_path": MODEL_PATH,
        "n_features": n_features,
        "feature_mode": "full",
    }
    joblib.dump(bundle, BUNDLE_PATH)
    print(f"wrote {BUNDLE_PATH}")


if __name__ == "__main__":
    main()