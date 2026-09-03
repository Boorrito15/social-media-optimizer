"""High-level orchestrator: a raw DataFrame -> a fully-encoded (no y) feature DataFrame.

This is the *shared* feature pipeline used by both training (in
:mod:`models.linreg` and :mod:`models.clas`) and prediction (in
:mod:`models.predict`).

Crucially, NOTHING here:
    filters rows by validity/IQR/cutoff (those are train-only),
    splits into train/test (train-only),
    builds the y target (model-specific).
"""

from __future__ import annotations

import pandas as pd

from .categorical_ohe import encode_categorical
from .clustering import cluster_texts
from .drop import drop_alias_counts, drop_columns, round_duration
from .json_encode import encode_json_fields
from .json_parse import parse_json_column
from .post_encode import encode_post_features


def preprocess(df_raw: pd.DataFrame, state: dict) -> pd.DataFrame:
    """Apply the shared feature pipeline.

    Parameters
    ----------
    df_raw : pd.DataFrame
        Any DataFrame shaped like ``processed.csv``. Will be copied.
    state : dict
        The loaded :mod:`models.shared.build_state` artefact containing
        vocabularies, fitted KMeans models and the categorical OneHotEncoder.
    """
    df = df_raw.copy()

    description, json_by_idx = parse_json_column(df["description_json"])
    df["description"] = description
    df = df.drop(columns=["description_json"])

    df = pd.concat([df, encode_json_fields(df, json_by_idx, state["json_vocab"])], axis=1)
    df = encode_post_features(df, state)

    df["description_cluster"] = cluster_texts(
        df["description"].fillna("").tolist(), state["description_kmeans"]
    )
    df["title_cluster"] = cluster_texts(
        df["content"].fillna("").tolist(), state["title_kmeans"]
    )

    df = pd.concat([df, encode_categorical(df, state["platform_page_ohe"])], axis=1)

    df = drop_columns(df)
    df = round_duration(df)
    df = drop_alias_counts(df)
    return df
