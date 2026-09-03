"""Build the ``pipeline_state.joblib`` artefact (vocabs + KMeans + OHE).

This runs **once** per configuration (or whenever ``RETRAIN_STATE`` is true)
and produces the small ``pipeline_state.joblib`` file shared by every
downstream SVM model and the ``predict_*`` API.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder, normalize

from ..config import (
    PIPELINE_STATE_VERSION,
    STATE_PATH,
    PipelineConfig,
    processed_csv_path,
)
from .json_parse import JSON_LIST_FIELDS, parse_json_column
from .post_encode import EMOJI_PATTERN, HASHTAG_PATTERN, MENTION_PATTERN


def _vocab_with_freq(pattern, transform, texts, min_freq: int) -> list[str]:
    counter = Counter(transform(v) for text in texts for v in pattern.findall(text))
    return sorted({v for v, n in counter.items() if n >= min_freq})


def build_state(
    config: PipelineConfig,
    source_csv: Path | str | None = None,
    *,
    force: bool = False,
) -> dict:
    """Build the shared state and write to ``STATE_PATH``.

    Parameters
    ----------
    config : PipelineConfig
        Used to determine ``min_vocab_frequency`` and the cluster counts.
    source_csv : path-like, optional
        Path to ``processed.csv``. Resolved from the env / default by default.
    force : bool
        If True, always rebuild even when an existing artefact is present.
    """
    csv_path = Path(source_csv) if source_csv else processed_csv_path()

    if (
        not force
        and STATE_PATH.exists()
    ):
        existing = joblib.load(STATE_PATH)
        if existing.get("version") == PIPELINE_STATE_VERSION and existing.get(
            "config"
        ) == config:
            return existing

    df = pd.read_csv(csv_path)
    df = df[df["description_json"].notna()].copy()

    description, json_by_idx = parse_json_column(df["description_json"])
    df["description"] = description

    json_vocab = {
        f: sorted(
            {v for d in json_by_idx.values() for v in (d.get(f, []) or [])}
        )
        for f in JSON_LIST_FIELDS
    }

    texts = df["content"].fillna("").astype(str)
    hashtag_vocab = _vocab_with_freq(
        HASHTAG_PATTERN, lambda v: v.casefold(), texts, config.min_vocab_frequency
    )
    mention_vocab = _vocab_with_freq(
        MENTION_PATTERN, lambda v: v.rstrip(".").casefold(), texts, config.min_vocab_frequency
    )
    emoji_vocab = _vocab_with_freq(
        EMOJI_PATTERN, lambda v: v, texts, config.min_vocab_frequency
    )

    from .clustering import _embedder

    descriptions = df["description"].fillna("").astype(str)
    X_desc = _embedder().encode(descriptions.tolist(), show_progress_bar=True)
    description_kmeans = KMeans(n_clusters=3, n_init="auto", random_state=42).fit(
        normalize(X_desc)
    )

    X_title = _embedder().encode(texts.tolist(), show_progress_bar=True)
    title_kmeans = KMeans(n_clusters=12, n_init="auto", random_state=42).fit(
        normalize(X_title)
    )

    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    ohe.fit(df[["platform", "page"]].astype(str))

    state = {
        "version": PIPELINE_STATE_VERSION,
        "config": config,
        "json_vocab": json_vocab,
        "hashtag_vocab": hashtag_vocab,
        "mention_vocab": mention_vocab,
        "emoji_vocab": emoji_vocab,
        "description_kmeans": description_kmeans,
        "title_kmeans": title_kmeans,
        "platform_page_ohe": ohe,
    }
    joblib.dump(state, STATE_PATH)
    return state
