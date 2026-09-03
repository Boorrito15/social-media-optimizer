"""Canonical DROP / round / finalise column operations."""

from __future__ import annotations

import pandas as pd

DROP_COLS = [
    "campaign",
    "platform",
    "media_type",
    "category_l0",
    "category_l1",
    "category_l2",
    "page",
    "platform",
    "url",
    "content",
    "hours",
    "description",
    "cost_nzd",
    "year",
]


def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop any columns in :data:`DROP_COLS` that are present in ``df``.

    The early ``platform`` duplicate is preserved because historical
    pipelines accidentally added ``platform`` twice; that's harmless on the
    receiving end (pandas drop ignores absent columns).
    """
    return df.drop(columns=[c for c in DROP_COLS if c in df.columns])


def round_duration(df: pd.DataFrame) -> pd.DataFrame:
    """Round ``duration_seconds`` to int (mirrors the original notebook)."""
    df = df.copy()
    if "duration_seconds" in df.columns:
        df["duration_seconds"] = df["duration_seconds"].round().astype(int)
    return df


def drop_alias_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the unaliased ``n_mentions`` / ``n_hashtags`` / ``n_emojis`` columns.

    Their values were copied onto the ``*_count`` columns during the
    encoding step; the originals are no longer needed.
    """
    return df.drop(
        columns=[c for c in ("n_hashtags", "n_emojis", "n_mentions") if c in df.columns]
    )
