"""Train-time filter steps. NOT applied at predict time."""

from __future__ import annotations

import pandas as pd


def validity_mask(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where ``views <= 0`` or ``engagement <= 0``."""
    mask = (df["views"] > 0) & (df["engagement"] > 0)
    return df[mask].copy()


def iqr_slice(df: pd.DataFrame, bottom: float, top: float) -> pd.DataFrame:
    """Drop rows outside the IQR-style bounds for ``views`` and ``engagement``.

    Mirrors the original notebook:
        Q1 = quantile(bottom); Q3 = quantile(top); IQR = Q3 - Q1
        keep rows with value in [Q1 - 1.5*IQR, Q3 + 1.5*IQR]

    Parameters mirror :attr:`PipelineConfig.bottom_iqr_percentile` and
    :attr:`PipelineConfig.top_iqr_percentile`.
    """
    out = df
    for col in ("views", "engagement"):
        q1 = out[col].quantile(bottom)
        q3 = out[col].quantile(top)
        iqr = q3 - q1
        lo = q1 - 1.5 * iqr
        hi = q3 + 1.5 * iqr
        out = out[(out[col] >= lo) & (out[col] <= hi)]
    return out.copy()
