"""
Miscellaneous utility functions for processing social media video data.

This module provides scoring and sorting logic to rank videos by views and
engagement, using percentile-based scales to classify what counts as "low",
"medium", "high", or "very high" for each metric.
"""

import pandas as pd
import numpy as np
from typing import Literal, Optional

# ---------------------------------------------------------------------------
# Default percentile thresholds for the five-tier scale
# ---------------------------------------------------------------------------
_VIEW_BINS = [0.0, 0.20, 0.40, 0.60, 0.80, 1.0]
_ENG_BINS  = [0.0, 0.20, 0.40, 0.60, 0.80, 1.0]

_VIEW_LABELS = ["Very Low", "Low", "Medium", "High", "Very High"]
_ENG_LABELS  = ["Very Low", "Low", "Medium", "High", "Very High"]

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def add_view_scale(
    df: pd.DataFrame,
    view_col: str = "views",
    bins: Optional[list[float]] = None,
    labels: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Add a `view_scale` column based on percentile bins of *view_col*.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    view_col : str
        Name of the numeric column containing view counts.
    bins : list[float] or None
        Percentile cut points (0‑1 range).  Defaults to
        ``[0, .20, .40, .60, .80, 1]``.
    labels : list[str] or None
        Labels for each bin.  Must be ``len(bins) - 1``.  Defaults to
        ``["Very Low", "Low", "Medium", "High", "Very High"]``.

    Returns
    -------
    pd.DataFrame
        The original DataFrame with an extra ``view_scale`` column.
    """
    bins = bins or _VIEW_BINS
    labels = labels or _VIEW_LABELS
    return _add_percentile_scale(df, col=view_col, bins=bins, labels=labels,
                                 out_col="view_scale")


def add_engagement_scale(
    df: pd.DataFrame,
    eng_col: str = "engagement",
    bins: Optional[list[float]] = None,
    labels: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Add an ``engagement_scale`` column based on percentile bins.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    eng_col : str
        Name of the numeric column containing engagement counts.
    bins : list[float] or None
        Percentile cut points (0‑1 range).  Defaults to
        ``[0, .20, .40, .60, .80, 1]``.
    labels : list[str] or None
        Labels for each bin.  Must be ``len(bins) - 1``.  Defaults to
        ``["Very Low", "Low", "Medium", "High", "Very High"]``.

    Returns
    -------
    pd.DataFrame
        The original DataFrame with an extra ``engagement_scale`` column.
    """
    bins = bins or _ENG_BINS
    labels = labels or _ENG_LABELS
    return _add_percentile_scale(df, col=eng_col, bins=bins, labels=labels,
                                 out_col="engagement_scale")


def add_combined_score(
    df: pd.DataFrame,
    view_weight: float = 0.5,
    eng_weight: float = 0.5,
    view_col: str = "views",
    eng_col: str = "engagement",
    out_col: str = "combined_score",
) -> pd.DataFrame:
    """Add a normalised combined score that blends views and engagement.

    Each metric is min‑max scaled to [0, 1] before being combined with the
    supplied weights.  The result is added as a new column.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    view_weight : float
        Weight for the normalised view count (default 0.5).
    eng_weight : float
        Weight for the normalised engagement (default 0.5).
    view_col : str
        Column name for view counts (default ``"views"``).
    eng_col : str
        Column name for engagement (default ``"engagement"``).
    out_col : str
        Name of the output column (default ``"combined_score"``).

    Returns
    -------
    pd.DataFrame
        The original DataFrame with the extra ``combined_score`` column.
    """
    df = df.copy()

    views_norm = _min_max_scale(df[view_col].values)
    eng_norm   = _min_max_scale(df[eng_col].values)

    df[out_col] = (view_weight * views_norm + eng_weight * eng_norm).round(4)
    return df


def sort_by_combined(
    df: pd.DataFrame,
    view_weight: float = 0.5,
    eng_weight: float = 0.5,
    ascending: bool = False,
    view_col: str = "views",
    eng_col: str = "engagement",
) -> pd.DataFrame:
    """Return the DataFrame sorted by the combined score (views + engagement).

    This is a convenience wrapper that calls ``add_combined_score`` and then
    sorts by the resulting column.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    view_weight : float
        Weight for the normalised view count (default 0.5).
    eng_weight : float
        Weight for the normalised engagement (default 0.5).
    ascending : bool
        Sort ascending (default ``False`` → highest first).
    view_col : str
        Column name for view counts.
    eng_col : str
        Column name for engagement.

    Returns
    -------
    pd.DataFrame
        Sorted DataFrame with scale columns attached.
    """
    df = add_combined_score(df, view_weight=view_weight, eng_weight=eng_weight,
                            view_col=view_col, eng_col=eng_col)
    df = add_view_scale(df, view_col=view_col)
    df = add_engagement_scale(df, eng_col=eng_col)
    return df.sort_values("combined_score", ascending=ascending).reset_index(drop=True)


def load_and_sort(
    path: str = "data/processed/processed.csv",
    view_weight: float = 0.5,
    eng_weight: float = 0.5,
    ascending: bool = False,
) -> pd.DataFrame:
    """Load the processed CSV, score & scale each video, and return a sorted
    DataFrame.

    Parameters
    ----------
    path : str
        Path to the processed CSV.
    view_weight : float
        Weight for the normalised view count.
    eng_weight : float
        Weight for the normalised engagement.
    ascending : bool
        Sort ascending (default ``False`` → highest first).

    Returns
    -------
    pd.DataFrame
        Sorted, scored, and scaled DataFrame.
    """
    df = pd.read_csv(path)
    return sort_by_combined(df, view_weight=view_weight, eng_weight=eng_weight,
                            ascending=ascending)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _add_percentile_scale(
    df: pd.DataFrame,
    col: str,
    bins: list[float],
    labels: list[str],
    out_col: str,
) -> pd.DataFrame:
    df = df.copy()
    # Compute percentile cut points from the data
    _, cut_points = pd.qcut(df[col], q=bins, retbins=True, duplicates="drop")
    # Use pd.cut so the labels are human-readable
    df[out_col] = pd.cut(
        df[col],
        bins=cut_points,
        labels=labels[: len(cut_points) - 1],
        include_lowest=True,
    )
    return df


def _min_max_scale(values: np.ndarray) -> np.ndarray:
    vmin, vmax = values.min(), values.max()
    if vmax == vmin:
        return np.zeros_like(values, dtype=float)
    return (values - vmin) / (vmax - vmin)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Sort social-media videos by views & engagement."
    )
    parser.add_argument(
        "--input", "-i", default="data/processed/processed.csv",
        help="Path to the processed CSV (default: data/processed/processed.csv)."
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Path to write the sorted CSV (default: print top 20 rows to stdout)."
    )
    parser.add_argument(
        "--view-weight", type=float, default=0.5,
        help="Weight for normalised views (default 0.5)."
    )
    parser.add_argument(
        "--eng-weight", type=float, default=0.5,
        help="Weight for normalised engagement (default 0.5)."
    )
    parser.add_argument(
        "--ascending", action="store_true",
        help="Sort ascending (lowest first)."
    )
    parser.add_argument(
        "--top", type=int, default=20,
        help="Number of rows to show when no --output is given (default 20)."
    )

    args = parser.parse_args()

    result = load_and_sort(
        path=args.input,
        view_weight=args.view_weight,
        eng_weight=args.eng_weight,
        ascending=args.ascending,
    )

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"Sorted CSV written to {args.output}")
    else:
        cols = ["url", "views", "engagement", "view_scale",
                "engagement_scale", "combined_score"]
        # Only show columns that exist
        show_cols = [c for c in cols if c in result.columns]
        pd.set_option("display.max_colwidth", 60)
        print(result[show_cols].head(args.top).to_string(index=False))