"""Cleaning / normalisation logic for the master dataset.

Pure functions (pandas in/out) so they can be reused by scripts, notebooks and
tests. The source CSV uses a fixed vendor schema with `NZR - ...` prefixed
columns; this module maps those onto a stable, lowercase project schema.

Duplicate handling:
    A row is a *duplicate* when it shares the exact same post URL with an
    already-seen row. When duplicates exist we keep a single representative row
    (the one with the most views, i.e. most complete engagement) and drop the
    rest. Rows with a missing/blank URL are never treated as duplicates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Platforms kept for short-form content.
DEFAULT_PLATFORMS = ["FB", "IG", "TT", "YT"]

# Media types classified as short-form in this project.
SHORT_MEDIA_TYPES = ["Short Video"]

# Map raw vendor columns -> cleaner project schema.
COLUMN_MAP = {
    "NZR - Paid campaign name (Short)": "campaign",
    "NZR - Year": "year",
    "NZR - Page Name": "page",
    "NZR - Platform Code": "platform",
    "NZR - Media Type": "media_type",
    "NZR - Content Hierarchy L0": "category_l0",
    "NZR - Content Hierarchy L1": "category_l1",
    "NZR - Content Hierarchy L2": "category_l2",
    "NZR - Post URL": "url",
    "NZR - Post Content (Low Case)": "content",
    "NZR - Cost (NZD)": "cost_nzd",
    "NZR - Views": "views",
    "NZR - Hours": "hours",
    "NZR - Engagement": "engagement",
}

# Order of the project schema columns in the output.
COLUMN_ORDER = [
    "campaign", "year", "page", "platform", "media_type",
    "category_l0", "category_l1", "category_l2",
    "url", "content", "cost_nzd", "views", "hours", "engagement",
]

# Latest schema marker consumers use to detect structural changes.
SCHEMA_VERSION = 1


def _clean_string(s: pd.Series) -> pd.Series:
    """Trim strings; map empty/whitespace-only to NaN."""
    return s.astype("string").str.strip().replace(r"^\s*$", pd.NA, regex=True)


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map raw vendor columns to the stable project schema."""
    present = {k: v for k, v in COLUMN_MAP.items() if k in df.columns}
    out = df.rename(columns=present)
    missing = [v for v in COLUMN_ORDER if v not in out.columns]
    if missing:
        raise ValueError(f"Input is missing expected columns: {missing}")
    return out


def remove_duplicate_links(df: pd.DataFrame, url_col: str = "url") -> pd.DataFrame:
    """Drop rows whose URL collides with a previously seen row.

    For each group of rows sharing the same URL we keep the single row with the
    most complete engagement (highest 'views', then 'engagement', then 'hours')
    and drop the rest. Rows with a blank/NA URL are preserved as-is.

    Returns a copy of ``df`` with duplicate-link rows removed.
    """
    df = df.copy()

    # Normalise the URL (strip whitespace) before grouping.
    mask = df[url_col].notna() & df[url_col].astype(str).str.strip().ne("")
    norm_url = df.loc[mask, url_col].astype(str).map(str.strip).str.lower()

    # Assign a group id per row: a distinct id per unique URL, and a unique
    # singleton id for every blank-URL row so those are never collapsed.
    levels, _ = pd.factorize(norm_url)
    group = pd.Series(index=df.index, dtype="int64")
    # Unique negative ids for each blank-URL row (so none are grouped).
    n_blank = int((~mask).sum())
    group.loc[~mask] = -(np.arange(1, n_blank + 1, dtype="int64"))
    group.loc[mask] = levels

    work = df.assign(_group=group, _valid=df[["views", "engagement", "hours"]].notna().sum(axis=1))
    work = work.sort_values(
        ["_group", "_valid", "views", "engagement", "hours"],
        ascending=[True, False, False, False, False],
        na_position="last",
    )
    # Keep the first row of every group (<=1 per unique URL, incl. each blank row).
    kept = work.groupby("_group", sort=False, dropna=False).head(1)
    kept = kept.drop(columns=["_group", "_valid"]).reset_index(drop=True)
    return kept


def clean_dataframe(
    df: pd.DataFrame,
    platforms: list[str] | None = None,
    media_types: list[str] | None = None,
    drop_duplicates: bool = True,
    min_full_rows: int = 0,
) -> pd.DataFrame:
    """One-stop cleaning: rename, filter to short-form, dedupe and sanitise.

    Returns the cleaned dataframe. A :class:`~src.ingestion.summary.CleanSummary`
    describing what was removed at each stage is attached to the result's
    ``df.attrs["summary"]``.

    Parameters
    ----------
    df : raw master dataframe with vendor column names.
    platforms : platform codes to keep (default: FB, IG, TT, YT).
    media_types : media type values classified as short-form (default Short Video).
    drop_duplicates : remove duplicate-link rows (keep the most-engaged one).
    min_full_rows : drop rows with fewer valid *engagement* columns than this.
    """
    from src.ingestion.summary import CleanSummary

    platforms = platforms or DEFAULT_PLATFORMS
    media_types = media_types or SHORT_MEDIA_TYPES

    summary = CleanSummary(
        platforms=platforms,
        media_types=media_types,
        dedupe_enabled=drop_duplicates,
        min_full_rows=min_full_rows,
    )
    summary.counts["input"] = len(df)

    df = rename_columns(df)

    # Filter to targeted platforms & short-form media type.
    df = df[df["platform"].astype(str).str.strip().str.upper().isin(platforms)].copy()
    summary.counts["after_platform_filter"] = len(df)

    df = df[df["media_type"].astype(str).str.strip().isin(media_types)].copy()
    summary.counts["after_media_type_filter"] = len(df)

    # Trim strings and fix dtypes.
    for col in ["campaign", "page", "platform", "media_type",
                "category_l0", "category_l1", "category_l2", "url", "content"]:
        df[col] = _clean_string(df[col])
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    for col in ["cost_nzd", "views", "hours", "engagement"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    summary.counts["after_type_clean"] = len(df)

    # Drop rows that are unusable (no URL and no content).
    df = df.dropna(subset=["url", "content"], how="all")
    summary.counts["after_drop_blank"] = len(df)

    if drop_duplicates:
        df = remove_duplicate_links(df)
    summary.counts["after_dedupe"] = len(df)

    # Optionally keep only rows meeting a minimum completeness bar.
    if min_full_rows > 0:
        engagement_cols = ["views", "hours", "engagement"]
        df = df[df[engagement_cols].notna().sum(axis=1) >= min_full_rows]
    summary.counts["after_min_rows"] = len(df)

    df = df.dropna(how="all").reset_index(drop=True)
    summary.counts["output"] = len(df)

    # Tag the source schema + run summary in attributes for introspection.
    # NOTE: attrs are serialised into Parquet metadata, so only store plain
    # JSON-serialisable values here (reconstruct CleanSummary via from_dict).
    df.attrs["schema_version"] = SCHEMA_VERSION
    df.attrs["summary"] = summary.to_json()
    return df[COLUMN_ORDER]
