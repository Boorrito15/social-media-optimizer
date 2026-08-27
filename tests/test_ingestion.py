"""Tests for the Phase 2 cleaning / de-duplication logic."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.clean import clean_dataframe, remove_duplicate_links


SAMPLE = pd.DataFrame(
    {
        "NZR - Paid campaign name (Short)": ["a", "a", "b", "b"],
        "NZR - Year": [2024, 2024, 2025, 2025],
        "NZR - Page Name": ["ABXV", "ABXV", "ABXV", "ABXV"],
        "NZR - Platform Code": ["FB", "FB", "IG", "IG"],
        "NZR - Media Type": ["Short Video", "Short Video", "Display", "Short Video"],
        "NZR - Content Hierarchy L0": ["-", "-", "-", "-"],
        "NZR - Content Hierarchy L1": ["-", "-", "-", "-"],
        "NZR - Content Hierarchy L2": ["-", "-", "-", "-"],
        "NZR - Post URL": [
            "https://www.facebook.com/reel/1/",
            "https://www.facebook.com/reel/1/",
            "https://www.instagram.com/reel/2/",
            "https://www.instagram.com/reel/2/",
        ],
        "NZR - Post Content (Low Case)": ["one", "one", "two", "two"],
        "NZR - Cost (NZD)": [0.0, 0.0, 0.0, 0.0],
        "NZR - Views": [10, 500, 20, 20],
        "NZR - Hours": [1.0, 1.0, 1.0, 1.0],
        "NZR - Engagement": [2, 50, 3, 3],
    }
)


def test_remove_duplicate_links_keeps_most_engaged():
    cl = clean_dataframe(SAMPLE, platforms=["FB", "IG", "TT", "YT"], drop_duplicates=False)
    # Duplicate FB URL in input; remove_duplicate_links operates on cleaned schema.
    out = remove_duplicate_links(cl.copy())
    # No duplicate URLs overall; the FB dup keeps the 500-view row.
    assert out["url"].duplicated().sum() == 0
    fb = out[out["url"].str.contains("facebook")]
    assert list(fb["views"]) == [500]


def test_clean_dataframe_filters_platform_and_media_type():
    out = clean_dataframe(SAMPLE, platforms=["FB"], drop_duplicates=False)
    # Only FB + Short Video rows survive.
    assert set(out["platform"]) == {"FB"}
    assert list(out["platform"]) == ["FB", "FB"]
    assert (out["media_type"] == "Short Video").all()


def test_clean_dataframe_dedupes_across_full_pipeline():
    out = clean_dataframe(SAMPLE, platforms=["FB", "IG", "TT", "YT"])
    # 2 short-video FB rows + 1 short-video IG row -> 2 after FB dedupe.
    assert out["url"].duplicated().sum() == 0
    assert len(out) == 2


def test_blank_url_rows_preserved():
    df = SAMPLE.copy()
    # Make one short-video FB row have a blank URL.
    df.loc[df["NZR - Post URL"] == "https://www.facebook.com/reel/1/", "NZR - Post URL"] = ""
    out = clean_dataframe(df, platforms=["FB", "IG", "TT", "YT"], drop_duplicates=True)
    # Blank-URL row is preserved (not treated as a duplicate), no crash.
    assert out["url"].isna().any() or (out["url"].fillna("") == "").any()
    assert len(out) >= 1


def test_clean_summary_reports_removed_counts():
    out = clean_dataframe(SAMPLE, platforms=["FB", "IG", "TT", "YT"], drop_duplicates=True)
    from src.ingestion.summary import CleanSummary

    s = CleanSummary.from_dict(out.attrs["summary"])
    # Input had 4 rows; dedupe removed 1 (the FB duplicate).
    assert s.counts["input"] == 4
    assert s.dropped.get("after_dedupe", 0) == 1
    # The "Display" row on IG was dropped by the media-type filter.
    assert s.counts["after_media_type_filter"] == 3
    assert (s.dropped.get("after_media_type_filter", 0) == 1 or
            s.dropped.get("after_platform_filter", 0) == 0)


def test_no_dedupe_keeps_all_rows_and_reports_zero_removed():
    out = clean_dataframe(SAMPLE, platforms=["FB", "IG", "TT", "YT"], drop_duplicates=False)
    from src.ingestion.summary import CleanSummary

    s = CleanSummary.from_dict(out.attrs["summary"])
    # No de-dupe stage -> dropped dedupe is 0.
    assert s.counts["after_dedupe"] == s.counts["after_media_type_filter"]
    assert s.dropped.get("after_dedupe", 0) == 0
    assert not s.dedupe_enabled
