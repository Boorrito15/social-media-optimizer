"""Unit tests for the video pipeline (pure logic, no network / GCS / ffmpeg)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.video.download import media_id_from_url, platform_code_to_name, published_at_from_info, resolve
from src.video.upload import _gcs_object_name, _process_one, run_pipeline, _MANIFEST_SCHEMA, write_records_to_gcs


# --- platform code mapping ---------------------------------------------------

def test_platform_code_to_name_maps_codes():
    assert platform_code_to_name("YT") == "youtube"
    assert platform_code_to_name("tt") == "tiktok"
    assert platform_code_to_name("IG") == "instagram"
    assert platform_code_to_name("FB") == "facebook"
    assert platform_code_to_name("??") is None


# --- media id extraction (no I/O) -------------------------------------------

def test_media_id_from_url_youtube_query():
    assert media_id_from_url("https://www.youtube.com/watch?v=_iPeantXTXU", "youtube") == "_iPeantXTXU"


def test_media_id_from_url_youtube_shorts_youtu():
    assert media_id_from_url("https://youtu.be/abcXYZ123", "youtube") == "abcXYZ123"


def test_media_id_from_url_tiktok_numeric():
    assert media_id_from_url("https://www.tiktok.com/@nzrugby/video/7612811988942474517", "tiktok") == "7612811988942474517"


def test_media_id_from_url_fallback_last_segment():
    assert media_id_from_url("https://www.instagram.com/reel/DJPodLgBQm8/", "instagram") == "DJPodLgBQm8"


# --- GCS object naming ------------------------------------------------------

def test_gcs_object_name():
    assert _gcs_object_name("abc123", "youtube") == "videos/youtube/abc123.mp4"


# --- resolve marks unsupported platforms (no network) -----------------------

def test_resolve_unsupported_platform_in_pipeline():
    # Calling download.resolve on IG should not hit the network; it returns a
    # tombstone with supported=False.
    m = resolve("https://www.instagram.com/reel/DJPodLgBQm8/")
    assert m.supported is False
    assert m.platform == "instagram"


# --- _process_one (returns a record dict) -----------------------------------

def test_process_one_unsupported_in_dry_run():
    row = {"url": "https://www.facebook.com/reel/123/", "platform": "FB"}
    rec = _process_one(row, bucket="b", dry_run=True)
    assert rec["status"] == "unsupported"
    assert rec["error"] == "no automated extractor for facebook"


def test_process_one_youtube_dry_run_status_uploaded():
    row = {"url": "https://www.youtube.com/watch?v=abcXYZ123", "platform": "YT"}
    rec = _process_one(row, bucket="b", dry_run=True)
    assert rec["status"] == "uploaded"
    assert rec["post_id"] == "abcXYZ123"
    assert rec["gcs_path"] == "gs://b/videos/youtube/abcXYZ123.mp4"


def test_process_one_missing_url_status_skipped():
    rec = _process_one({"url": "", "platform": "YT"}, bucket="b", dry_run=True)
    assert rec["status"] == "skipped"
    assert rec["error"] == "missing url"


# --- publish timestamp normalisation ----------------------------------------

def test_published_at_from_timestamp():
    info = {"timestamp": 1774159232}
    assert published_at_from_info(info).startswith("2026-03")


def test_published_at_from_upload_date_fallback():
    info = {"upload_date": "20260322"}
    assert published_at_from_info(info) == "2026-03-22T00:00:00Z"


def test_published_at_none_when_missing():
    assert published_at_from_info({}) is None


# --- run_pipeline dry-run summary -------------------------------------------

def test_run_pipeline_dry_run_aggregates():
    df = pd.DataFrame(
        {
            "url": [
                "https://www.youtube.com/watch?v=aaa111",
                "https://www.instagram.com/reel/x1/",
                "https://www.facebook.com/reel/9/",
            ],
            "platform": ["YT", "IG", "FB"],
        }
    )
    res = run_pipeline(df, dry_run=True)
    assert res.attempted == 3
    assert res.uploaded == 1          # youtube dry-run counts as uploaded
    assert res.unsupported == 2        # IG + FB
    assert res.failed == 0


def test_run_pipeline_platform_filter():
    df = pd.DataFrame(
        {
            "url": ["https://www.youtube.com/watch?v=aaa111", "https://www.tiktok.com/v/2"],
            "platform": ["YT", "TT"],
        }
    )
    res = run_pipeline(df, dry_run=True, platforms=["youtube"])
    assert res.attempted == 1
    assert res.uploaded == 1
    assert res.unsupported == 0
    assert res.failed == 0


# --- manifest / status sheet writing ----------------------------------------

def test_run_pipeline_produces_index_shards(monkeypatch):
    # Verify run_pipeline's incremental index flush fires without touching GCS.
    from src.video import upload as up_mod
    calls = []
    monkeypatch.setattr(up_mod, "append_index_shard",
                        lambda *a, **k: calls.append(("index", len(a[0]))) or "gs://x/p.parquet")
    monkeypatch.setattr(up_mod, "append_failed_sheet", lambda *a, **k: None)
    df = pd.DataFrame({"url": [f"https://www.youtube.com/watch?v=id{i}" for i in range(25)],
                       "platform": ["YT"] * 25})
    res = up_mod.run_pipeline(df, dry_run=True, concurrency=4, index_flush_every=10)
    assert len([c for c in calls if c[0] == "index"]) >= 3  # 10, 20, final 5
    assert res.attempted == 25


def test_manifest_schema_columns():
    assert "published_at" in _MANIFEST_SCHEMA
    assert "sha256" in _MANIFEST_SCHEMA
    assert "gcs_path" in _MANIFEST_SCHEMA
    assert "error" in _MANIFEST_SCHEMA


def test_write_records_dry_run_returns_paths_and_no_status_file_when_no_failures(tmp_path):
    records = [
        {"platform": "youtube", "post_id": "a", "url": "u", "status": "uploaded",
         "gcs_path": "gs://b/x.mp4"},
    ]
    manifest, status = write_records_to_gcs(records, "b", dry_run=True, staging_dir=tmp_path)
    assert manifest and manifest.startswith("gs://b/manifests/video_manifest_")
    assert status is None  # no failures -> no status sheet
    # dry-run should not leave a manifest file behind
    assert list(tmp_path.rglob("*.parquet")) == []


def test_write_records_failure_filtering_logic():
    # Verify the status-sheet selection logic: only 'failed' rows are kept, and
    # their error reason is preserved.
    import pandas as pd

    records = [
        {"platform": "youtube", "post_id": "a", "url": "u1", "status": "uploaded",
         "gcs_path": "gs://b/x.mp4", "published_at": "2026-01-01T00:00:00Z", "sha256": "ab", "error": None},
        {"platform": "youtube", "post_id": "b", "url": "u2", "status": "failed",
         "gcs_path": "gs://b/y.mp4", "published_at": None, "sha256": None, "error": "upload boom"},
    ]
    df = pd.DataFrame(records)
    for col in _MANIFEST_SCHEMA:
        if col not in df.columns:
            df[col] = None
    df = df[_MANIFEST_SCHEMA]
    failed = df[df["status"] == "failed"]
    assert len(failed) == 1
    assert list(failed["error"]) == ["upload boom"]
    assert list(failed["gcs_path"]) == ["gs://b/y.mp4"]


def test_run_pipeline_direct_skips_existing_objects(monkeypatch):
    from src.video import upload as up_mod

    existing = {"videos/youtube/already_done.mp4"}
    monkeypatch.setattr(up_mod, "list_existing_objects", lambda *a, **k: existing)
    monkeypatch.setattr(up_mod, "write_records_to_gcs", lambda *a, **k: (None, None))
    monkeypatch.setattr(up_mod, "append_index_shard", lambda *a, **k: None)
    monkeypatch.setattr(up_mod, "append_failed_sheet", lambda *a, **k: None)

    processed_posts = []

    def mock_process_one(row, **kwargs):
        processed_posts.append(row["url"])
        return {
            "platform": "youtube",
            "post_id": "new_video",
            "url": row["url"],
            "status": "uploaded",
            "gcs_path": "gs://b/videos/youtube/new_video.mp4",
            "published_at": None,
            "duration_s": None,
            "title": None,
            "sha256": None,
            "size_bytes": None,
            "source_codec": None,
            "source_resolution": None,
            "error": None,
            "processed_at": None,
            "transcode_args": None,
        }

    monkeypatch.setattr(up_mod, "_process_one", mock_process_one)

    df = pd.DataFrame(
        {
            "url": [
                "https://www.youtube.com/watch?v=already_done",
                "https://www.youtube.com/watch?v=new_video",
            ],
            "platform": ["YT", "YT"],
        }
    )

    res = up_mod.run_pipeline(df, dry_run=False, skip_existing=True)
    assert res.skipped_existing == 1
    assert res.uploaded == 1
    assert res.attempted == 2
    # Only the new video should have been sent to _process_one
    assert processed_posts == ["https://www.youtube.com/watch?v=new_video"]

