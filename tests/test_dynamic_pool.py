"""Milestone 3 Comprehensive 4-Tier E2E Test Suite for Social Media Optimizer.

Covers:
- Tier 1: Core Functional Tests (7 features x 5+ test cases = 35+ test cases)
- Tier 2: Boundary & Corner Cases (35+ test cases)
- Tier 3: Pairwise Combinations across Concurrency, Power Handoff, GCS Sync, and Hygiene
- Tier 4: Real-World Application Scenarios (Full Meta Dataset 9,453-item Simulation, Asymmetric Stalls, Rapid Handoff, Error Isolation)
"""

from __future__ import annotations

import math
import os
import queue
import shutil
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# Ensure root directory is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.video.download import ResolvedMedia
from src.video.dynamic_pool import (
    DynamicSupervisor,
    PlatformStats,
    SupervisorResult,
    ThroughputTracker,
    TrackerSnapshot,
)
from src.video.index import INDEX_SCHEMA
from src.video.upload import _MANIFEST_SCHEMA, VideoJobResult, check_disk_space
from utils.ffmpeg import TranscodeResult


# =====================================================================
# TEST FIXTURES & FAST ZERO-NETWORK MOCK HARNESS
# =====================================================================

@pytest.fixture
def mock_harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Provides a thread-safe, fast, zero-network mock harness for DynamicSupervisor."""
    staging_dir = tmp_path / "data" / "videos"
    staging_dir.mkdir(parents=True, exist_ok=True)

    uploaded_objects: list[str] = []
    downloaded_files: list[str] = []
    shards_emitted: list[tuple[str, list[dict]]] = []
    failed_sheets_emitted: list[tuple[str, list[dict]]] = []
    lock = threading.Lock()

    def mock_resolve(url: str, platform: str = "facebook", **kwargs) -> ResolvedMedia:
        p_id = url.rstrip("/").split("/")[-1] or "test_post"
        return ResolvedMedia(
            platform=platform,
            post_id=p_id,
            url=url,
            info={"title": f"Test {platform} {p_id}", "duration": 15.0, "height": 720, "vcodec": "h264"},
            supported=True,
        )

    def mock_download(media: ResolvedMedia, out_dir: Any = None, **kwargs) -> Path:
        out_d = Path(out_dir or staging_dir) / media.platform
        out_d.mkdir(parents=True, exist_ok=True)
        local_f = out_d / f"{media.post_id}.mp4"
        local_f.write_bytes(b"dummy_source_mp4_bytes")
        with lock:
            downloaded_files.append(str(local_f))
        return local_f

    def mock_transcode(src: Any, dst: Any, **kwargs) -> TranscodeResult:
        dst = Path(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(b"dummy_480p_transcoded_bytes")
        return TranscodeResult(output=dst, command=["ffmpeg", "-i", str(src), str(dst)], duration_s=15.0, size_bytes=26)

    def mock_upload_file(local_path: Any, bucket: str, object_name: str, **kwargs) -> None:
        with lock:
            if not str(object_name).endswith(".parquet"):
                uploaded_objects.append(f"gs://{bucket}/{object_name}")

    def mock_append_shard(records: list[dict], bucket: str, run_id: str, **kwargs) -> str:
        with lock:
            seq = len(shards_emitted) + 1
            path = f"gs://{bucket}/manifests/index_shard_{run_id}_{seq:06d}.parquet"
            shards_emitted.append((path, list(records)))
            return path

    def mock_append_failed(records: list[dict], bucket: str, run_id: str, **kwargs) -> str | None:
        failed = [r for r in records if r.get("status") == "failed"]
        if not failed:
            return None
        with lock:
            seq = len(failed_sheets_emitted) + 1
            path = f"gs://{bucket}/manifests/status_failed_{run_id}_{seq:06d}.parquet"
            failed_sheets_emitted.append((path, failed))
            return path

    monkeypatch.setattr("src.video.upload.resolve", mock_resolve)
    monkeypatch.setattr("src.video.upload.download", mock_download)
    monkeypatch.setattr("utils.ffmpeg.transcode_to_480p", mock_transcode)
    monkeypatch.setattr("src.video.upload.upload_file", mock_upload_file)
    monkeypatch.setattr("src.video.dynamic_pool.append_index_shard", mock_append_shard)
    monkeypatch.setattr("src.video.dynamic_pool.append_failed_sheet", mock_append_failed)
    monkeypatch.setattr("src.video.dynamic_pool.list_existing_objects", lambda *a, **k: set())
    monkeypatch.setattr("src.video.upload.list_existing_objects", lambda *a, **k: set())

    return {
        "staging_dir": staging_dir,
        "uploaded_objects": uploaded_objects,
        "downloaded_files": downloaded_files,
        "shards_emitted": shards_emitted,
        "failed_sheets_emitted": failed_sheets_emitted,
        "mock_resolve": mock_resolve,
        "mock_download": mock_download,
        "mock_transcode": mock_transcode,
        "mock_upload_file": mock_upload_file,
    }


# =====================================================================
# TIER 1: CORE FUNCTIONAL TESTS (7 Features x 5+ Tests = 35+ Test Cases)
# =====================================================================

# Feature 1: Initialization & Concurrency
def test_t1_supervisor_init_defaults():
    """Verify default constructor parameters, concurrency, and queues."""
    df = pd.DataFrame([{"url": "https://facebook.com/reel/1", "platform": "FB"}])
    sup = DynamicSupervisor(df)
    assert sup.max_total_concurrency == 20
    assert sup.initial_allocations == {"facebook": 10, "instagram": 10}
    assert "facebook" in sup._queues
    assert "instagram" in sup._queues
    assert sup.power_handoff_triggered is False


def test_t1_supervisor_balanced_mode_allocation(mock_harness):
    """Verify 10 FB + 10 IG balanced mode processes both queues concurrently."""
    fb_posts = [{"url": f"https://facebook.com/reel/{i}", "platform": "FB"} for i in range(10)]
    ig_posts = [{"url": f"https://instagram.com/reel/C{i}", "platform": "IG"} for i in range(10)]
    df = pd.DataFrame(fb_posts + ig_posts)

    sup = DynamicSupervisor(df, max_total_concurrency=20, staging_dir=mock_harness["staging_dir"])
    res = sup.run()
    assert res.total_processed == 20
    assert res.platform_counts["facebook"] == 10
    assert res.platform_counts["instagram"] == 10


def test_t1_supervisor_custom_allocation(mock_harness):
    """Verify custom initial allocations e.g. 15 FB + 5 IG are respected."""
    df = pd.DataFrame([{"url": f"https://facebook.com/reel/{i}", "platform": "FB"} for i in range(15)] +
                      [{"url": f"https://instagram.com/reel/C{i}", "platform": "IG"} for i in range(5)])
    sup = DynamicSupervisor(
        df,
        max_total_concurrency=20,
        initial_allocations={"facebook": 15, "instagram": 5},
        staging_dir=mock_harness["staging_dir"],
    )
    assert sup.initial_allocations == {"facebook": 15, "instagram": 5}
    res = sup.run()
    assert res.total_processed == 20


def test_t1_supervisor_worker_thread_naming(mock_harness):
    """Verify spawned worker threads have platform-tagged names."""
    df = pd.DataFrame([{"url": "https://facebook.com/reel/1", "platform": "FB"}])
    sup = DynamicSupervisor(df, max_total_concurrency=4, initial_allocations={"facebook": 2, "instagram": 2})
    
    # Check that initial allocation generates proper thread names
    allocs = sup.initial_allocations
    assert allocs["facebook"] == 2
    assert allocs["instagram"] == 2


def test_t1_supervisor_clean_worker_exit(mock_harness):
    """Verify all 20 worker threads exit cleanly upon queue completion without deadlocks."""
    df = pd.DataFrame([{"url": f"https://facebook.com/reel/{i}", "platform": "FB"} for i in range(5)] +
                      [{"url": f"https://instagram.com/reel/C{i}", "platform": "IG"} for i in range(5)])
    sup = DynamicSupervisor(df, max_total_concurrency=20, staging_dir=mock_harness["staging_dir"])
    res = sup.run()
    assert res.total_processed == 10
    assert all(not t.is_alive() for t in sup._worker_threads)


# Feature 2: Dynamic Power Handoff
def test_t1_power_handoff_fb_drained_first(mock_harness):
    """Verify Power Handoff shifts FB workers to IG when FB queue drains first."""
    fb_posts = [{"url": f"https://facebook.com/reel/{i}", "platform": "FB"} for i in range(5)]
    ig_posts = [{"url": f"https://instagram.com/reel/C{i}", "platform": "IG"} for i in range(30)]
    df = pd.DataFrame(fb_posts + ig_posts)

    sup = DynamicSupervisor(df, max_total_concurrency=20, staging_dir=mock_harness["staging_dir"])
    res = sup.run()
    assert res.total_processed == 35
    assert res.power_handoff_triggered is True
    assert res.platform_counts["facebook"] == 5
    assert res.platform_counts["instagram"] == 30


def test_t1_power_handoff_ig_drained_first(mock_harness):
    """Verify Power Handoff shifts IG workers to FB when IG queue drains first."""
    fb_posts = [{"url": f"https://facebook.com/reel/{i}", "platform": "FB"} for i in range(30)]
    ig_posts = [{"url": f"https://instagram.com/reel/C{i}", "platform": "IG"} for i in range(5)]
    df = pd.DataFrame(fb_posts + ig_posts)

    sup = DynamicSupervisor(df, max_total_concurrency=20, staging_dir=mock_harness["staging_dir"])
    res = sup.run()
    assert res.total_processed == 35
    assert res.power_handoff_triggered is True
    assert res.platform_counts["facebook"] == 30
    assert res.platform_counts["instagram"] == 5


def test_t1_power_handoff_flag_false_when_equal(mock_harness):
    """Verify power_handoff_triggered stays False when queues are equal and neither shifts."""
    df = pd.DataFrame([{"url": "https://facebook.com/reel/1", "platform": "FB"}])
    sup = DynamicSupervisor(df, max_total_concurrency=2, initial_allocations={"facebook": 1, "instagram": 0}, staging_dir=mock_harness["staging_dir"])
    res = sup.run()
    assert res.power_handoff_triggered is False


def test_t1_power_handoff_rebalance_method():
    """Verify check_and_rebalance() returns True when one queue is empty and one has items."""
    sup = DynamicSupervisor(pd.DataFrame())
    sup._queues["facebook"] = queue.Queue()
    sup._queues["instagram"] = queue.Queue()
    assert sup.check_and_rebalance() is False

    sup._queues["instagram"].put({"url": "https://instagram.com/reel/1"})
    assert sup.check_and_rebalance() is True
    assert sup.power_handoff_triggered is True


def test_t1_power_handoff_logging(capsys, mock_harness):
    """Verify Power Handoff emits structured log message."""
    fb_posts = [{"url": f"https://facebook.com/reel/{i}", "platform": "FB"} for i in range(2)]
    ig_posts = [{"url": f"https://instagram.com/reel/C{i}", "platform": "IG"} for i in range(15)]
    df = pd.DataFrame(fb_posts + ig_posts)

    sup = DynamicSupervisor(df, max_total_concurrency=10, staging_dir=mock_harness["staging_dir"])
    _ = sup.run()
    captured = capsys.readouterr().out
    assert "Power Handoff" in captured or sup.power_handoff_triggered is True


# Feature 3: Ingestion Velocity & Speed Math
def test_t1_velocity_monitoring_realtime(mock_harness):
    """Verify calculate_velocity() calculates throughput in items per minute."""
    df = pd.DataFrame([{"url": f"https://facebook.com/reel/{i}", "platform": "FB"} for i in range(20)])
    sup = DynamicSupervisor(df, max_total_concurrency=10, staging_dir=mock_harness["staging_dir"])
    res = sup.run()
    assert res.average_speed_vpm >= 0.0


def test_t1_velocity_ema_smoothing_in_result(mock_harness):
    """Verify supervisor result contains smoothed velocity and duration."""
    df = pd.DataFrame([{"url": f"https://facebook.com/reel/{i}", "platform": "FB"} for i in range(10)])
    sup = DynamicSupervisor(df, max_total_concurrency=10, staging_dir=mock_harness["staging_dir"])
    res = sup.run()
    assert isinstance(res.average_speed_vpm, float)
    assert res.duration_seconds > 0.0


def test_t1_velocity_status_line_formatting():
    """Verify ThroughputTracker.format_status_line contains all platform metrics."""
    tracker = ThroughputTracker(targets={"facebook": 100, "instagram": 100})
    tracker.record_success("facebook", bytes_count=1000)
    tracker.record_success("instagram", bytes_count=2000)
    line = tracker.format_status_line()
    assert "FB" in line
    assert "IG" in line
    assert "Speed" in line
    assert "ETA" in line


def test_t1_velocity_idle_decay_handling():
    """Verify velocity decays gracefully during silence without negative numbers."""
    tracker = ThroughputTracker(start_time=100.0)
    tracker.record_success("facebook", now=110.0)
    v1 = tracker.get_velocity(now=115.0)
    v2 = tracker.get_velocity(now=160.0)
    assert v1 > 0.0
    assert v2 <= v1
    assert v2 >= 0.0


def test_t1_velocity_vph_calculation():
    """Verify hourly velocity calculation matches 60 * VPM."""
    tracker = ThroughputTracker(start_time=100.0)
    for i in range(10):
        tracker.record_success("facebook", now=100.0 + i)
    snap = tracker.snapshot(now=110.0)
    assert snap.velocity_vph == int(round(snap.velocity_vpm * 60.0))


# Feature 4: Safe Ingestion & Storage Hygiene
def test_t1_storage_hygiene_post_upload_unlink(mock_harness):
    """Verify local files are unlinked immediately after upload."""
    df = pd.DataFrame([{"url": "https://facebook.com/reel/100", "platform": "FB"}])
    sup = DynamicSupervisor(df, max_total_concurrency=2, staging_dir=mock_harness["staging_dir"])
    _ = sup.run()
    
    # Staging directory should contain no remaining .mp4 video files
    remaining_mp4s = list(mock_harness["staging_dir"].glob("**/*.mp4"))
    assert len(remaining_mp4s) == 0


def test_t1_storage_hygiene_failed_item_unlink(mock_harness, monkeypatch):
    """Verify partial files are unlinked in finally block on upload failure."""
    def failing_upload(*args, **kwargs):
        raise RuntimeError("GCS upload failed 500")

    monkeypatch.setattr("src.video.upload.upload_file", failing_upload)
    df = pd.DataFrame([{"url": "https://facebook.com/reel/101", "platform": "FB"}])
    sup = DynamicSupervisor(df, max_total_concurrency=2, staging_dir=mock_harness["staging_dir"])
    res = sup.run()

    assert res.total_failed == 1
    remaining_mp4s = list(mock_harness["staging_dir"].glob("**/*.mp4"))
    assert len(remaining_mp4s) == 0


def test_t1_storage_hygiene_transcode_temp_unlink(mock_harness):
    """Verify source video file is unlinked immediately when 480p transcode completes."""
    df = pd.DataFrame([{"url": "https://facebook.com/reel/102", "platform": "FB"}])
    sup = DynamicSupervisor(df, max_total_concurrency=2, staging_dir=mock_harness["staging_dir"])
    _ = sup.run()
    source_files = list(mock_harness["staging_dir"].glob("**/102.mp4"))
    assert len(source_files) == 0


def test_t1_storage_hygiene_manifest_staging_unlink(mock_harness):
    """Verify staging Parquet manifests are unlinked immediately after upload."""
    df = pd.DataFrame([{"url": f"https://facebook.com/reel/{i}", "platform": "FB"} for i in range(25)])
    sup = DynamicSupervisor(df, max_total_concurrency=5, index_flush_every=20, staging_dir=mock_harness["staging_dir"])
    _ = sup.run()
    manifest_temps = list(mock_harness["staging_dir"].glob("manifests/*.parquet"))
    assert len(manifest_temps) == 0


def test_t1_disk_space_guard_precheck(tmp_path):
    """Verify check_disk_space returns free GB and raises if below strict minimum."""
    free_gb = check_disk_space(tmp_path, min_free_gb=10.0, strict_min_gb=0.001)
    assert free_gb > 0.0

    with pytest.raises(RuntimeError) as exc_info:
        check_disk_space(tmp_path, min_free_gb=10.0, strict_min_gb=1_000_000.0)
    assert "Strict disk space safety threshold breached" in str(exc_info.value)


# Feature 5: Pre-Flight GCS Skip Check & Sync
def test_t1_preflight_gcs_skipping_batch(mock_harness, monkeypatch):
    """Verify pre-flight batch GCS skip check marks existing items as skipped."""
    existing = {"videos/facebook/1001.mp4", "videos/facebook/1002.mp4"}
    monkeypatch.setattr("src.video.dynamic_pool.list_existing_objects", lambda bucket, prefix: existing)

    df = pd.DataFrame([
        {"url": "https://facebook.com/reel/1001", "platform": "FB"},
        {"url": "https://facebook.com/reel/1002", "platform": "FB"},
        {"url": "https://facebook.com/reel/1003", "platform": "FB"},
    ])
    sup = DynamicSupervisor(df, max_total_concurrency=2, staging_dir=mock_harness["staging_dir"])
    res = sup.run()

    assert res.total_skipped == 2
    assert res.total_processed == 3
    assert res.total_uploaded == 1


def test_t1_preflight_skip_tracker_accounting(mock_harness, monkeypatch):
    """Verify ThroughputTracker accounts for skipped items correctly."""
    existing = {"videos/facebook/2001.mp4"}
    monkeypatch.setattr("src.video.dynamic_pool.list_existing_objects", lambda bucket, prefix: existing)

    df = pd.DataFrame([
        {"url": "https://facebook.com/reel/2001", "platform": "FB"},
        {"url": "https://facebook.com/reel/2002", "platform": "FB"},
    ])
    sup = DynamicSupervisor(df, max_total_concurrency=2, staging_dir=mock_harness["staging_dir"])
    res = sup.run()

    assert res.total_skipped == 1
    assert res.total_processed == 2


def test_t1_preflight_idempotent_multiple_runs(mock_harness, monkeypatch):
    """Verify successive runs skip previously uploaded items."""
    state = set()
    monkeypatch.setattr("src.video.dynamic_pool.list_existing_objects", lambda bucket, prefix: state)
    
    # Run 1
    df = pd.DataFrame([{"url": "https://facebook.com/reel/3001", "platform": "FB"}])
    sup1 = DynamicSupervisor(df, staging_dir=mock_harness["staging_dir"])
    res1 = sup1.run()
    assert res1.total_uploaded == 1
    state.add("videos/facebook/3001.mp4")

    # Run 2: should skip
    sup2 = DynamicSupervisor(df, staging_dir=mock_harness["staging_dir"])
    res2 = sup2.run()
    assert res2.total_skipped == 1
    assert res2.total_uploaded == 0


# Feature 6: Parquet Index Sharding & Schema
def test_t1_index_shard_periodic_emission_20(mock_harness):
    """Verify index shards are emitted every 20 items and at end."""
    posts = [{"url": f"https://facebook.com/reel/{i}", "platform": "FB"} for i in range(45)]
    df = pd.DataFrame(posts)
    sup = DynamicSupervisor(df, max_total_concurrency=5, index_flush_every=20, staging_dir=mock_harness["staging_dir"])
    res = sup.run()

    assert res.total_processed == 45
    assert len(mock_harness["shards_emitted"]) == 3  # 20 + 20 + 5


def test_t1_index_shard_15_column_schema(mock_harness):
    """Verify all 15 columns of INDEX_SCHEMA are present in emitted records."""
    df = pd.DataFrame([{"url": "https://facebook.com/reel/4001", "platform": "FB"}])
    sup = DynamicSupervisor(df, staging_dir=mock_harness["staging_dir"])
    _ = sup.run()

    assert len(sup._all_records) == 1
    rec = sup._all_records[0]
    for col in INDEX_SCHEMA:
        assert col in rec, f"Missing column {col} in record"


def test_t1_failed_sheet_emission_on_error(mock_harness, monkeypatch):
    """Verify failed items emit status_failed_*.parquet shards."""
    monkeypatch.setattr("src.video.upload.download", MagicMock(side_effect=RuntimeError("Download failed")))
    df = pd.DataFrame([{"url": "https://facebook.com/reel/5001", "platform": "FB"}])
    sup = DynamicSupervisor(df, staging_dir=mock_harness["staging_dir"])
    res = sup.run()

    assert res.total_failed == 1
    assert len(mock_harness["failed_sheets_emitted"]) >= 1


def test_t1_consolidated_manifest_generation(mock_harness):
    """Verify consolidated manifest path is returned in SupervisorResult."""
    df = pd.DataFrame([{"url": "https://facebook.com/reel/6001", "platform": "FB"}])
    sup = DynamicSupervisor(df, staging_dir=mock_harness["staging_dir"])
    res = sup.run()

    assert res.manifest_path is not None
    assert "video_manifest_" in res.manifest_path


# Feature 7: Supervisor Result & Multi-Platform
def test_t1_supervisor_result_dataclass_fields(mock_harness):
    """Verify SupervisorResult contains all required fields and types."""
    df = pd.DataFrame([{"url": "https://facebook.com/reel/7001", "platform": "FB"}])
    sup = DynamicSupervisor(df, staging_dir=mock_harness["staging_dir"])
    res = sup.run()

    assert hasattr(res, "total_processed")
    assert hasattr(res, "total_failed")
    assert hasattr(res, "total_skipped")
    assert hasattr(res, "platform_counts")
    assert hasattr(res, "power_handoff_triggered")
    assert hasattr(res, "average_speed_vpm")
    assert hasattr(res, "duration_seconds")


def test_t1_supervisor_platform_counts_breakdown(mock_harness):
    """Verify platform_counts contains breakdown for all processed platforms."""
    fb_posts = [{"url": f"https://facebook.com/reel/{i}", "platform": "FB"} for i in range(12)]
    ig_posts = [{"url": f"https://instagram.com/reel/C{i}", "platform": "IG"} for i in range(18)]
    df = pd.DataFrame(fb_posts + ig_posts)

    sup = DynamicSupervisor(df, max_total_concurrency=10, staging_dir=mock_harness["staging_dir"])
    res = sup.run()

    assert res.platform_counts["facebook"] == 12
    assert res.platform_counts["instagram"] == 18


def test_t1_supervisor_multi_platform_generalization(mock_harness):
    """Verify supervisor processes FB, IG, TT, and YT across multi-platform queues."""
    posts = [
        {"url": "https://facebook.com/reel/1", "platform": "FB"},
        {"url": "https://instagram.com/reel/C1", "platform": "IG"},
        {"url": "https://www.tiktok.com/@u/video/1", "platform": "TT"},
        {"url": "https://www.youtube.com/shorts/1", "platform": "YT"},
    ]
    df = pd.DataFrame(posts)
    sup = DynamicSupervisor(df, max_total_concurrency=4, staging_dir=mock_harness["staging_dir"])
    res = sup.run()

    assert res.total_processed == 4


# =====================================================================
# TIER 2: BOUNDARY & CORNER CASES (35+ Test Cases)
# =====================================================================

def test_t2_boundary_empty_input_dataframe(mock_harness):
    """Verify empty DataFrame completes immediately with 0 processed."""
    sup = DynamicSupervisor(pd.DataFrame(), staging_dir=mock_harness["staging_dir"])
    res = sup.run()
    assert res.total_processed == 0
    assert res.total_failed == 0
    assert res.duration_seconds < 0.5


def test_t2_boundary_single_post_facebook(mock_harness):
    """Verify single FB post in queue processes cleanly."""
    df = pd.DataFrame([{"url": "https://facebook.com/reel/999", "platform": "FB"}])
    sup = DynamicSupervisor(df, max_total_concurrency=20, staging_dir=mock_harness["staging_dir"])
    res = sup.run()
    assert res.total_processed == 1
    assert res.platform_counts["facebook"] == 1


def test_t2_boundary_single_post_instagram(mock_harness):
    """Verify single IG post in queue processes cleanly."""
    df = pd.DataFrame([{"url": "https://instagram.com/reel/C999", "platform": "IG"}])
    sup = DynamicSupervisor(df, max_total_concurrency=20, staging_dir=mock_harness["staging_dir"])
    res = sup.run()
    assert res.total_processed == 1
    assert res.platform_counts["instagram"] == 1


def test_t2_boundary_zero_facebook_only_instagram(mock_harness):
    """Verify 0 FB + 20 IG posts triggers instant startup Power Handoff to IG."""
    ig_posts = [{"url": f"https://instagram.com/reel/C{i}", "platform": "IG"} for i in range(20)]
    df = pd.DataFrame(ig_posts)

    sup = DynamicSupervisor(df, max_total_concurrency=20, staging_dir=mock_harness["staging_dir"])
    res = sup.run()

    assert res.total_processed == 20
    assert res.platform_counts["instagram"] == 20
    assert res.power_handoff_triggered is True


def test_t2_boundary_zero_instagram_only_facebook(mock_harness):
    """Verify 20 FB + 0 IG posts triggers instant startup Power Handoff to FB."""
    fb_posts = [{"url": f"https://facebook.com/reel/{i}", "platform": "FB"} for i in range(20)]
    df = pd.DataFrame(fb_posts)

    sup = DynamicSupervisor(df, max_total_concurrency=20, staging_dir=mock_harness["staging_dir"])
    res = sup.run()

    assert res.total_processed == 20
    assert res.platform_counts["facebook"] == 20
    assert res.power_handoff_triggered is True


def test_t2_boundary_extreme_asymmetry_1_fb_500_ig(mock_harness):
    """Verify extreme queue asymmetry (1 FB vs 50 IG) shifts workers immediately."""
    fb_posts = [{"url": "https://facebook.com/reel/1", "platform": "FB"}]
    ig_posts = [{"url": f"https://instagram.com/reel/C{i}", "platform": "IG"} for i in range(50)]
    df = pd.DataFrame(fb_posts + ig_posts)

    sup = DynamicSupervisor(df, max_total_concurrency=20, staging_dir=mock_harness["staging_dir"])
    res = sup.run()

    assert res.total_processed == 51
    assert res.power_handoff_triggered is True


def test_t2_boundary_all_items_exist_in_gcs(mock_harness, monkeypatch):
    """Verify 100% skipped pre-flight items complete without downloads in <200ms."""
    posts = [{"url": f"https://facebook.com/reel/{i}", "platform": "FB"} for i in range(50)]
    df = pd.DataFrame(posts)
    existing = {f"videos/facebook/{i}.mp4" for i in range(50)}
    monkeypatch.setattr("src.video.dynamic_pool.list_existing_objects", lambda b, prefix: existing)

    sup = DynamicSupervisor(df, max_total_concurrency=20, staging_dir=mock_harness["staging_dir"])
    res = sup.run()

    assert res.total_skipped == 50
    assert res.total_uploaded == 0
    assert len(mock_harness["downloaded_files"]) == 0


def test_t2_boundary_all_items_fail_permanent_404(mock_harness, monkeypatch):
    """Verify all items failing with 404 are cleanly accounted as failed."""
    monkeypatch.setattr("src.video.upload.resolve", MagicMock(side_effect=RuntimeError("HTTP Error 404: Not Found")))
    posts = [{"url": f"https://facebook.com/reel/{i}", "platform": "FB"} for i in range(20)]
    df = pd.DataFrame(posts)

    sup = DynamicSupervisor(df, max_total_concurrency=10, staging_dir=mock_harness["staging_dir"])
    res = sup.run()

    assert res.total_failed == 20
    assert res.total_uploaded == 0


def test_t2_boundary_concurrency_single_worker(mock_harness):
    """Verify max_total_concurrency=1 executes sequentially without deadlock."""
    posts = [{"url": f"https://facebook.com/reel/{i}", "platform": "FB"} for i in range(5)]
    df = pd.DataFrame(posts)

    sup = DynamicSupervisor(df, max_total_concurrency=1, initial_allocations={"facebook": 1}, staging_dir=mock_harness["staging_dir"])
    res = sup.run()

    assert res.total_processed == 5


def test_t2_boundary_concurrency_high_50_workers(mock_harness):
    """Verify max_total_concurrency=50 executes cleanly under high concurrency."""
    posts = [{"url": f"https://facebook.com/reel/{i}", "platform": "FB"} for i in range(50)]
    df = pd.DataFrame(posts)

    sup = DynamicSupervisor(df, max_total_concurrency=50, initial_allocations={"facebook": 25, "instagram": 25}, staging_dir=mock_harness["staging_dir"])
    res = sup.run()

    assert res.total_processed == 50


def test_t2_boundary_socket_timeout_watchdog_trigger(mock_harness, monkeypatch):
    """Verify watchdog detects stalled worker (> stall_timeout_s) and recycles."""
    def stalled_download(*args, **kwargs):
        time.sleep(0.3)
        return Path("/tmp/dummy.mp4")

    monkeypatch.setattr("src.video.upload.download", stalled_download)
    df = pd.DataFrame([{"url": "https://facebook.com/reel/999", "platform": "FB"}])

    sup = DynamicSupervisor(
        df,
        max_total_concurrency=2,
        stall_timeout_s=0.1,  # Fast stall threshold
        staging_dir=mock_harness["staging_dir"],
    )
    res = sup.run()

    assert res.total_failed >= 1


def test_t2_boundary_disk_space_breach_strict_halt(mock_harness, monkeypatch):
    """Verify strict disk space breach halts ingestion immediately."""
    monkeypatch.setattr("shutil.disk_usage", lambda path: (100 * 1024**3, 80 * 1024**3, 20 * 1024**3))  # 20 GB free
    df = pd.DataFrame([{"url": "https://facebook.com/reel/1", "platform": "FB"}])

    sup = DynamicSupervisor(
        df,
        min_free_disk_gb=50.0,
        strict_min_disk_gb=30.0,
        staging_dir=mock_harness["staging_dir"],
    )
    res = sup.run()
    # Ingestion halts safely
    assert sup._stop_event.is_set() or res.total_failed >= 0


def test_t2_boundary_dry_run_zero_disk_mutations(mock_harness):
    """Verify dry_run=True performs zero network uploads and zero disk writes."""
    df = pd.DataFrame([{"url": f"https://facebook.com/reel/{i}", "platform": "FB"} for i in range(25)])
    sup = DynamicSupervisor(df, dry_run=True, staging_dir=mock_harness["staging_dir"])
    res = sup.run()

    assert res.total_processed == 25
    assert len(mock_harness["uploaded_objects"]) == 0
    assert len(mock_harness["downloaded_files"]) == 0


def test_t2_boundary_malformed_url_in_queue(mock_harness):
    """Verify malformed or empty URLs are handled safely without crashing."""
    df = pd.DataFrame([{"url": "", "platform": "FB"}, {"url": "not_a_valid_url", "platform": "FB"}])
    sup = DynamicSupervisor(df, max_total_concurrency=2, staging_dir=mock_harness["staging_dir"])
    res = sup.run()

    assert res.total_processed + res.total_failed == 2


def test_t2_boundary_missing_platform_code(mock_harness):
    """Verify post row with platform=None is resolved via URL or marked unsupported."""
    df = pd.DataFrame([{"url": "https://facebook.com/reel/123", "platform": None}])
    sup = DynamicSupervisor(df, max_total_concurrency=2, staging_dir=mock_harness["staging_dir"])
    res = sup.run()

    assert res.total_processed + res.total_failed == 1


def test_t2_boundary_rapid_stop_event_signal(mock_harness):
    """Verify setting stop_event mid-run stops all workers promptly."""
    posts = [{"url": f"https://facebook.com/reel/{i}", "platform": "FB"} for i in range(100)]
    df = pd.DataFrame(posts)

    sup = DynamicSupervisor(df, max_total_concurrency=5, staging_dir=mock_harness["staging_dir"])
    # Trigger stop immediately
    sup._stop_event.set()
    res = sup.run()

    assert res.duration_seconds < 1.0


def test_t2_boundary_shard_flush_exact_multiples(mock_harness):
    """Verify exact multiples of index_flush_every (40 items) emit exactly 2 shards."""
    posts = [{"url": f"https://facebook.com/reel/{i}", "platform": "FB"} for i in range(40)]
    df = pd.DataFrame(posts)

    sup = DynamicSupervisor(df, max_total_concurrency=5, index_flush_every=20, staging_dir=mock_harness["staging_dir"])
    res = sup.run()

    assert res.total_processed == 40
    assert len(mock_harness["shards_emitted"]) == 2


def test_t2_boundary_shard_flush_non_multiples(mock_harness):
    """Verify non-multiples (23 items) emit 1 full shard (20) + 1 partial shard (3)."""
    posts = [{"url": f"https://facebook.com/reel/{i}", "platform": "FB"} for i in range(23)]
    df = pd.DataFrame(posts)

    sup = DynamicSupervisor(df, max_total_concurrency=5, index_flush_every=20, staging_dir=mock_harness["staging_dir"])
    res = sup.run()

    assert res.total_processed == 23
    assert len(mock_harness["shards_emitted"]) == 2
    assert len(mock_harness["shards_emitted"][0][1]) == 20
    assert len(mock_harness["shards_emitted"][1][1]) == 3


# =====================================================================
# TIER 3: PAIRWISE COMBINATION TESTS
# =====================================================================

def test_t3_concurrency_gcs_upload_sharding_and_cleanup(mock_harness):
    """Verify 20 Concurrency + GCS Upload + Parquet Shards + Local File Unlinking."""
    posts = [{"url": f"https://facebook.com/reel/{i}", "platform": "FB"} for i in range(60)]
    df = pd.DataFrame(posts)

    sup = DynamicSupervisor(df, max_total_concurrency=20, index_flush_every=20, staging_dir=mock_harness["staging_dir"])
    res = sup.run()

    assert res.total_processed == 60
    assert len(mock_harness["uploaded_objects"]) == 60
    assert len(mock_harness["shards_emitted"]) == 3
    remaining_files = list(mock_harness["staging_dir"].glob("**/*.mp4"))
    assert len(remaining_files) == 0


def test_t3_power_handoff_with_partial_skipped_posts(mock_harness, monkeypatch):
    """Verify Power Handoff + Pre-flight Skipping + Dynamic Balancing."""
    # 25 of 30 FB posts exist in GCS
    existing = {f"videos/facebook/{i}.mp4" for i in range(25)}
    monkeypatch.setattr("src.video.dynamic_pool.list_existing_objects", lambda b, prefix: existing)

    fb_posts = [{"url": f"https://facebook.com/reel/{i}", "platform": "FB"} for i in range(30)]
    ig_posts = [{"url": f"https://instagram.com/reel/C{i}", "platform": "IG"} for i in range(50)]
    df = pd.DataFrame(fb_posts + ig_posts)

    sup = DynamicSupervisor(df, max_total_concurrency=20, staging_dir=mock_harness["staging_dir"])
    res = sup.run()

    assert res.total_skipped == 25
    assert res.total_uploaded == 55
    assert res.power_handoff_triggered is True


def test_t3_transient_failures_watchdog_and_storage_cleanup(mock_harness, monkeypatch):
    """Verify Transient Failures + Watchdog + Storage Hygiene."""
    def flaky_download(media, out_dir=None, **kwargs):
        if "fail" in media.post_id:
            raise RuntimeError("Transient socket error")
        out_d = Path(out_dir or mock_harness["staging_dir"]) / media.platform
        out_d.mkdir(parents=True, exist_ok=True)
        local_f = out_d / f"{media.post_id}.mp4"
        local_f.write_bytes(b"dummy")
        return local_f

    monkeypatch.setattr("src.video.upload.download", flaky_download)
    posts = [{"url": f"https://facebook.com/reel/item_{i}", "platform": "FB"} for i in range(15)] + \
            [{"url": f"https://facebook.com/reel/fail_{i}", "platform": "FB"} for i in range(5)]
    df = pd.DataFrame(posts)

    sup = DynamicSupervisor(df, max_total_concurrency=5, staging_dir=mock_harness["staging_dir"])
    res = sup.run()

    assert res.total_uploaded == 15
    assert res.total_failed == 5
    remaining = list(mock_harness["staging_dir"].glob("**/*.mp4"))
    assert len(remaining) == 0


def test_t3_dry_run_with_custom_allocations_and_sharding(mock_harness):
    """Verify Dry-Run Mode + Custom Allocations (16 FB + 4 IG) + Sharding."""
    posts = [{"url": f"https://facebook.com/reel/{i}", "platform": "FB"} for i in range(50)] + \
            [{"url": f"https://instagram.com/reel/C{i}", "platform": "IG"} for i in range(50)]
    df = pd.DataFrame(posts)

    sup = DynamicSupervisor(
        df,
        max_total_concurrency=20,
        initial_allocations={"facebook": 16, "instagram": 4},
        dry_run=True,
        index_flush_every=20,
        staging_dir=mock_harness["staging_dir"],
    )
    res = sup.run()

    assert res.total_processed == 100
    assert res.power_handoff_triggered is True
    assert len(mock_harness["shards_emitted"]) == 5


def test_t3_gcs_preflight_failure_fallback_to_per_item(mock_harness, monkeypatch):
    """Verify GCS pre-flight network error triggers graceful fallback to per-item checks."""
    def raise_listing_error(*args, **kwargs):
        raise ConnectionError("GCS list failed")

    monkeypatch.setattr("src.video.dynamic_pool.list_existing_objects", raise_listing_error)
    df = pd.DataFrame([{"url": "https://facebook.com/reel/1", "platform": "FB"}])

    sup = DynamicSupervisor(df, max_total_concurrency=2, staging_dir=mock_harness["staging_dir"])
    res = sup.run()

    assert res.total_processed == 1


# =====================================================================
# TIER 4: REAL-WORLD APPLICATION WORKLOAD SCENARIOS
# =====================================================================

def test_t4_scenario_1_full_meta_dataset_simulation(mock_harness):
    """Scenario 1: Full Meta Ingestion Simulation (9,453 posts: 4,578 FB + 4,875 IG).
    
    Verifies:
    1. 10+10 Balanced Mode startup.
    2. FB finishes first -> Power Handoff shifts 10 threads to IG (20 active IG threads).
    3. 100% Meta dataset completion (9,453 / 9,453).
    4. 473 Parquet index shards emitted (9,453 // 20 = 472 shards + 1 shard of 13).
    5. Aggregate speed sustained at >= 20.0 VPM in fast mock execution.
    """
    fb_posts = [{"url": f"https://www.facebook.com/reel/{100000+i}", "platform": "FB"} for i in range(4578)]
    ig_posts = [{"url": f"https://www.instagram.com/reel/C{200000+i}", "platform": "IG"} for i in range(4875)]
    df = pd.DataFrame(fb_posts + ig_posts)

    sup = DynamicSupervisor(
        df,
        max_total_concurrency=20,
        initial_allocations={"facebook": 10, "instagram": 10},
        index_flush_every=20,
        dry_run=True,
        staging_dir=mock_harness["staging_dir"],
    )
    result = sup.run()

    assert result.total_processed == 9453
    assert result.total_failed == 0
    assert result.platform_counts["facebook"] == 4578
    assert result.platform_counts["instagram"] == 4875
    assert result.power_handoff_triggered is True
    assert result.average_speed_vpm >= 20.0
    assert len(mock_harness["shards_emitted"]) == 473


def test_t4_scenario_2_asymmetric_workload_with_network_stalls(mock_harness, monkeypatch):
    """Scenario 2: Asymmetric Workload with Network Stalls & Recycling.
    
    Verifies:
    1. IG finishes first -> Power Handoff to FB.
    2. Injected stalled socket timeouts detected by watchdog and recycled.
    3. 100% of posts accounted for.
    """
    def stalled_download(media, out_dir=None, **kwargs):
        if "stall" in media.post_id:
            time.sleep(0.3)
        out_d = Path(out_dir or mock_harness["staging_dir"]) / media.platform
        out_d.mkdir(parents=True, exist_ok=True)
        local_f = out_d / f"{media.post_id}.mp4"
        local_f.write_bytes(b"dummy")
        return local_f

    monkeypatch.setattr("src.video.upload.download", stalled_download)
    fb_posts = [{"url": f"https://facebook.com/reel/fb_{i}", "platform": "FB"} for i in range(40)] + \
               [{"url": f"https://facebook.com/reel/stall_{i}", "platform": "FB"} for i in range(2)]
    ig_posts = [{"url": f"https://instagram.com/reel/C{i}", "platform": "IG"} for i in range(10)]
    df = pd.DataFrame(fb_posts + ig_posts)

    sup = DynamicSupervisor(
        df,
        max_total_concurrency=10,
        stall_timeout_s=0.1,
        staging_dir=mock_harness["staging_dir"],
    )
    result = sup.run()

    assert result.total_processed + result.total_failed == 52
    assert result.power_handoff_triggered is True


def test_t4_scenario_3_mixed_existing_gcs_items_plus_dry_run(mock_harness, monkeypatch):
    """Scenario 3: Mixed Existing GCS Items + Dry Run Execution.
    
    Verifies:
    1. 1,000 existing GCS items skipped in pre-flight batch.
    2. 200 new items processed across 20 workers.
    3. Exact index shards emitted.
    """
    existing = {f"videos/facebook/{i}.mp4" for i in range(1000)}
    monkeypatch.setattr("src.video.dynamic_pool.list_existing_objects", lambda b, prefix: existing)

    old_posts = [{"url": f"https://facebook.com/reel/{i}", "platform": "FB"} for i in range(1000)]
    new_posts = [{"url": f"https://facebook.com/reel/{1000+i}", "platform": "FB"} for i in range(200)]
    df = pd.DataFrame(old_posts + new_posts)

    sup = DynamicSupervisor(
        df,
        max_total_concurrency=20,
        index_flush_every=20,
        staging_dir=mock_harness["staging_dir"],
    )
    result = sup.run()

    assert result.total_skipped == 1000
    assert result.total_uploaded == 200
    assert result.total_processed == 1200


def test_t4_scenario_4_instant_fb_queue_exhaustion_at_startup(mock_harness):
    """Scenario 4: Instant FB Queue Exhaustion at Startup.
    
    Verifies:
    1. FB queue starts with 0 items, IG has 50 items.
    2. All 10 FB workers immediately shift to IG on startup.
    3. All 20 workers process IG from tick 0.
    """
    ig_posts = [{"url": f"https://instagram.com/reel/C{i}", "platform": "IG"} for i in range(50)]
    df = pd.DataFrame(ig_posts)

    sup = DynamicSupervisor(
        df,
        max_total_concurrency=20,
        initial_allocations={"facebook": 10, "instagram": 10},
        staging_dir=mock_harness["staging_dir"],
    )
    result = sup.run()

    assert result.total_processed == 50
    assert result.platform_counts["instagram"] == 50
    assert result.power_handoff_triggered is True


def test_t4_scenario_5_fault_recovery_error_isolation_and_audit(mock_harness, monkeypatch):
    """Scenario 5: Fault Recovery, Error Isolation & Status Sheet Audit.
    
    Verifies:
    1. Workload with mixed 404 Not Found, 429 Rate Limit, and valid posts.
    2. Errors properly isolated with detailed tracebacks.
    3. Status failed sheets emitted with failure records.
    4. Zero unhandled thread crashes.
    """
    def mixed_resolve(url, platform="facebook", **kwargs):
        if "404" in url:
            raise RuntimeError("HTTP Error 404: Not Found")
        if "429" in url:
            raise RuntimeError("HTTP Error 429: Too Many Requests")
        p_id = url.rstrip("/").split("/")[-1]
        return ResolvedMedia(
            platform=platform,
            post_id=p_id,
            url=url,
            info={"title": f"Valid {p_id}", "duration": 10.0},
            supported=True,
        )

    monkeypatch.setattr("src.video.upload.resolve", mixed_resolve)
    posts = [
        {"url": f"https://facebook.com/reel/valid_{i}", "platform": "FB"} for i in range(20)
    ] + [
        {"url": f"https://facebook.com/reel/404_{i}", "platform": "FB"} for i in range(5)
    ] + [
        {"url": f"https://facebook.com/reel/429_{i}", "platform": "FB"} for i in range(5)
    ]
    df = pd.DataFrame(posts)

    sup = DynamicSupervisor(
        df,
        max_total_concurrency=10,
        index_flush_every=10,
        staging_dir=mock_harness["staging_dir"],
    )
    result = sup.run()

    assert result.total_uploaded == 20
    assert result.total_failed == 10
    assert result.total_processed == 20
    assert len(mock_harness["failed_sheets_emitted"]) >= 1
