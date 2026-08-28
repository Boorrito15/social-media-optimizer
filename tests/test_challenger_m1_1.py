"""Milestone 1 Challenger 1 Empirical Stress Test & Adversarial Verification Suite.

Verification Scope:
1. High-concurrency hammer test on ThroughputTracker (50 threads firing 1,000 events each = 50,000 events)
   with simultaneous reader threads querying snapshot(), get_velocity(), get_eta_seconds(), format_status_line().
2. Non-monotonic time steps, extreme time jumps (forward & backward), rapid burst-decay cycles, zero duration,
   and numerical stability (no NaN, Inf, negative ETA, or math crashes).
3. Malformed, adversarial, and edge-case URLs across Facebook, Instagram, TikTok, and YouTube.
4. Error classification accuracy across diverse exception strings, nested tracebacks, and real yt-dlp error outputs.
5. build_ydl_opts option matrix and YoutubeDL instance configuration validation.
"""

from __future__ import annotations

import math
import re
import sys
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest
import yt_dlp

from src.video.dynamic_pool import (
    PlatformStats,
    SupervisorResult,
    ThroughputTracker,
    TrackerSnapshot,
)
from src.video.extractors import (
    EXTRACTORS,
    BaseExtractor,
    FacebookExtractor,
    InstagramExtractor,
    MediaResolutionError,
    PermanentExtractionError,
    ResolvedMedia,
    TikTokExtractor,
    TransientExtractionError,
    YouTubeExtractor,
    build_ydl_opts,
    classify_error,
    is_permanent_error,
    is_transient_error,
)
from utils.ffmpeg import TranscodeResult, transcode_to_480p


# =====================================================================
# 1. HIGH CONCURRENCY STRESS TEST (50 threads x 1,000 events = 50,000)
# =====================================================================

def test_throughput_tracker_high_concurrency_stress():
    """Stress test ThroughputTracker with 50 concurrent writer threads and 10 concurrent reader threads.

    Verifies:
    - Zero deadlocks under heavy contention.
    - Exact arithmetic consistency across 50,000 events.
    - Thread-safe snapshot generation without torn reads.
    - All snapshot fields maintain mathematical invariants.
    """
    num_writers = 50
    events_per_writer = 1000
    total_events = num_writers * events_per_writer  # 50,000

    targets = {
        "facebook": 25000,
        "instagram": 20000,
        "tiktok": 5000,
    }
    tracker = ThroughputTracker(targets=targets, start_time=1000.0)

    stop_readers = threading.Event()
    reader_exceptions = []
    read_snapshots = []

    def reader_worker(rid: int):
        while not stop_readers.is_set():
            try:
                snap = tracker.snapshot()
                read_snapshots.append(snap)
                _ = tracker.get_velocity()
                _ = tracker.get_session_velocity()
                _ = tracker.get_eta_seconds()
                _ = tracker.format_speed()
                _ = tracker.format_eta()
                _ = tracker.format_status_line()
                _ = tracker.get_platform_counts()
                time.sleep(0.001)
            except Exception as e:
                reader_exceptions.append(e)

    def writer_worker(tid: int):
        platforms = ["facebook", "instagram", "tiktok"]
        p = platforms[tid % len(platforms)]
        for i in range(events_per_writer):
            # Mix statuses: 70% uploaded, 20% skipped, 10% failed
            rem = i % 10
            now_t = 1000.0 + (i * 0.1) + (tid * 0.001)
            if rem < 7:
                tracker.record_success(p, bytes_count=100, now=now_t)
            elif rem < 9:
                tracker.record_skip(p, now=now_t)
            else:
                tracker.record_failure(p, error="Simulated failure", now=now_t)

    # Launch readers
    num_readers = 10
    readers = [threading.Thread(target=reader_worker, args=(i,)) for i in range(num_readers)]
    for r in readers:
        r.start()

    # Launch writers
    writers = [threading.Thread(target=writer_worker, args=(i,)) for i in range(num_writers)]
    for w in writers:
        w.start()

    # Wait for writers
    for w in writers:
        w.join(timeout=30.0)
        assert not w.is_alive(), "Writer thread deadlocked!"

    # Stop readers
    stop_readers.set()
    for r in readers:
        r.join(timeout=10.0)
        assert not r.is_alive(), "Reader thread deadlocked!"

    assert len(reader_exceptions) == 0, f"Reader encountered exceptions: {reader_exceptions}"

    # Verify final snapshot integrity
    snap = tracker.snapshot(now=2000.0)
    assert snap.total_attempted == total_events
    
    # Calculate expected counts:
    # Per writer: 700 uploaded, 200 skipped, 100 failed
    expected_uploaded = num_writers * 700
    expected_skipped = num_writers * 200
    expected_failed = num_writers * 100
    expected_processed = expected_uploaded + expected_skipped

    assert snap.total_uploaded == expected_uploaded
    assert snap.total_skipped == expected_skipped
    assert snap.total_failed == expected_failed
    assert snap.total_processed == expected_processed
    assert snap.total_bytes == expected_uploaded * 100
    assert snap.total_attempted == snap.total_processed + snap.total_failed

    # Platform breakdown consistency
    plat_uploaded_sum = sum(p.uploaded for p in snap.platform_stats.values())
    plat_skipped_sum = sum(p.skipped for p in snap.platform_stats.values())
    plat_failed_sum = sum(p.failed for p in snap.platform_stats.values())
    plat_bytes_sum = sum(p.total_bytes for p in snap.platform_stats.values())

    assert plat_uploaded_sum == expected_uploaded
    assert plat_skipped_sum == expected_skipped
    assert plat_failed_sum == expected_failed
    assert plat_bytes_sum == expected_uploaded * 100

    # Ensure all intermediate snapshots read during concurrency were structurally valid
    for s in read_snapshots:
        assert not math.isnan(s.velocity_vpm)
        assert not math.isnan(s.session_velocity_vpm)
        assert not math.isinf(s.velocity_vpm)
        assert s.progress_pct >= 0.0
        assert s.total_processed <= total_events


# =====================================================================
# 2. TIME STEP DISTORTION & NUMERICAL STABILITY HARNESS
# =====================================================================

def test_tracker_non_monotonic_backward_time():
    """Verify ThroughputTracker handles out-of-order and backward timestamps gracefully."""
    tracker = ThroughputTracker(start_time=100.0)

    # Record out-of-order events
    timestamps = [150.0, 120.0, 180.0, 110.0, 200.0, 90.0, 105.0]
    for i, t in enumerate(timestamps):
        tracker.record_success("facebook", bytes_count=1000, now=t)

    snap = tracker.snapshot(now=200.0)
    assert snap.total_uploaded == len(timestamps)
    assert snap.total_processed == len(timestamps)
    assert not math.isnan(snap.velocity_vpm)
    assert not math.isinf(snap.velocity_vpm)
    assert snap.velocity_vpm >= 0.0


def test_tracker_extreme_forward_time_jump():
    """Verify ThroughputTracker handles massive time jumps without overflow or domain errors."""
    tracker = ThroughputTracker(start_time=0.0)
    tracker.record_success("facebook", bytes_count=5000, now=10.0)
    tracker.record_success("facebook", bytes_count=5000, now=20.0)

    v_active = tracker.get_velocity(now=25.0)
    assert v_active > 0.0

    # Huge jump: 10 years into the future (315,360,000 seconds)
    jump_t = 315_360_000.0
    v_jump = tracker.get_velocity(now=jump_t)
    assert v_jump == 0.0

    snap = tracker.snapshot(now=jump_t)
    assert snap.velocity_vpm == 0.0
    assert snap.velocity_vph == 0
    assert not math.isnan(snap.session_velocity_vpm)
    assert snap.session_velocity_vpm >= 0.0
    assert tracker.format_speed(now=jump_t) == "Speed 0.0 /min"


def test_tracker_zero_elapsed_time_and_microsecond_bursts():
    """Verify ThroughputTracker does not divide by zero when start_time == now or dt is tiny."""
    tracker = ThroughputTracker(start_time=100.0)

    # Zero elapsed time query
    assert tracker.get_session_velocity(now=100.0) == 0.0
    assert tracker.get_velocity(now=100.0) == 0.0
    assert tracker.get_eta_seconds(now=100.0) is None

    # Query earlier than start_time (elapsed <= 0, protected by max(1.0, elapsed))
    assert tracker.get_session_velocity(now=50.0) == 0.0
    assert tracker.get_velocity(now=50.0) == 0.0

    # 1,000 events in same millisecond
    for _ in range(1000):
        tracker.record_success("facebook", bytes_count=10, now=100.0001)

    snap = tracker.snapshot(now=100.0001)
    assert snap.total_uploaded == 1000
    assert not math.isnan(snap.velocity_vpm)
    assert not math.isinf(snap.velocity_vpm)
    assert snap.velocity_vpm > 0.0


def test_tracker_rapid_burst_decay_oscillation():
    """Verify alternating burst-decay cycles do not cause numerical instability."""
    tracker = ThroughputTracker(start_time=0.0)
    t = 0.0

    for cycle in range(20):
        # Burst: 10 items in 1 second
        for i in range(10):
            t += 0.1
            tracker.record_success("facebook", now=t)
        v_burst = tracker.get_velocity(now=t, ema=True)
        assert v_burst > 0.0

        # Decay: 60 seconds silence
        t += 60.0
        v_decay = tracker.get_velocity(now=t, ema=True)
        assert v_decay < v_burst

    snap = tracker.snapshot(now=t)
    assert snap.total_uploaded == 200
    assert not math.isnan(snap.velocity_vpm)


def test_tracker_zero_targets_and_empty_configurations():
    """Verify ThroughputTracker behavior with 0 targets or custom targets."""
    # Zero targets
    tracker_zero = ThroughputTracker(targets={"facebook": 0, "instagram": 0}, start_time=0.0)
    snap = tracker_zero.snapshot(now=10.0)
    assert snap.progress_pct == 0.0
    assert snap.remaining == 0
    assert tracker_zero.get_eta_seconds(now=10.0) == 0.0
    assert tracker_zero.format_eta(now=10.0) == "Complete"

    # Empty dictionary targets
    tracker_empty = ThroughputTracker(targets={}, start_time=0.0)
    tracker_empty.record_success("unknown_plat", now=5.0)
    snap_empty = tracker_empty.snapshot(now=10.0)
    assert snap_empty.total_processed == 1
    assert "unknown_plat" in snap_empty.platform_stats


# =====================================================================
# 3. MALFORMED & ADVERSARIAL URL PARSING ACROSS PLATFORMS
# =====================================================================

def test_facebook_extractor_adversarial_urls():
    """Stress test FacebookExtractor.extract_id with malformed, complex, and adversarial URLs."""
    fb = FacebookExtractor()

    # Standard and edge cases
    assert fb.extract_id("https://www.facebook.com/watch/?v=1000777895026531") == "1000777895026531"
    assert fb.extract_id("https://facebook.com/watch/?extra=1&v=998877665544&foo=bar") == "998877665544"
    assert fb.extract_id("https://m.facebook.com/watch/?v=12345#section") == "12345"
    assert fb.extract_id("https://fb.watch/abcd1234_XYZ/?mibextid=wwXIfr") == "abcd1234_XYZ"
    assert fb.extract_id("https://www.facebook.com/reel/1000777895026531/?s=fb_reels") == "1000777895026531"
    assert fb.extract_id("https://www.facebook.com/reels/8877665544332211/") == "8877665544332211"
    assert fb.extract_id("https://www.facebook.com/share/r/ReelToken123/") == "ReelToken123"
    assert fb.extract_id("https://www.facebook.com/share/v/VideoToken456/") == "VideoToken456"
    assert fb.extract_id("https://www.facebook.com/videos/1234567890/") == "1234567890"

    # Malformed / Fallback resilience: must not crash
    assert fb.extract_id("https://www.facebook.com/user/posts/11223344") == "11223344"
    assert fb.extract_id("https://facebook.com/page/") == "page"
    assert isinstance(fb.extract_id("https://facebook.com/"), str)
    assert isinstance(fb.extract_id(""), str)
    assert isinstance(fb.extract_id("invalid_plain_text"), str)


def test_instagram_extractor_adversarial_urls():
    """Stress test InstagramExtractor.extract_id with malformed, complex, and adversarial URLs."""
    ig = InstagramExtractor()

    assert ig.extract_id("https://www.instagram.com/reel/C5de2n2Pnh6/?igsh=MWQ=") == "C5de2n2Pnh6"
    assert ig.extract_id("https://instagram.com/reels/C5de2n2Pnh6/") == "C5de2n2Pnh6"
    assert ig.extract_id("https://www.instagram.com/p/CsO_xntBC_T?utm_source=ig_web") == "CsO_xntBC_T"
    assert ig.extract_id("https://www.instagram.com/tv/B8xyz123_abc/#target") == "B8xyz123_abc"
    assert ig.extract_id("https://instagr.am/p/ABC_123-xyz/") == "ABC_123-xyz"

    # Fallback resilience
    assert ig.extract_id("https://www.instagram.com/stories/username/1234567890/") == "1234567890"
    assert isinstance(ig.extract_id("https://instagram.com/"), str)
    assert isinstance(ig.extract_id(""), str)


def test_tiktok_extractor_adversarial_urls():
    """Stress test TikTokExtractor.extract_id with malformed, complex, and adversarial URLs."""
    tt = TikTokExtractor()

    assert tt.extract_id("https://www.tiktok.com/@allblacks/video/7234567890123456789") == "7234567890123456789"
    assert tt.extract_id("https://www.tiktok.com/@user/v/7234567890123456789?lang=en") == "7234567890123456789"
    assert tt.extract_id("https://vm.tiktok.com/ZM8abc123/") == "ZM8abc123"
    assert tt.extract_id("https://www.tiktok.com/t/ZT8xyz456/?utm_campaign=share") == "ZT8xyz456"

    # Fallback resilience
    assert isinstance(tt.extract_id("https://www.tiktok.com/@creator"), str)
    assert isinstance(tt.extract_id(""), str)


def test_youtube_extractor_adversarial_urls():
    """Stress test YouTubeExtractor.extract_id with malformed, complex, and adversarial URLs."""
    yt = YouTubeExtractor()

    assert yt.extract_id("https://www.youtube.com/watch?v=zzXuxeuKlLI") == "zzXuxeuKlLI"
    assert yt.extract_id("https://www.youtube.com/watch?feature=share&v=zzXuxeuKlLI&t=10s") == "zzXuxeuKlLI"
    assert yt.extract_id("https://youtu.be/zwt1LE8X5yI?si=abcdef123") == "zwt1LE8X5yI"
    assert yt.extract_id("https://www.youtube.com/shorts/AbCdEfGhIjK") == "AbCdEfGhIjK"
    assert yt.extract_id("https://www.youtube.com/embed/AbCdEfGhIjK?rel=0") == "AbCdEfGhIjK"
    assert yt.extract_id("https://www.youtube.com/v/AbCdEfGhIjK") == "AbCdEfGhIjK"

    # Fallback resilience
    assert isinstance(yt.extract_id("https://youtube.com/"), str)
    assert isinstance(yt.extract_id(""), str)


# =====================================================================
# 4. ERROR CLASSIFICATION COMPREHENSIVE TAXONOMY & REAL YT-DLP ERRORS
# =====================================================================

def test_classify_error_real_world_ytdlp_traces():
    """Verify error classifier accurately categorizes authentic yt-dlp error outputs."""
    permanent_cases = [
        "ERROR: [facebook] 1000777895026531: This video is private",
        "ERROR: [Instagram] CsO_xntBC_T: Post unavailable or deleted",
        "ERROR: [youtube] dQw4w9WgXcQ: Video unavailable. This video is not available",
        "ERROR: [youtube] dQw4w9WgXcQ: Sign in if you have been granted access to this video",
        "ERROR: [generic] None: Unsupported URL: https://invalid-site.com/foo",
        "ERROR: [tiktok] 723456789: Content is not available",
        "ERROR: [facebook] 1234: HTTP Error 404: Not Found",
        "ERROR: [facebook] 1234: HTTP Error 410: Gone",
        "ERROR: [youtube] 1234: This video has been removed for violating YouTube's Terms of Service",
        "ERROR: [youtube] 1234: Video unavailable. This video has been removed by the uploader",
        "ERROR: [facebook] 1234: Login required to view this post",
        "ERROR: [youtube] 1234: copyright claim takedown",
        "ERROR: [Instagram] 1234: Account is private",
        "ERROR: [generic] No video formats found in the extracted webpage",
        "ERROR: [facebook] This content is no longer available",
    ]

    for msg in permanent_cases:
        cls = classify_error(msg)
        assert cls == "permanent", f"Expected 'permanent' for: {msg}, got {cls}"
        assert is_permanent_error(msg) is True
        assert is_transient_error(msg) is False

    transient_cases = [
        "ERROR: [generic] Unable to download webpage: The read operation timed out",
        "ERROR: [generic] HTTP Error 429: Too Many Requests",
        "ERROR: [generic] HTTP Error 500: Internal Server Error",
        "ERROR: [generic] HTTP Error 502: Bad Gateway",
        "ERROR: [generic] HTTP Error 503: Service Unavailable",
        "ERROR: [generic] HTTP Error 504: Gateway Timeout",
        "ERROR: [download] Got error: Connection reset by peer",
        "ERROR: [download] Got error: Remote end closed connection without response",
        "ERROR: [download] Got error: [Errno 32] Broken pipe",
        "ERROR: [generic] <urlopen error [Errno 61] Connection refused>",
        "ERROR: [generic] <urlopen error [Errno 51] Network is unreachable>",
        "ERROR: [generic] <urlopen error [Errno 8] Temporary failure in name resolution>",
        "ERROR: [generic] <urlopen error [Errno 8] Name or service not known>",
        "ERROR: [generic] <urlopen error TLS handshake failed>",
        "ERROR: [generic] EOF occurred in violation of protocol (_ssl.c:1129)",
        "ERROR: socket timeout occurred while waiting for chunk",
    ]

    for msg in transient_cases:
        cls = classify_error(msg)
        assert cls == "transient", f"Expected 'transient' for: {msg}, got {cls}"
        assert is_transient_error(msg) is True
        assert is_permanent_error(msg) is False


def test_classify_error_with_exception_objects_and_nesting():
    """Verify classify_error handles raw Exception objects and nested causes."""
    class CustomPermanent(Exception):
        pass

    class CustomTransient(Exception):
        pass

    e1 = CustomPermanent("HTTP Error 404: Not Found")
    assert classify_error(e1) == "permanent"

    e2 = CustomTransient("Connection reset by peer")
    assert classify_error(e2) == "transient"

    # Nested exception
    try:
        try:
            raise TimeoutError("The read operation timed out")
        except TimeoutError as orig:
            raise RuntimeError("Extraction failed") from orig
    except RuntimeError as wrapped:
        # Note: str(wrapped) is 'Extraction failed', but if message contains 'timeout':
        assert classify_error(wrapped.args[0]) == "unknown"
        assert classify_error(repr(wrapped)) == "transient" or classify_error(str(wrapped.__cause__)) == "transient"


# =====================================================================
# 5. BUILD_YDL_OPTS & YOUTUBEDL INITIALIZATION MATRIX
# =====================================================================

def test_build_ydl_opts_instantiation_in_youtubedl():
    """Verify yt-dlp's YoutubeDL class cleanly accepts all build_ydl_opts configurations."""
    # Test matrix of configurations
    configs = [
        build_ydl_opts(),
        build_ydl_opts(output_path="/tmp/test.mp4", socket_timeout=30),
        build_ydl_opts(socket_timeout=5, retries=1, fragment_retries=1),
        build_ydl_opts(format_override="bestvideo[height<=480]+bestaudio/best"),
        build_ydl_opts(quiet=False, nocheckcertificate=False, skip_unavailable_fragments=False),
    ]

    for opts in configs:
        with yt_dlp.YoutubeDL(opts) as ydl:
            assert ydl.params["format"] == opts["format"]
            assert ydl.params["socket_timeout"] == opts["socket_timeout"]
            assert ydl.params["merge_output_format"] == "mp4"


def test_base_extractor_error_translation_mocking(tmp_path: Path):
    """Verify BaseExtractor properly catches yt-dlp exceptions and raises structured subtypes."""
    ext = FacebookExtractor()

    # Permanent error simulation
    with patch("yt_dlp.YoutubeDL.extract_info", side_effect=Exception("HTTP Error 404: Not Found")):
        with pytest.raises(PermanentExtractionError) as exc_info:
            ext.resolve("https://facebook.com/reel/12345")
        assert "404" in str(exc_info.value)

    # Transient error simulation
    with patch("yt_dlp.YoutubeDL.extract_info", side_effect=Exception("The read operation timed out")):
        with pytest.raises(TransientExtractionError) as exc_info:
            ext.resolve("https://facebook.com/reel/12345")
        assert "timed out" in str(exc_info.value)

    # Unknown error simulation
    with patch("yt_dlp.YoutubeDL.extract_info", side_effect=ValueError("Unexpected corrupt JSON")):
        with pytest.raises(MediaResolutionError) as exc_info:
            ext.resolve("https://facebook.com/reel/12345")
        assert not isinstance(exc_info.value, (PermanentExtractionError, TransientExtractionError))
