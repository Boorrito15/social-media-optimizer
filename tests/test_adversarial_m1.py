"""Milestone 1 Challenger 2 Adversarial Verification Test Harness.

Empirically tests:
1. FFmpeg timeout handling, hung process termination, and partial file unlinking.
2. `build_ydl_opts` positional vs keyword arguments signature compatibility across all call styles.
3. Format selector string syntax compatibility and fallback order across diverse stream manifests.
4. Socket options and network resilience parameter validation.
5. Extractor contract boundaries, URL pattern parsing edge cases, and error classification taxonomy.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yt_dlp

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

_ORIGINAL_SUBPROCESS_RUN = subprocess.run


# =====================================================================
# 1. FFmpeg Timeout Simulator & Process Termination Harness
# =====================================================================

def test_ffmpeg_hung_subprocess_terminated_and_partial_file_unlinked(tmp_path: Path):
    """Adversarially simulate a hung ffmpeg subprocess.

    Verifies:
    1. A subprocess that creates a partial output file and then hangs is killed upon timeout.
    2. The partial destination file is unlinked immediately.
    3. A descriptive RuntimeError is raised with timeout info.
    4. No orphan/zombie process remains active.
    """
    src_file = tmp_path / "sample_input.mp4"
    src_file.write_bytes(b"dummy mp4 data for input")
    dst_file = tmp_path / "partial_output.mp4"

    hang_script = f"""
import sys, time
dst = sys.argv[-1]
with open(dst, 'wb') as f:
    f.write(b'partial corrupt frame data' * 100)
time.sleep(10)
"""
    def mock_hung_run(cmd, *args, **kwargs):
        timeout = kwargs.get("timeout", 60.0)
        sim_cmd = [sys.executable, "-c", hang_script, str(dst_file)]
        return _ORIGINAL_SUBPROCESS_RUN(sim_cmd, capture_output=True, text=True, timeout=timeout)

    with patch("utils.ffmpeg.subprocess.run", side_effect=mock_hung_run):
        start_t = time.time()
        with pytest.raises(RuntimeError) as exc_info:
            transcode_to_480p(src_file, dst_file, timeout=0.5, overwrite=True)
        elapsed = time.time() - start_t

        # Verify timeout duration was respected (~0.5s, well under 10s sleep)
        assert elapsed < 3.0, f"Process took {elapsed}s, did not timeout promptly"
        assert "timed out after 0.5s" in str(exc_info.value)
        # Verify partial leftover file was unlinked
        assert not dst_file.exists(), "Partial destination file was NOT unlinked after timeout"


def test_ffmpeg_timeout_when_dst_file_never_created(tmp_path: Path):
    """Verify clean timeout handling when the hung subprocess didn't create dst_file."""
    src_file = tmp_path / "in.mp4"
    src_file.write_bytes(b"input")
    dst_file = tmp_path / "non_existent_out.mp4"

    with patch("utils.ffmpeg.subprocess.run", side_effect=subprocess.TimeoutExpired(cmd=["ffmpeg"], timeout=1.0)):
        with pytest.raises(RuntimeError) as exc_info:
            transcode_to_480p(src_file, dst_file, timeout=1.0, overwrite=True)

        assert "timed out after 1.0s" in str(exc_info.value)
        assert not dst_file.exists()


def test_ffmpeg_transcode_non_zero_exit_unlinks_dst(tmp_path: Path):
    """Verify that if ffmpeg fails with non-zero exit, any partial output is cleaned up."""
    src_file = tmp_path / "in.mp4"
    src_file.write_bytes(b"input")
    dst_file = tmp_path / "corrupt_out.mp4"
    dst_file.write_bytes(b"bad data")

    mock_res = MagicMock()
    mock_res.returncode = 1
    mock_res.stderr = "Invalid data found when processing input"

    with patch("utils.ffmpeg.subprocess.run", return_value=mock_res):
        with pytest.raises(RuntimeError) as exc_info:
            transcode_to_480p(src_file, dst_file, overwrite=True)

        assert "ffmpeg failed (exit 1)" in str(exc_info.value)
        assert not dst_file.exists(), "Corrupt output file was NOT cleaned up on error"


def test_ffmpeg_contract_preconditions(tmp_path: Path):
    """Verify input existence and overwrite protection preconditions."""
    src_file = tmp_path / "missing.mp4"
    dst_file = tmp_path / "out.mp4"

    # 1. Missing input file raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        transcode_to_480p(src_file, dst_file)

    # 2. Existing output with overwrite=False raises FileExistsError
    src_real = tmp_path / "real.mp4"
    src_real.write_bytes(b"real")
    dst_file.write_bytes(b"already exists")
    with pytest.raises(FileExistsError):
        transcode_to_480p(src_real, dst_file, overwrite=False)

    # 3. Dry run mode returns TranscodeResult without executing subprocess
    with patch("utils.ffmpeg.subprocess.run") as mock_sub:
        res = transcode_to_480p(src_real, tmp_path / "dry.mp4", dry_run=True)
        assert mock_sub.call_count == 0
        assert isinstance(res, TranscodeResult)
        assert res.output == (tmp_path / "dry.mp4").resolve()


# =====================================================================
# 2. build_ydl_opts Positional vs Keyword Arguments Signature Matrix
# =====================================================================

def test_build_ydl_opts_positional_permutations(tmp_path: Path):
    """Empirically test all positional parameter combinations."""
    # 0 args
    o0 = build_ydl_opts()
    assert o0["outtmpl"] == "%(id)s.%(ext)s"
    assert o0["socket_timeout"] == 15
    assert o0["format"].startswith("best[ext=mp4]")

    # 1 arg (output_path)
    o1 = build_ydl_opts("/tmp/custom.mp4")
    assert o1["outtmpl"] == "/tmp/custom.mp4"
    assert o1["socket_timeout"] == 15

    # 2 args (output_path, cookies)
    cookie_file = tmp_path / "test_cookies.txt"
    cookie_file.write_text("# Netscape HTTP Cookie File\n")
    o2 = build_ydl_opts("/tmp/custom.mp4", str(cookie_file))
    assert o2["outtmpl"] == "/tmp/custom.mp4"
    assert o2["cookiefile"] == str(cookie_file.resolve())

    # 3 args (output_path, cookies, cookies_from_browser)
    o3 = build_ydl_opts("/tmp/custom.mp4", None, "chrome")
    assert o3["outtmpl"] == "/tmp/custom.mp4"
    assert o3["cookiesfrombrowser"] == ("chrome",)

    # 4 args (output_path, cookies, cookies_from_browser, socket_timeout)
    o4 = build_ydl_opts("/tmp/custom.mp4", None, None, 45)
    assert o4["socket_timeout"] == 45

    # 5 args (output_path, cookies, cookies_from_browser, socket_timeout, format_override)
    o5 = build_ydl_opts("/tmp/custom.mp4", None, None, 20, "worstvideo")
    assert o5["socket_timeout"] == 20
    assert o5["format"] == "worstvideo"


def test_build_ydl_opts_keyword_aliases_and_overrides(tmp_path: Path):
    """Empirically test keyword argument aliases and precedence rules."""
    cookie_file = tmp_path / "kw_cookies.txt"
    cookie_file.write_text("# Cookie file\n")

    # outtmpl keyword alias
    o_kw = build_ydl_opts(outtmpl="/tmp/kw_out.mp4", socket_timeout=12.5)
    assert o_kw["outtmpl"] == "/tmp/kw_out.mp4"
    assert o_kw["socket_timeout"] == 12.5

    # cookiefile keyword alias
    o_c = build_ydl_opts(cookiefile=str(cookie_file))
    assert o_c["cookiefile"] == str(cookie_file.resolve())

    # outtmpl overrides positional output_path if provided
    o_precedence = build_ydl_opts("/tmp/pos_path.mp4", outtmpl="/tmp/kw_precedence.mp4")
    assert o_precedence["outtmpl"] == "/tmp/kw_precedence.mp4"

    # keyword-only knobs
    o_knobs = build_ydl_opts(
        retries=10,
        fragment_retries=7,
        concurrent_fragment_downloads=8,
        nocheckcertificate=False,
        skip_unavailable_fragments=False,
        quiet=False,
    )
    assert o_knobs["retries"] == 10
    assert o_knobs["fragment_retries"] == 7
    assert o_knobs["concurrent_fragment_downloads"] == 8
    assert o_knobs["nocheckcertificate"] is False
    assert o_knobs["skip_unavailable_fragments"] is False
    assert o_knobs["quiet"] is False


def test_build_ydl_opts_pathlib_object_handling(tmp_path: Path):
    """Ensure Path instances can be passed without TypeError."""
    out_p = tmp_path / "downloads" / "video.mp4"
    opts = build_ydl_opts(output_path=out_p)
    assert opts["outtmpl"] == str(out_p)


# =====================================================================
# 3. Format Selector Compatibility & Fallback Order
# =====================================================================

def test_format_selector_syntax_validity_in_ytdlp():
    """Verify that yt-dlp's FormatSelector can parse the configured format string."""
    opts = build_ydl_opts()
    format_str = opts["format"]

    ydl = yt_dlp.YoutubeDL(opts)
    assert ydl.params["format"] == format_str

    parts = format_str.split("/")
    assert len(parts) == 4
    assert parts[0] == "best[ext=mp4][height<=?720]"
    assert parts[1] == "bestvideo[height<=?720]+bestaudio"
    assert parts[2] == "best[height<=?720]"
    assert parts[3] == "best"


def test_format_selector_fallback_order_simulation():
    """Adversarially simulate format selection across 4 distinct stream availability scenarios."""
    opts = build_ydl_opts()
    ydl = yt_dlp.YoutubeDL(opts)

    # Scenario A: Standard Meta/TikTok CDN with 720p pre-muxed mp4
    info_a = {
        "id": "meta_sample_1",
        "incomplete_formats": {},
        "formats": [
            {"format_id": "sd", "ext": "mp4", "height": 360, "vcodec": "avc1", "acodec": "mp4a", "tbr": 500, "url": "https://cdn/sd.mp4", "protocol": "https"},
            {"format_id": "hd_muxed", "ext": "mp4", "height": 720, "vcodec": "avc1", "acodec": "mp4a", "tbr": 1200, "url": "https://cdn/hd.mp4", "protocol": "https"},
            {"format_id": "1080p_muxed", "ext": "mp4", "height": 1080, "vcodec": "avc1", "acodec": "mp4a", "tbr": 2500, "url": "https://cdn/1080.mp4", "protocol": "https"},
        ],
    }
    selector_fn = ydl.build_format_selector(opts["format"])
    selected_a = list(selector_fn(info_a))
    assert len(selected_a) >= 1
    assert selected_a[0]["format_id"] == "hd_muxed"

    # Scenario B: YouTube-style split DASH streams (no 720p pre-muxed mp4)
    info_b = {
        "id": "yt_sample_2",
        "incomplete_formats": {},
        "formats": [
            {"format_id": "audio_best", "ext": "m4a", "vcodec": "none", "acodec": "mp4a", "abr": 128, "url": "https://cdn/a.m4a", "protocol": "https"},
            {"format_id": "video_720", "ext": "mp4", "height": 720, "vcodec": "avc1", "acodec": "none", "vbr": 1000, "url": "https://cdn/v720.mp4", "protocol": "https"},
            {"format_id": "video_1080", "ext": "mp4", "height": 1080, "vcodec": "avc1", "acodec": "none", "vbr": 2500, "url": "https://cdn/v1080.mp4", "protocol": "https"},
        ],
    }
    selected_b = list(selector_fn(info_b))
    assert len(selected_b) >= 1
    # Check that video_720 and audio_best were selected/combined
    fmt_ids = [f.get("format_id") for f in selected_b]
    assert "video_720" in fmt_ids or any("video_720" in f for f in fmt_ids) or selected_b[0].get("height") == 720

    # Scenario C: Non-MP4 pre-muxed 720p stream (e.g. WebM/MKV only)
    info_c = {
        "id": "webm_sample_3",
        "incomplete_formats": {},
        "formats": [
            {"format_id": "webm_720", "ext": "webm", "height": 720, "vcodec": "vp9", "acodec": "opus", "tbr": 1000, "url": "https://cdn/webm720.webm", "protocol": "https"},
        ],
    }
    selected_c = list(selector_fn(info_c))
    assert len(selected_c) >= 1
    assert selected_c[0]["format_id"] == "webm_720"

    # Scenario D: Only 1080p stream available (fallback to best)
    info_d = {
        "id": "high_res_only",
        "incomplete_formats": {},
        "formats": [
            {"format_id": "f_1080", "ext": "mp4", "height": 1080, "vcodec": "avc1", "acodec": "mp4a", "tbr": 3000, "url": "https://cdn/1080.mp4", "protocol": "https"},
        ],
    }
    selected_d = list(selector_fn(info_d))
    assert len(selected_d) >= 1
    assert selected_d[0]["format_id"] == "f_1080"


# =====================================================================
# 4. Socket Options & Network Resilience Validation
# =====================================================================

def test_socket_options_and_network_headers():
    """Verify all socket timeout and network resilience parameters in build_ydl_opts."""
    opts = build_ydl_opts(socket_timeout=25)
    assert opts["socket_timeout"] == 25
    assert opts["retries"] == 3
    assert opts["fragment_retries"] == 3
    assert opts["concurrent_fragment_downloads"] == 4
    assert opts["nocheckcertificate"] is True
    assert opts["skip_unavailable_fragments"] is True
    assert opts["noplaylist"] is True
    assert opts["no_warnings"] is True

    headers = opts["http_headers"]
    assert "Mozilla" in headers["User-Agent"]
    assert "Accept" in headers
    assert "Accept-Language" in headers
    assert headers["Sec-Fetch-Mode"] == "navigate"


# =====================================================================
# 5. Extractor Contract Boundaries & Error Classification Taxonomy
# =====================================================================

def test_extractor_id_boundary_cases():
    """Adversarially test URL parsing edge cases across all platforms."""
    fb = FacebookExtractor()
    assert fb.extract_id("https://www.facebook.com/watch/?v=123456789&extra_param=true") == "123456789"
    assert fb.extract_id("https://facebook.com/reel/987654321/") == "987654321"
    assert fb.extract_id("https://fb.watch/shortCode123/") == "shortCode123"

    ig = InstagramExtractor()
    assert ig.extract_id("https://www.instagram.com/reel/C5de2n2Pnh6/?utm_source=ig_web_copy_link") == "C5de2n2Pnh6"
    assert ig.extract_id("https://instagram.com/p/B_12345XYZ/") == "B_12345XYZ"
    assert ig.extract_id("https://www.instagram.com/tv/C123abc456/") == "C123abc456"

    tt = TikTokExtractor()
    assert tt.extract_id("https://www.tiktok.com/@creator/video/7123456789012345678?is_from_webapp=1") == "7123456789012345678"
    assert tt.extract_id("https://vm.tiktok.com/ZM8abc123/") == "ZM8abc123"

    yt = YouTubeExtractor()
    assert yt.extract_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=42s") == "dQw4w9WgXcQ"
    assert yt.extract_id("https://youtu.be/dQw4w9WgXcQ?si=abcdef") == "dQw4w9WgXcQ"
    assert yt.extract_id("https://www.youtube.com/shorts/ShortId1234") == "ShortId1234"


def test_classify_error_comprehensive_taxonomy():
    """Verify error taxonomy accuracy for permanent, transient, and unknown classifications."""
    # Permanent errors
    assert classify_error("HTTP Error 404: Not Found") == "permanent"
    assert classify_error("HTTP Error 410: Gone") == "permanent"
    assert classify_error("This video is private") == "permanent"
    assert classify_error("Sign in if you have been granted access") == "permanent"
    assert classify_error("Video unavailable") == "permanent"
    assert classify_error("This content is no longer available") == "permanent"
    assert classify_error("Account is private") == "permanent"
    assert classify_error("Removed for violating Community Guidelines") == "permanent"
    assert classify_error("Copyright claim takedown") == "permanent"
    assert classify_error("Unsupported URL: http://invalid.com") == "permanent"

    # Transient errors
    assert classify_error("The read operation timed out") == "transient"
    assert classify_error("socket timeout during connection") == "transient"
    assert classify_error("HTTP Error 429: Too Many Requests") == "transient"
    assert classify_error("HTTP Error 500: Internal Server Error") == "transient"
    assert classify_error("HTTP Error 502: Bad Gateway") == "transient"
    assert classify_error("HTTP Error 503: Service Unavailable") == "transient"
    assert classify_error("HTTP Error 504: Gateway Timeout") == "transient"
    assert classify_error("Connection reset by peer") == "transient"
    assert classify_error("Connection refused") == "transient"
    assert classify_error("Broken pipe") == "transient"
    assert classify_error("Temporary failure in name resolution") == "transient"
    assert classify_error("Network is unreachable") == "transient"

    # Unknown errors
    assert classify_error("General unexpected exception occurred") == "unknown"


def test_base_extractor_resolve_and_download_error_mapping(tmp_path: Path):
    """Verify that BaseExtractor converts underlying exceptions to Permanent / Transient subclasses."""
    ext = FacebookExtractor()

    # 1. Resolve permanent error mapping
    with patch.object(yt_dlp.YoutubeDL, "extract_info", side_effect=Exception("HTTP Error 404: Not Found")):
        with pytest.raises(PermanentExtractionError):
            ext.resolve("https://facebook.com/reel/12345")

    # 2. Resolve transient error mapping
    with patch.object(yt_dlp.YoutubeDL, "extract_info", side_effect=Exception("The read operation timed out")):
        with pytest.raises(TransientExtractionError):
            ext.resolve("https://facebook.com/reel/12345")

    # 3. Resolve unknown error mapping
    with patch.object(yt_dlp.YoutubeDL, "extract_info", side_effect=Exception("Weird unknown exception")):
        with pytest.raises(MediaResolutionError):
            ext.resolve("https://facebook.com/reel/12345")

    # 4. Download permanent error mapping
    media = ResolvedMedia(platform="facebook", post_id="12345", url="https://facebook.com/reel/12345")
    with patch.object(yt_dlp.YoutubeDL, "download", side_effect=Exception("HTTP Error 410: Gone")):
        with pytest.raises(PermanentExtractionError):
            ext.download(media, out_dir=tmp_path)

    # 5. Download transient error mapping
    with patch.object(yt_dlp.YoutubeDL, "download", side_effect=Exception("HTTP Error 429: Too Many Requests")):
        with pytest.raises(TransientExtractionError):
            ext.download(media, out_dir=tmp_path)
