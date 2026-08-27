"""ffmpeg transcode helpers.

Used by the video pipeline to re-encode scraped videos to a consistent
480p profile (H.264/AAC) before uploading to GCS. Kept in ``utils/`` so the
same re-encode is reusable by the frame-extraction stage later.

Requires ``ffmpeg`` on PATH. A clear error is raised if it is missing so the
caller can surface exactly what to install rather than a confusing subprocess
failure.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from utils.config import video_codec, video_crf, video_target_height

# Rate limit knobs for the video transcode.
_DEFAULT_MAXRATE = "3M"
_DEFAULT_BUFSIZE = "6M"
_DEFAULT_AUDIO_BITRATE = "96k"
_DEFAULT_FPS_CAP = 30


def _base_filter_chain(target_height: int, fps_cap: int | None) -> list[str]:
    """Build the ffmpeg filter graph argument list.

    Ensures even dimensions, yuv420p for universal decode, capping fps at the
    configured value, and swapping scan-order to avoid stream issues. Applied
    *before* scaling so the fps cap operates on the decoded stream.
    """
    chain = []
    if fps_cap:
        chain.append(f"fps={fps_cap}")
    chain.append(f"scale=-2:{target_height}")
    chain.append("format=yuv420p")
    return ["-vf", ",".join(chain)]


def build_480p_profile(
    *,
    target_height: int | None = None,
    codec: str | None = None,
    audio_bitrate: str | None = None,
    crf: int | None = None,
    maxrate: str | None = None,
    bufsize: str | None = None,
    fps_cap: int | None = None,
    threads: int | None = None,
) -> list[str]:
    """Build a full ffmpeg output command for a 480p short-form profile.

    Any ``None`` parameter falls back to the values configured via environment
    (``VIDEO_TARGET_HEIGHT``, ``VIDEO_CODEC``, ``VIDEO_CRF``) or built-in
    defaults.

    Parameters
    ----------
    target_height : output height (default from ``VIDEO_TARGET_HEIGHT``).
    codec : ``h264`` (libx264, default, universally playable) or ``av1``
        (SVTAV1, smaller but slower; requires ffmpeg built with libsvtav1).
    audio_bitrate : AAC audio bitrate (default ``96k``).
    crf : x264/SVT-AV1 CRF (default from ``VIDEO_CRF``, fallback 23).
    maxrate / bufsize : VBV rate cap to stop high-motion clips ballooning.
    fps_cap : cap output frame rate (default 30).
    threads : ffmpeg encode threads. ``0`` means all cores; when running
        multiple workers in parallel, set this to roughly ``ncpu / workers``
        to avoid oversubscribing the machine.
    """
    target_height = target_height or video_target_height()
    codec = codec or video_codec()
    audio_bitrate = audio_bitrate or _DEFAULT_AUDIO_BITRATE
    maxrate = maxrate or _DEFAULT_MAXRATE
    bufsize = bufsize or _DEFAULT_BUFSIZE
    crf = int(crf if crf is not None else video_crf())
    if fps_cap is None:
        fps_cap = _DEFAULT_FPS_CAP
    if threads is None:
        threads = 0

    if codec == "av1":
        vcodec = ["-c:v", "libsvtav1", "-preset", "8", "-crf", str(crf)]
    else:
        vcodec = ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf)]

    return [
        *vcodec,
        "-maxrate", maxrate,
        "-bufsize", bufsize,
        *_base_filter_chain(target_height, fps_cap),
        "-c:a", "aac",
        "-b:a", audio_bitrate,
        "-movflags", "+faststart",
        "-threads", str(threads),
    ]


# Back-compat: previously the module exposed a bare list; some callers/tests may
# reference it. Keep it as the default H.264 profile.
DEFAULT_PROFILE = build_480p_profile(codec="h264")


@dataclass
class TranscodeResult:
    output: Path
    command: list[str]
    duration_s: float | None = None
    size_bytes: int = 0


def ffmpeg_available() -> bool:
    """Return True if an ``ffmpeg`` binary is on PATH."""
    return shutil.which("ffmpeg") is not None


def transcode_to_480p(
    input_path: str | Path,
    output_path: str | Path,
    *,
    profile: list[str] | None = None,
    dry_run: bool = False,
    overwrite: bool = False,
) -> TranscodeResult:
    """Re-encode ``input_path`` to a 480p H.264/AAC file at ``output_path``.

    Parameters
    ----------
    input_path : source media file.
    output_path : destination MP4 path (parent dir is created as needed).
    profile : complete ffmpeg output args; defaults to ``build_480p_profile``.
    dry_run : only print the command without executing.
    overwrite : allow overwriting an existing ``output_path``.
    """
    if not ffmpeg_available():
        raise RuntimeError(
            "ffmpeg is not installed/on PATH. Install it, e.g. "
            "`brew install ffmpeg` on macOS, `apt install ffmpeg` on Debian,"
            " or use the gcloud-compute env where it is preinstalled."
        )

    src = Path(input_path).expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(f"Input video not found: {src}")

    dst = Path(output_path).expanduser().resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists (use overwrite=True to replace): {dst}"
        )

    profile = profile if profile is not None else build_480p_profile()
    cmd = [
        "ffmpeg", "-y" if overwrite else "-n",
        "-i", str(src),
        *profile,
        "-nostdin",
        str(dst),
    ]

    if dry_run:
        print(f"[ffmpeg][dry-run] {' '.join(cmd)}")
        return TranscodeResult(output=dst, command=cmd)

    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed (exit {res.returncode}):\n"
            f"{res.stderr.strip()[-1500:]}"
        )
    size = dst.stat().st_size if dst.exists() else 0
    duration = probe_duration(src)
    return TranscodeResult(output=dst, command=cmd, duration_s=duration, size_bytes=size)


def probe_duration(path: str | Path) -> float | None:
    """Return the media duration in seconds using ffprobe (or None if unknown)."""
    src = Path(path).expanduser().resolve()
    if not shutil.which("ffprobe"):
        return None
    try:
        res = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(src)],
            capture_output=True, text=True,
        )
        if res.returncode == 0 and res.stdout.strip():
            return float(res.stdout.strip())
    except (ValueError, subprocess.SubprocessError):
        pass
    return None
