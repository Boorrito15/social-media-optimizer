"""Video source resolution & download.

The scraping strategy is a *per-platform resolver with fallbacks*, not one
monolithic scraper. For the current scope:

* **YouTube** & **TikTok** are handled by ``yt-dlp`` directly (no browser).
* **Instagram** & **Facebook** are stubbed with a ``NotImplemented`` marker so
  the pipeline degrades gracefully and records the post as "needs login /
  not-yet-supported" instead of failing loudly.

``resolve()`` is the seam where a browser-based fallback (XHR interception /
Playwright) can later be added for IG/FB without touching the rest of the
pipeline: it just returns a *resolved media reference*, and the caller decides
how to consume it.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from utils.config import video_output_dir, video_request_delay, ytdlp_cookies

# Platforms this resolver has an automated (yt-dlp) path for.
SUPPORTED_EXTRACTORS = {"youtube", "tiktok"}
# Platforms that exist in the dataset but are not yet automated.
PLANNED_EXTRACTORS = {"instagram", "facebook"}

# Map the dataset's uppercase platform codes to resolver platform names.
PLATFORM_CODE_MAP = {
    "YT": "youtube",
    "TT": "tiktok",
    "IG": "instagram",
    "FB": "facebook",
}


def platform_code_to_name(code: str) -> str | None:
    """Translate a dataset platform code (e.g. ``YT``) to a resolver name (``youtube``)."""
    return PLATFORM_CODE_MAP.get(str(code).strip().upper())


def media_id_from_url(url: str, platform: str | None = None) -> str:
    """Extract a stable media id from a post URL without any network I/O.

    Uses only URL parsing and light regex:
      - YouTube  -> the ``v=`` query param (or youtu.be path)
      - TikTok   -> the trailing numeric id
      - IG / FB  -> the trailing path segment
    """
    platform = platform or _platform_from_url(url)
    if platform == "youtube":
        m = re.search(r"[?&]v=([A-Za-z0-9_-]{6,})", url) or re.search(
            r"youtu\.be/([A-Za-z0-9_-]{6,})", url
        )
        if m:
            return m.group(1)
    segs = [s for s in url.rstrip("/").split("/") if s]
    return segs[-1] if segs else url


def published_at_from_info(info: dict) -> str | None:
    """Normalise the publish timestamp from yt-dlp info to ISO-8601 UTC.

    Prefers the Unix ``timestamp``, falls back to ``upload_date`` (YYYYMMDD).
    Returns ``None`` if neither is present. Output like ``2026-03-22T12:00:32Z``.
    """
    ts = info.get("timestamp")
    if isinstance(ts, (int, float)) and ts:
        from datetime import datetime, timezone

        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    up = info.get("upload_date")
    if isinstance(up, str) and len(up) == 8 and up.isdigit():
        return f"{up[:4]}-{up[4:6]}-{up[6:8]}T00:00:00Z"
    return None

_DEFAULT_YDTPLP_OPTS = {
    "quiet": True,
    "no_warnings": True,
    "noplaylist": True,
    "format": "bestvideo[height<=?720]+bestaudio/best[height<=?720]/best",
    "merge_output_format": "mp4",
    "outtmpl": "%(id)s.%(ext)s",
}


@dataclass
class ResolvedMedia:
    """A resolved, download-able media reference for one post."""

    platform: str            # e.g. "youtube", "tiktok", "instagram", ...
    post_id: str             # extractor id (video id / reel code / tiktok id)
    url: str                 # canonical media URL (e.g. youtube.com/watch)
    info: dict = field(default_factory=dict)   # media metadata (title, duration...)
    supported: bool = False  # whether an automated download path exists
    direct_url: str | None = None  # a CDN URL if already known (yt/dl only)

    @property
    def display_name(self) -> str:
        title = (self.info.get("title") or f"{self.platform}-{self.post_id}").strip()
        # Keep filenames filesystem-safe.
        return re.sub(r"[^A-Za-z0-9._-]+", "-", title)[:120]


def _platform_from_url(url: str) -> str | None:
    host = (urlparse(url).netloc or "").lower()
    for name in ("youtube", "youtu.be"):
        if name in host:
            return "youtube"
    if "tiktok" in host:
        return "tiktok"
    if "instagram" in host:
        return "instagram"
    if any(x in host for x in ("facebook", "fb.watch")):
        return "facebook"
    return None


def _extract_id(url: str, platform: str) -> str:
    if platform == "youtube":
        m = re.search(r"[?&]v=([A-Za-z0-9_-]{6,})", url) or re.search(
            r"youtu\.be/([A-Za-z0-9_-]{6,})", url
        )
        return m.group(1) if m else url.rsplit("/", 1)[-1].split("?")[0]
    # TikTok & IG: last non-empty path segment.
    segs = [s for s in url.rstrip("/").split("/") if s]
    return segs[-1] if segs else url


def _info_to_resolved(platform: str, url: str, info: dict) -> ResolvedMedia:
    post_id = str(info.get("id") or _extract_id(url, platform))
    direct_url = info.get("url") or (
        info.get("requested_downloads", [{}])[0].get("url") if info.get("requested_downloads") else None
    )
    return ResolvedMedia(
        platform=platform,
        post_id=post_id,
        url=url,
        info=info,
        supported=platform in SUPPORTED_EXTRACTORS,
        direct_url=direct_url,
    )


def resolve(url: str, platform: str | None = None) -> ResolvedMedia:
    """Resolve a post URL to media metadata using yt-dlp.

    This only *inspects* the page (extract info) — it does not download. It
    returns the platform id/title and a direct CDN URL where yt-dlp can get it.
    Unsupported platforms (Instagram/Facebook) return a ``supported=False``
    reference so the caller can skip/queue them.

    Raises ``MediaResolutionError`` for a truly unresolvable URL.
    """
    platform = platform or _platform_from_url(url)
    if platform is None:
        raise MediaResolutionError(f"Unrecognised platform in URL: {url}")

    if platform not in SUPPORTED_EXTRACTORS:
        # No automated path yet — return a tombstone for graceful handling.
        return ResolvedMedia(
            platform=platform,
            post_id=_extract_id(url, platform),
            url=url,
            supported=False,
        )

    import yt_dlp

    opts = dict(_DEFAULT_YDTPLP_OPTS)
    cookies = ytdlp_cookies()
    if cookies:
        opts["cookiefile"] = cookies

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
            if info is None:
                raise MediaResolutionError(f"yt-dlp returned no info for: {url}")
        # normalize top-level dict (may be a playlist)
        entries = info.get("entries") or [info]
        return _info_to_resolved(platform, url, dict(entries[0]))
    except MediaResolutionError:
        raise
    except Exception as exc:  # yt-dlp raises broad exceptions
        raise MediaResolutionError(f"Failed to resolve {url}: {exc}") from exc


def download(
    media: ResolvedMedia,
    out_dir: str | Path | None = None,
    overwrite: bool = False,
) -> Path | None:
    """Download a resolved media reference into ``out_dir``.

    Returns the local file path, or ``None`` for unsupported platforms. The
    target file name is derived from the extractor id to stay stable across
    re-runs (``<out_dir>/<platform>/<post_id>.<ext>``).
    """
    if not media.supported:
        return None
    import yt_dlp

    out_dir = Path(out_dir or video_output_dir()).expanduser().resolve()
    platform_dir = out_dir / media.platform
    platform_dir.mkdir(parents=True, exist_ok=True)

    target = (platform_dir / media.post_id).with_suffix(".mp4")
    if target.exists() and not overwrite:
        return target

    opts = dict(_DEFAULT_YDTPLP_OPTS)
    opts["outtmpl"] = str(platform_dir / f"{media.post_id}.%(ext)s")
    opts["quiet"] = False
    cookies = ytdlp_cookies()
    if cookies:
        opts["cookiefile"] = cookies

    # Optional inter-request delay to stay under the platform's throttling
    # threshold when scraping anonymously (0 by default).
    delay = video_request_delay()
    if delay:
        time.sleep(delay)

    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([media.url])

    # Locate the produced file (mp4 merge; fall back to any file with the id).
    if target.exists():
        return target
    globbed = sorted(platform_dir.glob(f"{media.post_id}.*"))
    return globbed[0] if globbed else None


class MediaResolutionError(RuntimeError):
    """Raised when a post URL cannot be resolved to media."""
