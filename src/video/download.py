"""Video source resolution & download orchestrator.

Routes URLs to dedicated platform extractors:
- Instagram: src.video.extractors.instagram.InstagramExtractor
- Facebook:  src.video.extractors.facebook.FacebookExtractor
- YouTube:   src.video.extractors.youtube.YouTubeExtractor
- TikTok:    src.video.extractors.tiktok.TikTokExtractor
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

from src.video.extractors import EXTRACTORS, MediaResolutionError, ResolvedMedia
from src.video.extractors.base import build_ydl_opts, build_ydl_opts as _build_ydl_opts

# All platforms automated and supported
SUPPORTED_EXTRACTORS = set(EXTRACTORS.keys())
PLANNED_EXTRACTORS: set[str] = set()

PLATFORM_CODE_MAP = {
    "YT": "youtube",
    "TT": "tiktok",
    "IG": "instagram",
    "FB": "facebook",
}


def platform_code_to_name(code: str) -> str | None:
    return PLATFORM_CODE_MAP.get(str(code).strip().upper())


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


def media_id_from_url(url: str, platform: str | None = None) -> str:
    platform = platform or _platform_from_url(url)
    extractor = EXTRACTORS.get(platform)
    if extractor:
        return extractor.extract_id(url)
    clean = str(url).split("?")[0].split("#")[0].rstrip("/")
    segs = [s for s in clean.split("/") if s]
    return segs[-1] if segs else clean


def published_at_from_info(info: dict) -> str | None:
    ts = info.get("timestamp")
    if isinstance(ts, (int, float)) and ts:
        from datetime import datetime, timezone

        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    up = info.get("upload_date")
    if isinstance(up, str) and len(up) == 8 and up.isdigit():
        return f"{up[:4]}-{up[4:6]}-{up[6:8]}T00:00:00Z"
    return None


def resolve(
    url: str,
    platform: str | None = None,
    *,
    cookies: str | None = None,
    cookies_from_browser: str | None = None,
) -> ResolvedMedia:
    platform = platform or _platform_from_url(url)
    if platform is None:
        raise MediaResolutionError(f"Unrecognised platform in URL: {url}")

    extractor = EXTRACTORS.get(platform)
    if not extractor:
        return ResolvedMedia(
            platform=platform,
            post_id=media_id_from_url(url, platform),
            url=url,
            supported=False,
        )

    return extractor.resolve(
        url,
        cookies=cookies,
        cookies_from_browser=cookies_from_browser,
    )


def download(
    media: ResolvedMedia,
    out_dir: str | Path | None = None,
    overwrite: bool = False,
    *,
    cookies: str | None = None,
    cookies_from_browser: str | None = None,
) -> Path | None:
    if not media.supported:
        return None

    extractor = EXTRACTORS.get(media.platform)
    if not extractor:
        return None

    return extractor.download(
        media,
        out_dir=out_dir,
        overwrite=overwrite,
        cookies=cookies,
        cookies_from_browser=cookies_from_browser,
    )
