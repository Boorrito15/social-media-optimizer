from __future__ import annotations

import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from utils.config import (
    video_output_dir,
    video_request_delay,
    ytdlp_cookies,
    ytdlp_cookies_from_browser,
)


@dataclass
class ResolvedMedia:
    """A resolved, download-able media reference for one post."""

    platform: str
    post_id: str
    url: str
    info: dict = field(default_factory=dict)
    supported: bool = True
    direct_url: str | None = None

    @property
    def display_name(self) -> str:
        title = (self.info.get("title") or f"{self.platform}-{self.post_id}").strip()
        return re.sub(r"[^A-Za-z0-9._-]+", "-", title)[:120]


class MediaResolutionError(RuntimeError):
    """Raised when a post URL cannot be resolved to media."""


_DEFAULT_HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Sec-Fetch-Mode": "navigate",
}


def build_ydl_opts(
    *,
    cookiefile: str | None = None,
    cookies_from_browser: str | None = None,
    outtmpl: str | None = None,
    quiet: bool = True,
) -> dict:
    opts = {
        "quiet": quiet,
        "no_warnings": True,
        "noplaylist": True,
        "format": "bestvideo[height<=?720]+bestaudio/best[height<=?720]/best",
        "merge_output_format": "mp4",
        "outtmpl": outtmpl or "%(id)s.%(ext)s",
        "http_headers": _DEFAULT_HTTP_HEADERS,
        "retries": 3,
        "fragment_retries": 3,
    }

    cookiefile = cookiefile or ytdlp_cookies()
    if cookiefile and Path(cookiefile).is_file():
        opts["cookiefile"] = str(Path(cookiefile).expanduser())

    cookies_from_browser = cookies_from_browser or ytdlp_cookies_from_browser()
    if cookies_from_browser:
        opts["cookiesfrombrowser"] = (cookies_from_browser.strip().lower(),)

    return opts


class BaseExtractor(ABC):
    """Abstract base class for platform-specific extractors."""

    platform_name: str

    @abstractmethod
    def extract_id(self, url: str) -> str:
        pass

    def resolve(
        self,
        url: str,
        *,
        cookies: str | None = None,
        cookies_from_browser: str | None = None,
    ) -> ResolvedMedia:
        import yt_dlp

        opts = build_ydl_opts(
            cookiefile=cookies,
            cookies_from_browser=cookies_from_browser,
            quiet=True,
        )

        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=False)
                if info is None:
                    raise MediaResolutionError(f"yt-dlp returned no info for: {url}")
            entries = info.get("entries") or [info]
            top_info = dict(entries[0])
            post_id = str(top_info.get("id") or self.extract_id(url))
            direct_url = top_info.get("url") or (
                top_info.get("requested_downloads", [{}])[0].get("url")
                if top_info.get("requested_downloads")
                else None
            )
            return ResolvedMedia(
                platform=self.platform_name,
                post_id=post_id,
                url=url,
                info=top_info,
                supported=True,
                direct_url=direct_url,
            )
        except MediaResolutionError:
            raise
        except Exception as exc:
            raise MediaResolutionError(f"Failed to resolve [{self.platform_name}] {url}: {exc}") from exc

    def download(
        self,
        media: ResolvedMedia,
        out_dir: str | Path | None = None,
        overwrite: bool = False,
        *,
        cookies: str | None = None,
        cookies_from_browser: str | None = None,
    ) -> Path | None:
        import yt_dlp

        out_dir = Path(out_dir or video_output_dir()).expanduser().resolve()
        platform_dir = out_dir / media.platform
        platform_dir.mkdir(parents=True, exist_ok=True)

        target = (platform_dir / media.post_id).with_suffix(".mp4")
        if target.exists() and not overwrite:
            return target

        opts = build_ydl_opts(
            cookiefile=cookies,
            cookies_from_browser=cookies_from_browser,
            outtmpl=str(platform_dir / f"{media.post_id}.%(ext)s"),
            quiet=False,
        )

        delay = video_request_delay()
        if delay:
            time.sleep(delay)

        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([media.url])
        except Exception as exc:
            print(f"[video] Download failed [{media.platform}] {media.post_id}: {exc}")
            return None

        if target.exists():
            return target
        globbed = sorted(platform_dir.glob(f"{media.post_id}.*"))
        return globbed[0] if globbed else None
