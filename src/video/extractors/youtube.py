from __future__ import annotations

import re
from src.video.extractors.base import BaseExtractor


class YouTubeExtractor(BaseExtractor):
    """Extractor for YouTube Shorts & Videos."""

    platform_name = "youtube"

    def extract_id(self, url: str) -> str:
        m = re.search(r"[?&]v=([A-Za-z0-9_-]{6,})", url) or re.search(
            r"youtu\.be/([A-Za-z0-9_-]{6,})", url
        )
        if m:
            return m.group(1)
        clean = str(url).split("?")[0].split("#")[0].rstrip("/")
        segs = [s for s in clean.split("/") if s]
        return segs[-1] if segs else clean
