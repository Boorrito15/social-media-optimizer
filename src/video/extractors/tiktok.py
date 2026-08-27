from __future__ import annotations

from src.video.extractors.base import BaseExtractor


class TikTokExtractor(BaseExtractor):
    """Extractor for TikTok Videos."""

    platform_name = "tiktok"

    def extract_id(self, url: str) -> str:
        clean = str(url).split("?")[0].split("#")[0].rstrip("/")
        segs = [s for s in clean.split("/") if s]
        return segs[-1] if segs else clean
