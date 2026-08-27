from __future__ import annotations

from src.video.extractors.base import BaseExtractor


class InstagramExtractor(BaseExtractor):
    """Extractor for Instagram Reels & Videos."""

    platform_name = "instagram"

    def extract_id(self, url: str) -> str:
        clean = str(url).split("?")[0].split("#")[0].rstrip("/")
        segs = [s for s in clean.split("/") if s]
        return segs[-1] if segs else clean
