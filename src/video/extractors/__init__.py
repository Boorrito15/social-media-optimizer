from src.video.extractors.base import BaseExtractor, MediaResolutionError, ResolvedMedia
from src.video.extractors.facebook import FacebookExtractor
from src.video.extractors.instagram import InstagramExtractor
from src.video.extractors.tiktok import TikTokExtractor
from src.video.extractors.youtube import YouTubeExtractor

EXTRACTORS = {
    "youtube": YouTubeExtractor(),
    "tiktok": TikTokExtractor(),
    "instagram": InstagramExtractor(),
    "facebook": FacebookExtractor(),
}

__all__ = [
    "BaseExtractor",
    "ResolvedMedia",
    "MediaResolutionError",
    "YouTubeExtractor",
    "TikTokExtractor",
    "InstagramExtractor",
    "FacebookExtractor",
    "EXTRACTORS",
]
