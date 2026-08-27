from src.video.extractors.facebook import FacebookExtractor
from src.video.extractors.instagram import InstagramExtractor
from src.video.extractors.tiktok import TikTokExtractor
from src.video.extractors.youtube import YouTubeExtractor


def test_instagram_extractor_id():
    ext = InstagramExtractor()
    assert ext.extract_id("https://www.instagram.com/reel/C5de2n2Pnh6/?igsh=MWQ=") == "C5de2n2Pnh6"
    assert ext.extract_id("https://instagram.com/p/CsO_xntBC_T") == "CsO_xntBC_T"


def test_facebook_extractor_id():
    ext = FacebookExtractor()
    assert ext.extract_id("https://www.facebook.com/reel/1000777895026531") == "1000777895026531"
    assert ext.extract_id("https://fb.watch/abcd1234/?mibextid=wwXIfr") == "abcd1234"


def test_youtube_extractor_id():
    ext = YouTubeExtractor()
    assert ext.extract_id("https://www.youtube.com/watch?v=zzXuxeuKlLI") == "zzXuxeuKlLI"
    assert ext.extract_id("https://youtu.be/zwt1LE8X5yI?feature=shared") == "zwt1LE8X5yI"


def test_tiktok_extractor_id():
    ext = TikTokExtractor()
    assert ext.extract_id("https://www.tiktok.com/@allblacks/video/7234567890123456789") == "7234567890123456789"
