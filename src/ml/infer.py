"""Auto-generate all metadata from a free-text description so the UI needs
exactly one input (what you type or speak into the mic).

Uses lightweight keyword rules — deterministic and fast. Every field the
prediction API expects is produced here.
"""

from __future__ import annotations

import re
from typing import Dict, List

# Platform / page detection -------------------------------------------------
PAGES = {
    "all blacks": "All Blacks",
    "black fern": "Black Ferns",
    "fern": "Black Ferns",
    "nz sevens": "NZ Sevens",
    "sevens": "NZ Sevens",
    "sevens team": "NZ Sevens",
    "nzr": "NZR",
    "abxv": "ABXV",
    "all blacks xv": "ABXV",
    "bunnings": "Bunnings NPC",
    "npc": "Bunnings NPC",
    "super rugby": "SRP",
    "super": "SRP",
    "all blacks": "All Blacks",
}

PLATFORM_HINTS = {
    "tt": "TT",
    "tiktok": "TT",
    "instagram": "IG",
    "reel": "IG",
    "facebook": "FB",
    "yt": "YT",
    "youtube": "YT",
    "shorts": "YT",
}

# Theme / format / tone keyword maps ---------------------------------------
THEMES = {
    "try": ["try", "score", "touch down", "grounds the ball", "in goal"],
    "rugby_skills": ["skill", "offload", "chip", "step", "dummy", "kick", "pass"],
    "training": ["training", "drill", "pre-season", "session", "practice", "gym"],
    "celebration": ["celebrat", "trophy", "win the final", "lift the cup", "title"],
    "haka": ["haka"],
    "tackle": ["tackle", "hit", "crunch", "dominant"],
    "kick": ["kick", "conversion", "penalty", "drop goal", "drop"],
    "rivalry": ["rival", "derby", "grudge"],
    "player story": ["debut", "return", "journey", "story", "comeback", "career"],
    "challenges": ["challenge", "behind the scenes", "day in the life"],
}

FORMATS = {
    "highlight": ["highlight", "try", "score", "moment", "best of", "clips", "play"],
    "behind-the-scenes": ["behind the scenes", "bts", "training", "locker", "tour"],
    "interview": ["interview", "speaks", "talks to", "qa", "press", "ber"],
    "announcement": ["announce", "reveal", "squad", "signing", "selection", "named"],
    "reaction": ["reaction", "reacts", "respond"],
    "candid clip": ["candid", "blooper", "funny", "joke"],
    "promotion": ["watch", "don't miss", "vs ", "kick-off", "live on", "tune in"],
}

TONES = {
    "exciting": ["spectacular", "unbelievable", "incredible", "amazing", "edge"],
    "emotional": ["emotional", "tears", "tribute", "farewell", "poignant"],
    "celebratory": ["celebrat", "trophy", "champion", "win", "party"],
    "playful": ["funny", "joke", "banter", "playful", "blooper"],
    "competitive": ["grudge", "rival", "must-win", "desperate", "battle"],
    "inspiring": ["inspire", "dream", "hard work", "dedication"],
}

DURATION_DEFAULT = 20


def infer(description: str) -> Dict:
    """Return a complete metadata dict for the given description text."""
    text = (description or "").strip()

    title = _make_title(text)

    page = _page(text)
    platform = _platform(text)
    themes = _match(text, THEMES)
    formats = _match(text, FORMATS)
    tones = _match(text, TONES)
    duration = _duration(text)

    return {
        "title": title,
        "description": text,
        "platform": platform,
        "page": page,
        "year": 2025,
        "category_l0": "No Hashtag",
        "category_l1": "No Hashtag",
        "category_l2": "No Hashtag",
        "duration_seconds": duration,
        "content_themes": themes,
        "format_access": formats,
        "tones": tones,
        "cost": 0.0,
        "expected_rpm": 3.0,
        "expected_cpm": 5.0,
    }


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _page(text: str) -> str:
    low = text.lower()
    for key, value in PAGES.items():
        if key in low:
            return value
    return "All Blacks"


def _platform(text: str) -> str:
    low = text.lower()
    for key, value in PLATFORM_HINTS.items():
        if key in low:
            return value
    return "FB"


def _match(text: str, mapping: Dict[str, List[str]]) -> List[str]:
    low = text.lower()
    hits = []
    for field, keywords in mapping.items():
        if any(k in low for k in keywords):
            hits.append(field)
    return hits


def _duration(text: str) -> int:
    # matches like "30 seconds", "a minute", "45 sec"
    m = re.search(r"(\d+)\s*(?:sec|seconds|s)\b", text.lower())
    if m:
        return max(5, min(int(m.group(1)), 60))
    if re.search(r"\b(minute|min)\b", text.lower()):
        return 45
    return DURATION_DEFAULT


def _make_title(description: str) -> str:
    text = description.strip()
    if not text:
        return ""
    # first clause/sentence, trimmed to a punchy hook
    import re as _re

    parts = _re.split(r"(?<=[.])\s+", text)
    hook = parts[0].strip().rstrip(".")
    # trim to a reasonable caption length
    if len(hook) > 80:
        hook = hook[:77].rstrip() + "…"
    return hook
