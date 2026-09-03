"""Auto-generate all metadata from a free-text description aligned with
notebooks/describe_rob.ipynb, notebooks/rob.ipynb, and models/reference.md.

Uses deterministic keyword classification against the controlled vocabularies.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Platform / page detection
# ---------------------------------------------------------------------------
PAGES = {
    "all blacks": "All Blacks",
    "black fern": "Black Ferns",
    "fern": "Black Ferns",
    "nz sevens": "NZ Sevens",
    "sevens": "NZ Sevens",
    "nzr": "NZR",
    "abxv": "ABXV",
    "all blacks xv": "ABXV",
    "bunnings": "Bunnings NPC",
    "npc": "Bunnings NPC",
    "super rugby": "SRP",
    "srp": "SRP",
    "pacific": "SRP",
}

PLATFORM_HINTS = {
    "tt": "TT",
    "tiktok": "TT",
    "instagram": "IG",
    "ig": "IG",
    "reel": "IG",
    "facebook": "FB",
    "fb": "FB",
    "yt": "YT",
    "youtube": "YT",
    "shorts": "YT",
}

# ---------------------------------------------------------------------------
# Controlled Vocabularies from notebooks/describe_rob.ipynb & models/reference.md
# ---------------------------------------------------------------------------

THEMES = {
    "try": ["try", "score", "touch down", "ground the ball", "grounds the ball", "in-goal", "crosses the chalk", "corner"],
    "celebration": ["celebrat", "trophy", "win the final", "lift the cup", "title", "champ", "cheering", "hug", "jubilation"],
    "rugby_skills": ["skill", "offload", "chip", "step", "dummy", "pass", "fend", "cross-kick", "handling", "flair"],
    "training": ["training", "drill", "pre-season", "session", "practice", "gym", "treadmill", "weights", "conditioning"],
    "challenges": ["challenge", "day in the life", "q&a", "crossbar", "quiz"],
    "player story": ["debut", "return", "journey", "story", "comeback", "career", "veteran", "retire", "tribute", "farewell"],
    "kick": ["kick", "conversion", "penalty", "drop goal", "penalty goal", "touchline kick"],
    "tackle": ["tackle", "hit", "crunch", "smash", "dominant", "turnover", "defense", "defend"],
    "haka": ["haka", "ka mate", "kapa o pango", "challenge"],
    "rivalry": ["rival", "derby", "grudge", "vs", "clash", "showdown", "battle", "springboks", "wallabies", "ireland", "france", "england"],
    "run": ["sprint", "break", "line break", "downfield", "burst", "dash", "speed", "charge", "gallop"],
    "conversion": ["conversion", "add the two", "two points", "extras"],
    "non rugby related": [
        "bad", "terrible", "boring", "awful", "trash", "garbage", "poor", "dark",
        "blurry", "static", "nothing happens", "random", "test", "nonsense", "not rugby",
        "irrelevant", "unrelated", "fail", "slow", "quiet", "blank", "asleep", "cat", "cooking"
    ],
}

FORMATS = {
    "highlight": ["highlight", "try", "score", "moment", "best of", "clips", "play", "action", "breakout"],
    "behind-the-scenes": ["behind the scenes", "bts", "training", "locker", "tour", "changing room", "bus", "tunnel"],
    "interview": ["interview", "speaks", "talks to", "qa", "press conference", "microphone", "reflects"],
    "announcement": ["announce", "reveal", "squad", "signing", "selection", "named", "unveil"],
    "reaction": ["reaction", "reacts", "respond", "face", "shocked"],
    "candid clip": ["candid", "blooper", "funny", "joke", "prank", "laugh", "messing around"],
    "archive": ["archive", "throwback", "classic", "historic", "retro", "years ago", "vault"],
    "promotion": ["watch", "don't miss", "kick-off", "live on", "tune in", "match day", "tickets"],
    "ticket sales": ["buy tickets", "get tickets", "ticket sales", "seats available"],
    "montage": ["montage", "compilation", "tribute", "season recap", "mix"],
}

TONES = {
    "excitement": ["spectacular", "unbelievable", "incredible", "amazing", "electric", "thrilling", "insane", "sprints", "erupts", "roaring", "crunching"],
    "pride": ["pride", "proud", "jersey", "all black", "black fern", "honour", "legacy", "tradition", "anthem"],
    "tension": ["tension", "tense", "nail-biter", "final minute", "pressure", "clutch", "last-minute", "edge of seat"],
    "nostalgia": ["nostalgia", "throwback", "memories", "legend", "historic", "reminiscing", "golden era"],
    "humour": ["funny", "joke", "banter", "humour", "humor", "laugh", "blooper", "prank", "hilarious"],
    "wholesome": ["wholesome", "heartwarming", "family", "kids", "fan", "signed", "smile", "kindness"],
    "solemn": ["solemn", "memorial", "tribute", "respect", "sadness", "farewell", "retiring", "injury", "loss", "terrible", "boring", "bad", "poor"],
    "sadness": ["sad", "heartbreak", "defeat", "loss", "tears", "crying", "disappointment", "injury"],
    "lighthearted": ["lighthearted", "casual", "relaxed", "fun", "chill", "coffee", "hotel", "travel"],
    "provocative": ["provocative", "controversial", "referee", "card", "red card", "yellow card", "feud"],
}

CONTEXTS = {
    "match day": ["match day", "stadium", "crowd", "final", "kick-off", "pitch", "whistle", "game", "test match"],
    "pre-match": ["pre-match", "warm-up", "tunnel", "changing room", "anthem"],
    "post-match": ["post-match", "press conference", "trophy", "celebration", "interview", "after the whistle"],
    "gym": ["gym", "weights", "lifting", "bench", "squat"],
    "changing room": ["changing room", "locker room", "sheds"],
    "press conference": ["press conference", "media", "journalists"],
    "tour": ["tour", "overseas", "hotel", "flight", "bus", "travel"],
    "off-season": ["off-season", "break", "holiday"],
    "squad naming": ["squad naming", "team naming", "announcement"],
    "jersey reveal": ["jersey reveal", "kit", "new jersey"],
    "award": ["award", "player of the year", "medal"],
}

AUDIO_FORMATS = {
    "voice": ["interview", "speaks", "talking", "speech", "huddle", "commentary", "commentator", "voice"],
    "song": ["song", "music", "beat", "anthem", "track"],
    "ambient": ["stadium", "crowd", "roaring", "cheering", "applause", "ambient", "cheers", "sound"],
    "none": ["quiet", "silent", "muted", "dark", "boring", "static", "nothing", "none"],
    "other": ["sound effects", "sfx", "other"],
}

TEAM_CATEGORIES = {
    "women": ["black fern", "women", "girls", "female", "farah palmer", "aupiki"],
    "men": ["all black", "men", "male", "boys", "npc", "super rugby"],
    "veterans": ["veteran", "legend", "retired", "classic ab", "former player"],
    "maori": ["maori", "all blacks maori", "tangata"],
    "youth": ["u20", "youth", "school", "junior", "u85"],
}

PEOPLE_MAP = {

    "all blacks": ["all blacks", "all black", "ab", "kiwis", "men in black"],
    "black ferns": ["black ferns", "black fern", "ferns"],
    "nz sevens": ["sevens", "nz sevens", "all blacks sevens", "black ferns sevens"],
    "all blacks xv": ["all blacks xv", "abxv", "nz xv"],
    "beauden barrett": ["beauden", "beauden barrett", "b. barrett"],
    "ardie savea": ["ardie", "savea", "ardie savea"],
    "will jordan": ["will jordan", "jordan"],
    "rieo ioane": ["rieo", "ioane", "rieo ioane", "reiko ioane"],
    "codie taylor": ["codie", "taylor", "codie taylor"],
    "scott barrett": ["scott barrett", "captain barrett"],
    "sam cane": ["sam cane", "cane"],
    "damian mckenzie": ["damian", "mckenzie", "dmac", "damian mckenzie"],
    "caleb clarke": ["caleb", "clarke", "caleb clarke"],
    "jordie barrett": ["jordie", "jordie barrett"],
    "cam roigard": ["roigard", "cam roigard"],
    "tyrel lomax": ["lomax", "tyrel lomax"],
    "tamaiti williams": ["tamaiti", "tamaiti williams"],
    "sevu reece": ["sevu", "reece", "sevu reece"],
    "tupou vaa'i": ["vaa'i", "vaai", "tupou vaa'i"],
    "asafo aumua": ["aumua", "asafo aumua"],
    "wallabies": ["wallabies", "australia", "aussie"],
    "springboks": ["springboks", "springbok", "south africa", "boks"],
    "england rugby": ["england rugby", "england", "red roses"],
    "ireland rugby": ["ireland rugby", "ireland", "irish"],
    "france rugby": ["france rugby", "france", "french", "les bleus"],
}

BRAND_MAP = {
    "adidas": ["adidas", "three stripes", "jersey", "boots", "kit"],
    "ineos": ["ineos"],
    "altrad": ["altrad"],
    "tudor": ["tudor", "watch", "timepiece"],
    "ford": ["ford", "truck", "ranger"],
    "asahi": ["asahi", "beer"],
    "smart rugby": ["smart rugby", "smart ball", "microchip"],
    "sky sport": ["sky sport", "sky", "broadcast", "commentary"],
    "nib": ["nib", "health", "insurance"],
    "gatorade": ["gatorade", "hydration", "sports drink"],
    "red bull": ["red bull", "energy drink"],
    "aig": ["aig"],
    "barfoot & thompson": ["barfoot", "thompson", "barfoot & thompson"],
}


DURATION_DEFAULT = 20.0


def infer(description: str) -> Dict[str, Any]:
    """Return a complete metadata dict aligned with describe_rob.ipynb and reference.md."""
    text = (description or "").strip()
    low = text.lower()

    title = _make_title(text)
    page = _page(low)
    platform = _platform(low)

    # Check for negative / low quality / non-rugby signals
    bad_signals = [
        "bad", "terrible", "boring", "awful", "trash", "garbage", "poor",
        "blurry", "static", "nothing happens", "random", "nonsense", "not rugby",
        "irrelevant", "unrelated", "fail", "blank"
    ]
    is_bad = any(b in low for b in bad_signals)

    themes = _match(low, THEMES)
    formats = _match(low, FORMATS)
    tones = _match(low, TONES)
    contexts = _match(low, CONTEXTS)
    audios = _match(low, AUDIO_FORMATS)
    teams = _match(low, TEAM_CATEGORIES)
    duration = _duration(low)

    # Strict defaults if empty or bad
    if is_bad or not themes:
        if is_bad:
            themes = ["non rugby related"]
            tones = ["solemn"]
            formats = ["candid clip"]
            audios = ["none"]
        else:
            # Neutral / generic rugby default
            themes = ["rugby_skills"]
            tones = ["excitement"]
            formats = ["highlight"]
            audios = ["ambient"]

    if not formats:
        formats = ["highlight"]
    if not tones:
        tones = ["excitement"]
    if not contexts:
        contexts = ["match day"]
    if not audios:
        audios = ["ambient"]
    if not teams:
        teams = ["men"]

    # Infer people / players / teams
    people = []
    for standard_name, keywords in PEOPLE_MAP.items():
        if any(k in low for k in keywords):
            people.append(standard_name)
    if not people:
        people = ["all blacks" if page == "All Blacks" else page.lower()]

    # Infer brands / sponsors
    brands = []
    for standard_brand, keywords in BRAND_MAP.items():
        if any(k in low for k in keywords):
            brands.append(standard_brand)
    if not brands:
        brands = ["adidas"]


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
        "content_theme": themes,
        "content_themes": themes,  # backward compat
        "format_access": formats,
        "tone": tones,
        "tones": tones,            # backward compat
        "people": people,
        "brands": brands,
        "event": [page.lower()],
        "context": contexts,
        "overall_team": teams,
        "audio_format": audios,
        "cost": 0.0,
        "expected_rpm": 3.0,
        "expected_cpm": 5.0,
    }


def _page(low: str) -> str:
    for key, value in PAGES.items():
        if key in low:
            return value
    return "All Blacks"


def _platform(low: str) -> str:
    for key, value in PLATFORM_HINTS.items():
        if key in low:
            return value
    return "ALL"


def _match(low: str, mapping: Dict[str, List[str]]) -> List[str]:
    hits = []
    for field, keywords in mapping.items():
        if any(k in low for k in keywords):
            hits.append(field)
    return hits


def _duration(low: str) -> float:
    m = re.search(r"(\d+)\s*(?:sec|seconds|s)\b", low)
    if m:
        return float(max(5, min(int(m.group(1)), 120)))
    if re.search(r"\b(minute|min)\b", low):
        return 60.0
    return DURATION_DEFAULT


def _make_title(description: str) -> str:
    text = description.strip()
    if not text:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", text)
    hook = parts[0].strip().rstrip(".!?")
    if len(hook) > 80:
        hook = hook[:77] + "..."
    return hook
