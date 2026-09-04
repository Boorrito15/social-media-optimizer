"""Social Media Optimizer — Studio & Prediction Engine.

Revamped Architecture (Using models/ directly as specified in models/reference.md):
- Section 1: Video Description & Caption
- Section 2:
    - Part A: Target Platform (default none), Account/Channel (default none), Duration (default 20s)
    - Part B: Full description_json metadata (content_theme, format_access, people, brands, event, tone, context, overall_team, audio_format)
- Section 3: Output
    - Part A: 2x2 Bins / Quadrant (views_<i>, engagement_<j>)
    - Part B: Regressions (actual predicted views & engagement numbers from predict_lin)
    - Part C: Production Budget & Value / ROI Calculator with platform-specific CPM & Engagement rates
"""

from __future__ import annotations

import functools
import html
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import exclusively from models package
from models import predict_clas, predict_lin

# ---------------------------------------------------------------------------
# Streamlit Page Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Social Media Optimizer",
    page_icon="🏉",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Official Platform Brand SVG Logos
# ---------------------------------------------------------------------------
PLATFORM_LOGOS = {
    "FB": """<svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="vertical-align:middle; display:inline-block;"><path d="M24 12.073c0-6.627-5.373-12-12-12s-12 5.373-12 12c0 5.99 4.388 10.954 10.125 11.854v-8.385H7.078v-3.47h3.047V9.43c0-3.007 1.792-4.669 4.533-4.669 1.312 0 2.686.235 2.686.235v2.953H15.83c-1.491 0-1.956.925-1.956 1.874v2.25h3.328l-.532 3.47h-2.796v8.385C19.612 23.027 24 18.062 24 12.073z" fill="#1877F2"/></svg>""",
    "IG": """<svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="vertical-align:middle; display:inline-block;"><radialGradient id="ig-grad" cx="20%" cy="105%" r="120%"><stop offset="0%" stop-color="#fdf497"/><stop offset="5%" stop-color="#fdf497"/><stop offset="45%" stop-color="#fd5949"/><stop offset="60%" stop-color="#d6249f"/><stop offset="90%" stop-color="#285AEB"/></radialGradient><path d="M12 2.163c3.204 0 3.584.012 4.85.07 3.252.148 4.771 1.691 4.919 4.919.058 1.265.069 1.645.069 4.849 0 3.205-.012 3.584-.069 4.849-.149 3.225-1.664 4.771-4.919 4.919-1.266.058-1.644.07-4.85.07-3.204 0-3.584-.012-4.849-.07-3.26-.149-4.771-1.699-4.919-4.92-.058-1.265-.07-1.644-.07-4.849 0-3.204.013-3.583.07-4.849.149-3.227 1.664-4.771 4.919-4.919 1.266-.057 1.645-.069 4.849-.069zm0-2.163c-3.259 0-3.667.014-4.947.072-4.358.2-6.78 2.618-6.98 6.98-.059 1.281-.073 1.689-.073 4.948 0 3.259.014 3.668.072 4.948.2 4.358 2.618 6.78 6.98 6.98 1.281.058 1.689.072 4.948.072 3.259 0 3.668-.014 4.948-.072 4.354-.2 6.782-2.618 6.979-6.98.059-1.28.073-1.689.073-4.948 0-3.259-.014-3.667-.072-4.947-.196-4.354-2.617-6.78-6.979-6.98-1.281-.059-1.69-.073-4.949-.073zm0 5.838c-3.403 0-6.162 2.759-6.162 6.162s2.759 6.163 6.162 6.163 6.162-2.759 6.162-6.163c0-3.403-2.759-6.162-6.162-6.162zm0 10.162c-2.209 0-4-1.79-4-4 0-2.209 1.791-4 4-4s4 1.791 4 4c0 2.21-1.791 4-4 4zm6.406-11.845c-.796 0-1.441.645-1.441 1.44s.645 1.44 1.441 1.44c.795 0 1.439-.645 1.439-1.44s-.644-1.44-1.439-1.44z" fill="url(#ig-grad)"/></svg>""",
    "TT": """<svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="vertical-align:middle; display:inline-block;"><path d="M19.59 6.69a4.83 4.83 0 0 1-3.77-4.25V2h-3.45v13.67a2.89 2.89 0 0 1-2.88 2.89 2.89 2.89 0 0 1-2.89-2.89 2.89 2.89 0 0 1 2.89-2.89c.28 0 .54.04.79.1v-3.5a6.37 6.37 0 0 0-.79-.05A6.34 6.34 0 0 0 3 15.67a6.34 6.34 0 0 0 6.34 6.33 6.34 6.34 0 0 0 6.34-6.33V9.05c1.47 1.05 3.27 1.67 5.22 1.71V7.31c-.44 0-.88-.22-1.31-.62z" fill="#000000"/></svg>""",
    "YT": """<svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="vertical-align:middle; display:inline-block;"><path d="M23.498 6.186a3.016 3.016 0 0 0-2.122-2.136C19.505 3.545 12 3.545 12 3.545s-7.505 0-9.377.505A3.017 3.017 0 0 0 .502 6.186C0 8.07 0 12 0 12s0 3.93.502 5.814a3.016 3.016 0 0 0 2.122 2.136c1.871.505 9.376.505 9.376.505s7.505 0 9.377-.505a3.015 3.015 0 0 0 2.122-2.136C24 15.93 24 12 24 12s0-3.93-.502-5.814zM9.545 15.568V8.432L15.818 12l-6.273 3.568z" fill="#FF0000"/></svg>""",
}

# ---------------------------------------------------------------------------
# Valuation Economics Parameters (Per User Prompt)
# ---------------------------------------------------------------------------
# Views CPM (Per 1,000 views)
VIEWS_CPM = {
    "YT": 35.0,
    "IG": 17.5,
    "TT": 10.0,
    "FB": 10.0,
}

# Value per individual Engagement
ENGAGEMENT_VALUE = {
    "IG": 2.30,
    "FB": 2.10,
    "TT": 2.75,
    "YT": 0.75,
}

PLATFORM_NAMES = {
    "FB": "Facebook",
    "IG": "Instagram",
    "TT": "TikTok",
    "YT": "YouTube",
}

PAGE_OPTIONS = [
    "All Blacks",
    "Black Ferns",
    "NZ Sevens",
    "NZR",
    "ABXV",
    "Bunnings NPC",
    "Super Rugby",
]

THEME_OPTIONS = [
    "try",
    "celebration",
    "rugby_skills",
    "training",
    "challenges",
    "player story",
    "kick",
    "tackle",
    "haka",
    "rivalry",
    "run",
    "conversion",
    "non rugby related",
]

FORMAT_OPTIONS = [
    "highlight",
    "interview",
    "montage",
    "behind-the-scenes",
    "announcement",
    "candid clip",
    "archive",
    "reaction",
    "promotion",
    "ticket sales",
]

PEOPLE_OPTIONS = [
    "all blacks",
    "black ferns",
    "nz sevens",
    "all blacks xv",
    "beauden barrett",
    "ardie savea",
    "will jordan",
    "rieo ioane",
    "codie taylor",
    "scott barrett",
    "sam cane",
    "damian mckenzie",
    "caleb clarke",
    "jordie barrett",
    "cam roigard",
    "tyrel lomax",
    "tamaiti williams",
    "sevu reece",
    "tupou vaa'i",
    "asafo aumua",
    "wallabies",
    "springboks",
    "england rugby",
    "ireland rugby",
    "france rugby",
]

BRAND_OPTIONS = [
    "adidas",
    "ineos",
    "altrad",
    "tudor",
    "ford",
    "asahi",
    "smart rugby",
    "sky sport",
    "nib",
    "gatorade",
    "red bull",
    "aig",
    "barfoot & thompson",
]

EVENT_OPTIONS = [
    "rugby world cup",
    "rugby championship",
    "bledisloe cup",
    "autumn nations series",
    "super rugby pacific",
    "bunnings npc",
    "sevens series",
]

TONE_OPTIONS = [
    "excitement",
    "pride",
    "tension",
    "nostalgia",
    "humour",
    "wholesome",
    "solemn",
    "sadness",
    "lighthearted",
    "provocative",
]

CONTEXT_OPTIONS = [
    "match day",
    "stadium",
    "pre-match",
    "post-match",
    "gym",
    "changing room",
    "press conference",
    "announcement",
    "tour",
    "travel",
    "off-season",
    "squad naming",
    "jersey reveal",
    "award",
]

TEAM_OPTIONS = ["men", "women", "veterans", "maori", "youth"]
AUDIO_OPTIONS = ["ambient", "voice", "song", "none", "other"]

# ---------------------------------------------------------------------------
# Auto-Inference Keyword Dictionaries (Deterministic Keyword Extraction)
# ---------------------------------------------------------------------------
THEMES_DICT = {
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

FORMATS_DICT = {
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

TONES_DICT = {
    "excitement": ["spectacular", "unbelievable", "incredible", "amazing", "electric", "thrilling", "insane", "sprints", "erupts", "roaring", "crunching"],
    "pride": ["pride", "proud", "jersey", "all black", "black fern", "honour", "legacy", "tradition", "anthem"],
    "tension": ["tension", "tense", "nail-biter", "final minute", "pressure", "clutch", "last-minute", "edge of seat"],
    "nostalgia": ["nostalgia", "throwback", "memories", "legend", "historic", "reminiscing", "golden era"],
    "humour": ["funny", "joke", "banter", "humour", "humor", "laugh", "blooper", "prank", "hilarious"],
    "wholesome": ["wholesome", "heartwarming", "family", "kids", "fan", "signed", "smile", "kindness"],
    "solemn": ["solemn", "memorial", "tribute", "respect", "sadness", "farewell", "retiring", "injury", "loss"],
    "sadness": ["sad", "heartbreak", "defeat", "loss", "tears", "crying", "disappointment"],
    "lighthearted": ["lighthearted", "casual", "relaxed", "fun", "chill", "coffee", "hotel", "travel"],
    "provocative": ["provocative", "controversial", "referee", "card", "red card", "yellow card", "feud"],
}

CONTEXTS_DICT = {
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

AUDIO_FORMATS_DICT = {
    "voice": ["interview", "speaks", "talking", "speech", "huddle", "commentary", "commentator", "voice"],
    "song": ["song", "music", "beat", "anthem", "track"],
    "ambient": ["stadium", "crowd", "roaring", "cheering", "applause", "ambient", "cheers", "sound"],
    "none": ["quiet", "silent", "muted", "dark", "boring", "static", "nothing", "none"],
    "other": ["sound effects", "sfx", "other"],
}

TEAM_CATEGORIES_DICT = {
    "women": ["black fern", "women", "girls", "female", "farah palmer", "aupiki"],
    "men": ["all black", "men", "male", "boys", "npc", "super rugby"],
    "veterans": ["veteran", "legend", "retired", "classic ab", "former player"],
    "maori": ["maori", "all blacks maori", "tangata"],
    "youth": ["u20", "youth", "school", "junior", "u85"],
}

PEOPLE_MAP_DICT = {
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

BRAND_MAP_DICT = {
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

PAGES_DICT = {
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
    "super rugby": "Super Rugby",
    "srp": "Super Rugby",
}


def _match_keywords(text_low: str, mapping: Dict[str, List[str]], valid_options: List[str]) -> List[str]:
    hits = []
    for field, keywords in mapping.items():
        if field in valid_options and any(k in text_low for k in keywords):
            hits.append(field)
    return hits


def auto_infer_metadata(description: str, caption: str) -> Dict[str, Any]:
    """Auto-generate all metadata taxonomies from description and caption text."""
    combined = f"{description or ''} {caption or ''}".strip().lower()
    if not combined:
        return {}

    themes = _match_keywords(combined, THEMES_DICT, THEME_OPTIONS) or ["try", "celebration"]
    formats = _match_keywords(combined, FORMATS_DICT, FORMAT_OPTIONS) or ["highlight"]
    tones = _match_keywords(combined, TONES_DICT, TONE_OPTIONS) or ["excitement"]
    contexts = _match_keywords(combined, CONTEXTS_DICT, CONTEXT_OPTIONS) or ["stadium", "match day"]
    audios = _match_keywords(combined, AUDIO_FORMATS_DICT, AUDIO_OPTIONS) or ["ambient"]
    teams = _match_keywords(combined, TEAM_CATEGORIES_DICT, TEAM_OPTIONS) or ["men"]

    people = []
    for std_name, kws in PEOPLE_MAP_DICT.items():
        if std_name in PEOPLE_OPTIONS and any(k in combined for k in kws):
            people.append(std_name)
    if not people:
        people = ["all blacks"]

    brands = []
    for std_brand, kws in BRAND_MAP_DICT.items():
        if std_brand in BRAND_OPTIONS and any(k in combined for k in kws):
            brands.append(std_brand)

    # Inferred Page/Account
    detected_pages = []
    for key, value in PAGES_DICT.items():
        if key in combined and value in PAGE_OPTIONS:
            detected_pages.append(value)
            break

    # Inferred Duration
    duration = 20.0
    m = re.search(r"(\d+)\s*(?:sec|seconds|s)\b", combined)
    if m:
        duration = float(max(5, min(int(m.group(1)), 120)))
    elif re.search(r"\b(minute|min)\b", combined):
        duration = 60.0

    return {
        "content_theme": themes,
        "format_access": formats,
        "tone": tones,
        "context": contexts,
        "audio_format": audios,
        "overall_team": teams,
        "people": people,
        "brands": brands,
        "account_pages": detected_pages,
        "duration_seconds": duration,
    }

# ---------------------------------------------------------------------------
# Custom CSS: Light Minimalist Theme
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background-color: #FFFFFF !important;
    color: #0D0D0D !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

[data-testid="stHeader"], [data-testid="stToolbar"] {
    display: none !important;
}

.main .block-container {
    max-width: 1060px !important;
    padding: 2rem 1.5rem 5rem 1.5rem !important;
    margin: 0 auto !important;
}

/* Section Header Typography */
.section-header {
    margin-top: 1.5rem;
    margin-bottom: 1.25rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(0, 0, 0, 0.08);
}

.section-tag {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #555555;
    margin-bottom: 0.25rem;
}

.section-title {
    font-size: 1.5rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: #0D0D0D;
    line-height: 1.2;
}

.section-subtitle {
    font-size: 0.88rem;
    color: #555555;
    margin-top: 0.25rem;
}

/* Clean Card Wrapper with subtle hover elevation */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: #FFFFFF !important;
    border: 1px solid rgba(0, 0, 0, 0.08) !important;
    border-radius: 12px !important;
    padding: 1.25rem 1.4rem !important;
    margin-bottom: 1rem !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.02) !important;
    transition: box-shadow 0.2s ease, border-color 0.2s ease !important;
}

[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: rgba(0, 0, 0, 0.14) !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.04) !important;
}

/* Tabs Styling */
button[data-baseweb="tab"] {
    background-color: #F8FAFC !important;
    border: 1px solid rgba(0, 0, 0, 0.08) !important;
    border-radius: 8px 8px 0 0 !important;
    padding: 0.65rem 1.35rem !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    color: #64748B !important;
    margin-right: 0.35rem !important;
    transition: all 0.15s ease !important;
}

button[data-baseweb="tab"]:hover {
    color: #0F172A !important;
    background-color: #F1F5F9 !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    background-color: #FFFFFF !important;
    border-color: rgba(0, 0, 0, 0.15) !important;
    border-bottom: 2px solid #0D0D0D !important;
    color: #0D0D0D !important;
    font-weight: 700 !important;
}

div[data-baseweb="tab-list"] {
    gap: 4px !important;
    border-bottom: 1px solid rgba(0, 0, 0, 0.1) !important;
    margin-bottom: 1rem !important;
}

/* Metric Display Cards */
.metric-box {
    background: #FFFFFF;
    border: 1px solid rgba(0, 0, 0, 0.08);
    border-radius: 10px;
    padding: 1.2rem;
    text-align: center;
    box-shadow: 0 1px 2px rgba(0,0,0,0.02);
}

.metric-label {
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    color: #64748B;
    margin-bottom: 0.25rem;
    letter-spacing: 0.04em;
}

.metric-value {
    font-size: 2.1rem;
    font-weight: 800;
    color: #0D0D0D;
    letter-spacing: -0.02em;
}

/* Primary Action Button with Gradient & Pulse */
div.stButton > button {
    border-radius: 8px !important;
    font-weight: 600 !important;
    background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%) !important;
    color: #FFFFFF !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    padding: 0.75rem 1.6rem !important;
    font-size: 0.95rem !important;
    box-shadow: 0 4px 12px rgba(15, 23, 42, 0.15) !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
}

div.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 16px rgba(15, 23, 42, 0.25) !important;
}

div.stButton > button:active {
    transform: translateY(0px) !important;
}

/* Pulse & Shimmer Animations */
@keyframes pulse-dim {
    0% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.55; transform: scale(0.995); }
    100% { opacity: 1; transform: scale(1); }
}

@keyframes shimmer-glow {
    0% { background-position: -200% 0; }
    100% { background-position: 200% 0; }
}

.pulsing-ticker {
    animation: pulse-dim 1.4s ease-in-out infinite;
    background: linear-gradient(90deg, #F8FAFC 0%, #EFF6FF 50%, #F8FAFC 100%);
    background-size: 200% 100%;
    border: 1px solid #BFDBFE;
    border-radius: 10px;
    padding: 0.9rem 1.25rem;
    display: flex;
    align-items: center;
    gap: 0.85rem;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.08);
}

.spinner-dot {
    width: 12px;
    height: 12px;
    background-color: #2563EB;
    border-radius: 50%;
    animation: pulse-dim 0.8s ease-in-out infinite alternate;
    box-shadow: 0 0 10px rgba(37, 99, 235, 0.6);
}

/* Status Widget Modern Look */
[data-testid="stStatusWidget"] {
    background: #FFFFFF !important;
    border: 1px solid rgba(0, 0, 0, 0.1) !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04) !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Top Header Banner
# ---------------------------------------------------------------------------
st.markdown(
    """
<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:1.5rem; padding-bottom:0.75rem; border-bottom:1px solid rgba(0,0,0,0.1);">
    <div style="font-size:1.25rem; font-weight:700; color:#0D0D0D;">🏉 Social Media Optimizer <span style="font-size:0.75rem; background:#0D0D0D; color:#FFFFFF; padding:0.15rem 0.45rem; border-radius:4px; font-weight:600; margin-left:0.4rem;">MODEL STUDIO</span></div>
    <div style="font-size:0.82rem; color:#555555;">Models Package Engine · reference.md Compliant</div>
</div>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Helper: Build DataFrame Shaped for models.predict_* (from reference.md)
# ---------------------------------------------------------------------------
def build_model_dataframe(
    caption: str,
    play_by_play: str,
    platform: str,
    page: str,
    duration_seconds: float,
    content_theme: List[str],
    format_access: List[str],
    people: List[str],
    brands: List[str],
    event: List[str],
    tone: List[str],
    context: List[str],
    overall_team: List[str],
    audio_format: List[str],
) -> pd.DataFrame:
    """Build DataFrame formatted precisely as documented in models/reference.md."""
    desc_json_dict = {
        "play_by_play": play_by_play.strip() if play_by_play else "",
        "content_theme": content_theme,
        "format_access": format_access,
        "people": people,
        "brands": brands,
        "event": event,
        "tone": tone,
        "context": context,
        "overall_team": overall_team,
        "audio_format": audio_format,
    }

    row = {
        "campaign": "Organic | Website",
        "year": 2025,
        "page": page if page else "ABXV",
        "platform": platform if platform else "FB",
        "media_type": "Short Video",
        "category_l0": "No Hashtag",
        "category_l1": "No Hashtag",
        "category_l2": "No Hashtag",
        "url": "https://example.com/dummy-reel",
        "content": caption.strip() if caption else "",
        "cost_nzd": None,
        "views": 123,
        "engagement": 456,
        "hours": 64.0,
        "description_json": json.dumps(desc_json_dict),
        "duration_seconds": float(duration_seconds),
    }

    return pd.DataFrame([row])


# ---------------------------------------------------------------------------
# Helper: Recover the classifier's high/low bin thresholds
# ---------------------------------------------------------------------------
# predict_clas bins (views_0/1, engagement_0/1) are defined at train time by
# _iqr_binned() in models/clas.py:
#     lo = quantile(0.1); hi = quantile(0.9); iqr = hi - lo
#     edges = linspace(lo - 1.5*iqr, hi + 1.5*iqr, n_bins + 1)   # n_bins = 2
# so the bin-0/bin-1 split is the midpoint (q10 + q90) / 2 computed on the
# IQR-sliced training set. The quadrant split lines must use these same
# thresholds so the plotted regression point and the classification tier share
# one yardstick.
@functools.lru_cache(maxsize=1)
def classifier_bin_thresholds() -> Dict[str, float]:
    """Return ``{"views": <split>, "eng": <split>}`` matching ``predict_clas`` bins.

    Recomputed from the same train-time split the classifier used, so it stays
    correct across retrains. Falls back to hardcoded values derived from the
    current training data if ``processed.csv`` is unavailable.
    """
    fallback = {"views": 217970.0, "eng": 9780.0}
    try:
        from models.config import PipelineConfig, processed_csv_path
        from models.shared.filter import iqr_slice, validity_mask
        from models.shared.split import split_indices

        cfg = PipelineConfig()
        df = pd.read_csv(processed_csv_path())
        df = validity_mask(df)
        df = iqr_slice(df, cfg.bottom_iqr_percentile, cfg.top_iqr_percentile)
        df = df.reset_index(drop=True)
        idx_tr, _ = split_indices(
            len(df), cfg.test_size, cfg.train_test_random_seed
        )
        tr = df.iloc[idx_tr]
        out: Dict[str, float] = {}
        for col, key in (("views", "views"), ("engagement", "eng")):
            lo = float(tr[col].quantile(cfg.bottom_iqr_percentile))
            hi = float(tr[col].quantile(cfg.top_iqr_percentile))
            out[key] = (lo + hi) / 2.0
        return out
    except Exception:
        return fallback


def _tier_is_high(bin_label: str) -> bool:
    """True when a ``views_<i>`` / ``engagement_<j>`` label means the high bin."""
    return str(bin_label).endswith("1") or ("high" in str(bin_label).lower())


def format_tier_label(bin_label: str, metric_type: str = "views") -> str:
    """Convert raw machine bin labels to clean human-readable titles.
    
    Examples:
        'views_0' -> 'Low Views'
        'views_1' -> 'High Views'
        'engagement_0' -> 'Low Engagement'
        'engagement_1' -> 'High Engagement'
    """
    is_hi = _tier_is_high(bin_label)
    if metric_type.lower().startswith("view"):
        return "High Views" if is_hi else "Low Views"
    else:
        return "High Engagement" if is_hi else "Low Engagement"


def format_dual_tier(views_bin: str, eng_bin: str) -> str:
    """Convert a pair of bin labels to a unified clean label.
    
    Example: ('views_1', 'engagement_1') -> 'High Views · High Engagement'
    """
    v_text = format_tier_label(views_bin, "views")
    e_text = format_tier_label(eng_bin, "engagement")
    return f"{v_text} · {e_text}"


def _tier_center(
    views_high: bool,
    eng_high: bool,
    split_v: float,
    split_e: float,
    v_max: float,
    e_max: float,
) -> tuple[float, float]:
    """Return the geometric centre of the 2x2 cell implied by the classification tier.

    The quadrant is now a purely categorical map of the SVC tier, so each
    platform's marker sits at the centre of its predicted cell instead of at
    the raw regression coordinates.
    """
    x = (split_v / 2.0) if not views_high else (split_v + v_max * 1.25) / 2.0
    y = (split_e / 2.0) if not eng_high else (split_e + e_max * 1.25) / 2.0
    return x, y


def _cell_spread_offsets(
    platforms: List[str],
    clas_by_platform: Dict[str, Dict[str, Any]],
) -> Dict[str, tuple[float, float]]:
    """Deterministic unit-fraction offsets so platforms sharing a classification
    cell fan out in a staggered column instead of stacking at the same point."""
    groups: Dict[tuple[bool, bool], List[str]] = {}
    for p in platforms:
        c = clas_by_platform.get(p, {})
        key = (
            _tier_is_high(c.get("views", "views_0")),
            _tier_is_high(c.get("engagement", "engagement_0")),
        )
        groups.setdefault(key, []).append(p)

    offsets: Dict[str, tuple[float, float]] = {}
    for members in groups.values():
        n = len(members)
        if n == 1:
            offsets[members[0]] = (0.0, 0.0)
            continue
        for i, m in enumerate(members):
            t = -0.22 + 0.44 * (i / (n - 1))  # even vertical stagger
            x_shift = 0.035 if i % 2 == 0 else -0.035  # alternate horizontally
            offsets[m] = (x_shift, t)
    return offsets


# ---------------------------------------------------------------------------
# SECTION 1: CONTENT & CREATIVE
# ---------------------------------------------------------------------------
st.markdown(
    """
<div class="section-header">
    <div class="section-tag">Section 1</div>
    <div class="section-title">Creative Content</div>
    <div class="section-subtitle">Define the video action narrative and post caption. Metadata fields below auto-populate on edit.</div>
</div>
""",
    unsafe_allow_html=True,
)

# Initialize Session State Defaults
if "init_defaults" not in st.session_state:
    st.session_state.init_defaults = True
    st.session_state.video_description = "A male rugby player sprints downfield, breaks a tackle and scores a try under the posts."
    st.session_state.post_caption = "@jacobkneepkens crosses the chalk 🤙 #AllBlacks #TryTime"
    st.session_state.last_inferred_text = ""
    st.session_state.meta_theme = ["try", "celebration"]
    st.session_state.meta_format = ["highlight"]
    st.session_state.meta_people = ["all blacks"]
    st.session_state.meta_brands = []
    st.session_state.meta_event = []
    st.session_state.meta_tone = ["excitement"]
    st.session_state.meta_context = ["stadium", "match day"]
    st.session_state.meta_team = ["men"]
    st.session_state.meta_audio = ["ambient"]
    st.session_state.meta_pages = []
    st.session_state.meta_duration = 20.0

# Callback function when description or caption changes (triggers on blur / enter)
def on_text_changed():
    desc = st.session_state.get("video_description_input", "")
    cap = st.session_state.get("post_caption_input", "")
    inferred = auto_infer_metadata(desc, cap)
    if inferred:
        st.session_state.meta_theme = inferred.get("content_theme", ["try", "celebration"])
        st.session_state.meta_format = inferred.get("format_access", ["highlight"])
        st.session_state.meta_people = inferred.get("people", ["all blacks"])
        st.session_state.meta_brands = inferred.get("brands", [])
        st.session_state.meta_tone = inferred.get("tone", ["excitement"])
        st.session_state.meta_context = inferred.get("context", ["stadium", "match day"])
        st.session_state.meta_team = inferred.get("overall_team", ["men"])
        st.session_state.meta_audio = inferred.get("audio_format", ["ambient"])
        if inferred.get("account_pages"):
            st.session_state.meta_pages = inferred.get("account_pages")
        if inferred.get("duration_seconds"):
            st.session_state.meta_duration = inferred.get("duration_seconds")

col_s1_a, col_s1_b = st.columns(2, gap="medium")

with col_s1_a:
    video_description = st.text_area(
        "Video Description (Play-by-play)",
        value=st.session_state.get("video_description", "A male rugby player sprints downfield, breaks a tackle and scores a try under the posts."),
        key="video_description_input",
        height=130,
        on_change=on_text_changed,
        help="Detailed play-by-play visual and action sequence (maps to description_json.play_by_play).",
    )

with col_s1_b:
    post_caption = st.text_area(
        "Caption (Post Copy)",
        value=st.session_state.get("post_caption", "@jacobkneepkens crosses the chalk 🤙 #AllBlacks #TryTime"),
        key="post_caption_input",
        height=130,
        on_change=on_text_changed,
        help="Social media caption, hashtags, and mentions (maps to DataFrame content).",
    )

# ---------------------------------------------------------------------------
# SECTION 2: PARAMETERS (Part A & Part B)
# ---------------------------------------------------------------------------
st.markdown(
    """
<div class="section-header">
    <div class="section-tag">Section 2</div>
    <div class="section-title">Model Parameters & Taxonomy</div>
    <div class="section-subtitle">Configure target channel distribution and description_json metadata.</div>
</div>
""",
    unsafe_allow_html=True,
)

# Part A: Target Platform, Account/Channel, Duration
with st.container(border=True):
    st.markdown('<div style="font-size:0.92rem; font-weight:700; color:#0D0D0D; margin-bottom:0.75rem;">Part A: Target Distribution & Duration</div>', unsafe_allow_html=True)
    col_pA1, col_pA2, col_pA3 = st.columns([1.2, 1.2, 1], gap="medium")

    with col_pA1:
        selected_platforms = st.multiselect(
            "Target Platform(s)",
            options=["FB", "IG", "TT", "YT"],
            default=[],  # Default to nothing selected as requested
            format_func=lambda x: f"{x} - {PLATFORM_NAMES.get(x, x)}",
            help="Select one or more platforms to evaluate (defaults to nothing selected).",
        )

    with col_pA2:
        selected_pages = st.multiselect(
            "Account / Channel",
            options=PAGE_OPTIONS,
            default=st.session_state.get("meta_pages", []),
            help="Target team or brand channel (defaults to nothing selected).",
        )

    with col_pA3:
        duration_sec = st.number_input(
            "Duration (seconds)",
            min_value=1.0,
            max_value=300.0,
            value=float(st.session_state.get("meta_duration", 20.0)),
            step=1.0,
            help="Runtime of the video clip in seconds (default: 20).",
        )

# Part B: description_json fields
with st.container(border=True):
    st.markdown(
        """
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.75rem;">
            <div style="font-size:0.92rem; font-weight:700; color:#0D0D0D;">Part B: description_json Fields</div>
            <div style="font-size:0.75rem; color:#6B7280;">⚡ Auto-populated from content above (editable)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_pB1, col_pB2, col_pB3 = st.columns(3, gap="medium")

    with col_pB1:
        sel_theme = st.multiselect(
            "Content Theme",
            options=THEME_OPTIONS,
            default=[x for x in st.session_state.get("meta_theme", ["try", "celebration"]) if x in THEME_OPTIONS],
        )
        sel_format = st.multiselect(
            "Format Access",
            options=FORMAT_OPTIONS,
            default=[x for x in st.session_state.get("meta_format", ["highlight"]) if x in FORMAT_OPTIONS],
        )
        sel_people = st.multiselect(
            "People / Entities",
            options=PEOPLE_OPTIONS,
            default=[x for x in st.session_state.get("meta_people", ["all blacks"]) if x in PEOPLE_OPTIONS],
        )

    with col_pB2:
        sel_brands = st.multiselect(
            "Brands / Sponsors",
            options=BRAND_OPTIONS,
            default=[x for x in st.session_state.get("meta_brands", []) if x in BRAND_OPTIONS],
        )
        sel_event = st.multiselect(
            "Event / Competition",
            options=EVENT_OPTIONS,
            default=st.session_state.get("meta_event", []),
        )
        sel_tone = st.multiselect(
            "Tone",
            options=TONE_OPTIONS,
            default=[x for x in st.session_state.get("meta_tone", ["excitement"]) if x in TONE_OPTIONS],
        )

    with col_pB3:
        sel_context = st.multiselect(
            "Context / Setting",
            options=CONTEXT_OPTIONS,
            default=[x for x in st.session_state.get("meta_context", ["stadium", "match day"]) if x in CONTEXT_OPTIONS],
        )
        sel_team = st.multiselect(
            "Overall Team",
            options=TEAM_OPTIONS,
            default=[x for x in st.session_state.get("meta_team", ["men"]) if x in TEAM_OPTIONS],
        )
        sel_audio = st.multiselect(
            "Audio Format",
            options=AUDIO_OPTIONS,
            default=[x for x in st.session_state.get("meta_audio", ["ambient"]) if x in AUDIO_OPTIONS],
        )

# ---------------------------------------------------------------------------
# Storage / Persistence Helper
# ---------------------------------------------------------------------------
HISTORY_FILE = REPO_ROOT / "prediction_history.json"

def save_prediction_record(record: Dict[str, Any]) -> None:
    """Append a prediction run to a local JSON history file."""
    history = []
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            history = []
    history.append(record)
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        st.warning(f"Could not write history to disk: {e}")


def load_prediction_history() -> List[Dict[str, Any]]:
    """Load historical prediction records."""
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


# Predict Trigger Button
st.markdown("<div style='margin: 1.25rem 0;'></div>", unsafe_allow_html=True)
run_btn = st.button("🚀 Run Model Inference", use_container_width=True, type="primary")

# ---------------------------------------------------------------------------
# Inference Execution & Results
# ---------------------------------------------------------------------------
# Determine active platforms and page for evaluation
eval_platforms = selected_platforms if len(selected_platforms) > 0 else ["FB", "IG", "TT", "YT"]
eval_page = selected_pages[0] if len(selected_pages) > 0 else "All Blacks"

if "last_results" not in st.session_state:
    st.session_state.last_results = None

# If user clicked the button or results exist in session state
if run_btn:
    with st.status("🎬 Processing Video Concept Through AI Engine...", expanded=True) as status:
        prog_bar = st.progress(5)
        info_ticker = st.empty()
        
        def show_ticker(icon: str, title: str, subtitle: str):
            info_ticker.markdown(
                f"""
                <div class="pulsing-ticker">
                    <div class="spinner-dot"></div>
                    <div>
                        <div style="font-size:0.92rem; font-weight:700; color:#1E40AF;">{icon} {title}</div>
                        <div style="font-size:0.8rem; color:#475569; margin-top:0.15rem;">{subtitle}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # 1. Feature Extraction & Embedding
        show_ticker("🔍", "Step 1/3: Parsing Play-by-Play & NLP Semantic Tokens", "Extracting multi-hot action keywords, player entities, and KMeans cluster embeddings...")
        prog_bar.progress(18)
        time.sleep(0.4)

        results_by_platform: Dict[str, Dict[str, Any]] = {}
        total_p = len(eval_platforms)

        # 2. Multi-Platform SVR & SVC Evaluation
        for idx, p_code in enumerate(eval_platforms):
            p_name_cur = PLATFORM_NAMES.get(p_code, p_code)
            pct = 20 + int((idx + 1) / total_p * 60)
            prog_bar.progress(pct)
            
            show_ticker(
                "⚡",
                f"Step 2/3: Evaluating {p_name_cur} ({p_code}) Models",
                f"Running Support Vector Regressor (SVR) & Polynomial SVC Classifier for {p_name_cur}..."
            )
            
            df_single = build_model_dataframe(
                caption=post_caption,
                play_by_play=video_description,
                platform=p_code,
                page=eval_page,
                duration_seconds=duration_sec,
                content_theme=sel_theme,
                format_access=sel_format,
                people=sel_people,
                brands=sel_brands,
                event=sel_event,
                tone=sel_tone,
                context=sel_context,
                overall_team=sel_team,
                audio_format=sel_audio,
            )

            try:
                lin_res = predict_lin(df_single)[0]
                clas_res = predict_clas(df_single)[0]
                results_by_platform[p_code] = {
                    "lin": lin_res,
                    "clas": clas_res,
                }
            except Exception as e:
                st.error(f"Inference error for {p_code}: {e}")
                raise e
            time.sleep(0.35)

        # 3. Final Synthesis
        prog_bar.progress(95)
        show_ticker("📊", "Step 3/3: Synthesizing Cross-Platform Value & ROI", "Calculating CPM valuation, net media margins, and empirical matrix placements...")
        time.sleep(0.35)
        
        prog_bar.progress(100)
        info_ticker.markdown(
            """
            <div style="background:#F0FDF4; border:1px solid #86EFAC; border-radius:10px; padding:0.9rem 1.25rem; display:flex; align-items:center; gap:0.75rem;">
                <div style="font-size:1.15rem;">✨</div>
                <div>
                    <div style="font-size:0.92rem; font-weight:700; color:#15803D;">Assessment Complete!</div>
                    <div style="font-size:0.8rem; color:#166534;">Strategic matrix, performance predictions, and budget calculator generated below.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        status.update(label="✅ Assessment Complete!", state="complete", expanded=False)

        # Save to session and disk
        st.session_state.last_results = {
            "results": results_by_platform,
            "platforms": eval_platforms,
            "page": eval_page,
            "caption": post_caption,
            "video_description": video_description,
            "duration": duration_sec,
        }

        # Save to persistent history file
        import datetime
        record = {
            "timestamp": datetime.datetime.now().isoformat(),
            "video_description": video_description,
            "caption": post_caption,
            "account_page": eval_page,
            "duration_seconds": duration_sec,
            "metadata": {
                "content_theme": sel_theme,
                "format_access": sel_format,
                "people": sel_people,
                "brands": sel_brands,
                "event": sel_event,
                "tone": sel_tone,
                "context": sel_context,
                "overall_team": sel_team,
                "audio_format": sel_audio,
            },
            "evaluations": {
                p: {
                    "projected_views": results_by_platform[p]["lin"].get("views"),
                    "projected_engagement": results_by_platform[p]["lin"].get("engagement"),
                    "quadrant_views": results_by_platform[p]["clas"].get("views"),
                    "quadrant_engagement": results_by_platform[p]["clas"].get("engagement"),
                    "quadrant_tier": format_dual_tier(
                        results_by_platform[p]["clas"].get("views", "views_0"),
                        results_by_platform[p]["clas"].get("engagement", "engagement_0"),
                    ),
                }
                for p in results_by_platform
            }
        }
        save_prediction_record(record)
        prog_bar.progress(100)
        status.update(label="✨ Assessment Complete! Optimized insights generated below.", state="complete", expanded=False)

# ---------------------------------------------------------------------------
# SECTION 3: OUTPUT (Part A: 2x2 Quadrant, Part B: Regressions, Part C: Calculator)
# ---------------------------------------------------------------------------
st.markdown(
    """
<div class="section-header">
    <div class="section-tag">Section 3</div>
    <div class="section-title">Model Output & Performance Insights</div>
    <div class="section-subtitle">Quadrant classification, regression estimates, and production budget calculator.</div>
</div>
""",
    unsafe_allow_html=True,
)

if not st.session_state.last_results:
    st.info("👈 Configure your creative content and parameters above, then click **'🚀 Run Model Inference'** to generate predictions.")
else:
    active_run = st.session_state.last_results
    results_by_platform = active_run.get("results", {})
    # Filter to platforms successfully evaluated
    eval_platforms = [p for p in active_run.get("platforms", []) if p in results_by_platform]
    eval_page = active_run.get("page", "All Blacks")

    if not eval_platforms:
        st.error("No platform evaluations were successfully generated. Please check model inputs and try again.")
    else:
        # Build tabs: All Platforms master comparison + individual platform tabs
        all_tab_labels = ["🌐 All Platforms (Overview)"] + [f"{PLATFORM_NAMES.get(p, p)} ({p})" for p in eval_platforms]
        tabs = st.tabs(all_tab_labels)

    # -----------------------------------------------------------------------
    # TAB 0: ALL PLATFORMS MASTER OVERVIEW
    # -----------------------------------------------------------------------
    with tabs[0]:
        st.markdown(
            f"""
            <div style="background:#FFFFFF; border:1px solid rgba(0,0,0,0.1); border-radius:12px; padding:1.2rem 1.4rem; margin-bottom:1.25rem; display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <div style="font-size:1.2rem; font-weight:700; color:#0D0D0D;">Cross-Platform Strategic Comparison</div>
                    <div style="font-size:0.82rem; color:#555555;">Target Account: <b style="color:#0D0D0D;">{eval_page}</b> · Duration: <b style="color:#0D0D0D;">{duration_sec:.0f}s</b> · Evaluating <b>{len(eval_platforms)} Platforms</b></div>
                </div>
                <div style="display:flex; gap:0.5rem; align-items:center;">
                    {' '.join([f'<span style="display:inline-block; padding:0.25rem 0.5rem; background:#F1F5F9; border-radius:6px; font-size:0.75rem; font-weight:700;">{p}</span>' for p in eval_platforms])}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Cross-platform Master Quadrant
        with st.container(border=True):
            st.markdown(
                """
                <div style="margin-bottom:0.6rem;">
                    <div style="font-size:0.95rem; font-weight:700; color:#0D0D0D;">Part A: All-Platform Comparative Performance Matrix</div>
                    <div style="font-size:0.8rem; color:#6B7280;">Simultaneous categorical placement of all evaluated platforms into their predicted classification cells.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            fig_all = go.Figure()

            # Global boundaries based on all predictions
            all_v = [max(0.0, float(results_by_platform[p]["lin"].get("views", 0.0))) for p in eval_platforms]
            all_e = [max(0.0, float(results_by_platform[p]["lin"].get("engagement", 0.0))) for p in eval_platforms]
            
            # Quadrant split lines = predict_clas high/low bin thresholds
            _thr_all = classifier_bin_thresholds()
            ref_v = _thr_all["views"]
            ref_e = _thr_all["eng"]
            max_v_all = max(max(all_v) * 1.3, ref_v * 2.2, 50000)
            max_e_all = max(max(all_e) * 1.3, ref_e * 2.2, 2500)

            # Quadrant Zones
            fig_all.add_shape(type="rect", x0=0, x1=ref_v, y0=0, y1=ref_e, fillcolor="rgba(244, 63, 94, 0.035)", line=dict(width=0))
            fig_all.add_shape(type="rect", x0=0, x1=ref_v, y0=ref_e, y1=max_e_all * 1.25, fillcolor="rgba(245, 158, 11, 0.045)", line=dict(width=0))
            fig_all.add_shape(type="rect", x0=ref_v, x1=max_v_all * 1.25, y0=0, y1=ref_e, fillcolor="rgba(59, 130, 246, 0.045)", line=dict(width=0))
            fig_all.add_shape(type="rect", x0=ref_v, x1=max_v_all * 1.25, y0=ref_e, y1=max_e_all * 1.25, fillcolor="rgba(16, 185, 129, 0.065)", line=dict(width=0))

            # Reference Lines
            fig_all.add_vline(x=ref_v, line=dict(color="rgba(15, 23, 42, 0.25)", width=1.5, dash="dot"))
            fig_all.add_hline(y=ref_e, line=dict(color="rgba(15, 23, 42, 0.25)", width=1.5, dash="dot"))

            # Corner Annotations
            fig_all.add_annotation(x=ref_v * 0.48, y=max_e_all * 1.1, text="<b>NICHE / DISCUSSION</b><br><span style='font-size:10px; color:#854D0E;'>Low Views · High Engagement</span>", showarrow=False, font=dict(family="Inter, sans-serif", size=10, color="#A16207"))
            fig_all.add_annotation(x=ref_v + (max_v_all - ref_v) * 0.52, y=max_e_all * 1.1, text="<b>OPTIMAL VIRAL HIT</b><br><span style='font-size:10px; color:#14532D;'>High Views · High Engagement</span>", showarrow=False, font=dict(family="Inter, sans-serif", size=10, color="#15803D"))
            fig_all.add_annotation(x=ref_v * 0.48, y=ref_e * 0.16, text="<b>LOW SIGNAL</b><br><span style='font-size:10px; color:#881337;'>Low Views · Low Engagement</span>", showarrow=False, font=dict(family="Inter, sans-serif", size=10, color="#BE123C"))
            fig_all.add_annotation(x=ref_v + (max_v_all - ref_v) * 0.52, y=ref_e * 0.16, text="<b>BROAD AWARENESS</b><br><span style='font-size:10px; color:#1E3A8A;'>High Views · Low Engagement</span>", showarrow=False, font=dict(family="Inter, sans-serif", size=10, color="#1D4ED8"))

            PLATFORM_PALETTE = {
                "FB": "#1877F2",
                "IG": "#E1306C",
                "TT": "#000000",
                "YT": "#FF0000",
            }

            # Plot every evaluated platform at its classification cell centre
            _clas_map_all = {
                p: results_by_platform[p]["clas"] for p in eval_platforms
            }
            _spread_all = _cell_spread_offsets(eval_platforms, _clas_map_all)
            for p in eval_platforms:
                p_lin = results_by_platform[p]["lin"]
                p_clas = results_by_platform[p]["clas"]
                p_v = max(0.0, float(p_lin.get("views", 0.0)))
                p_e = max(0.0, float(p_lin.get("engagement", 0.0)))
                _v_hi = _tier_is_high(p_clas.get("views", "views_0"))
                _e_hi = _tier_is_high(p_clas.get("engagement", "engagement_0"))
                p_x, p_y = _tier_center(
                    _v_hi, _e_hi, ref_v, ref_e, max_v_all, max_e_all,
                )
                _dx, _dy = _spread_all[p]
                p_x += _dx * (ref_v if not _v_hi else max_v_all * 1.25 - ref_v)
                p_y += _dy * (ref_e if not _e_hi else max_e_all * 1.25 - ref_e)
                p_col = PLATFORM_PALETTE.get(p, "#2563EB")
                clean_tier = format_dual_tier(p_clas.get("views", "views_0"), p_clas.get("engagement", "engagement_0"))

                # Halo ring
                fig_all.add_trace(
                    go.Scatter(
                        x=[p_x],
                        y=[p_y],
                        mode="markers",
                        marker=dict(size=32, color=p_col, opacity=0.2),
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )

                # Solid point
                fig_all.add_trace(
                    go.Scatter(
                        x=[p_x],
                        y=[p_y],
                        mode="markers+text",
                        marker=dict(size=18, color=p_col, symbol="circle", line=dict(color="#FFFFFF", width=2.5)),
                        text=[f"<b>{PLATFORM_NAMES.get(p, p)}</b>"],
                        textposition="bottom center",
                        textfont=dict(size=10.5, color="#0F172A", family="Inter, sans-serif"),
                        name=PLATFORM_NAMES.get(p, p),
                        showlegend=False,
                        hoverinfo="text",
                        hovertext=(
                            f"<b>{PLATFORM_NAMES.get(p, p)} ({p})</b><br>"
                            f"Predicted Tier: <b>{clean_tier}</b><br>"
                            f"Regression: {p_v:,.0f} views · {p_e:,.0f} engagements"
                        ),
                    )
                )

            fig_all.update_layout(
                height=420,
                margin=dict(l=50, r=30, t=20, b=45),
                paper_bgcolor="#FFFFFF",
                plot_bgcolor="#FAFAFC",
                hovermode=False,
                xaxis=dict(
                    title="<b>Views Tier (Low Views ⟵ Split ⟶ High Views)</b>",
                    title_font=dict(size=11, color="#64748B", family="Inter, sans-serif"),
                    range=[0, max_v_all * 1.15],
                    showgrid=False,
                    zeroline=False,
                    fixedrange=True,
                    tickformat=",.0f",
                    tickfont=dict(size=10, color="#64748B"),
                ),
                yaxis=dict(
                    title="<b>Engagement Tier (Low Engagement ⟵ Split ⟶ High Engagement)</b>",
                    title_font=dict(size=11, color="#64748B", family="Inter, sans-serif"),
                    range=[0, max_e_all * 1.15],
                    showgrid=False,
                    zeroline=False,
                    fixedrange=True,
                    tickformat=",.0f",
                    tickfont=dict(size=10, color="#64748B"),
                ),
            )

            st.plotly_chart(fig_all, use_container_width=True, config={"displayModeBar": False, "staticPlot": True})

        # Cross-platform Comparative Table (Parts B & C Overview)
        with st.container(border=True):
            st.markdown('<div style="font-size:0.95rem; font-weight:700; color:#0D0D0D; margin-bottom:0.75rem;">Platform Comparison & Value Summary ($500 Standard Budget)</div>', unsafe_allow_html=True)
            
            comp_rows = []
            for p in eval_platforms:
                p_lin = results_by_platform[p]["lin"]
                p_clas = results_by_platform[p]["clas"]
                p_v = max(0.0, float(p_lin.get("views", 0.0)))
                p_e = max(0.0, float(p_lin.get("engagement", 0.0)))
                p_cpm = VIEWS_CPM.get(p, 10.0)
                p_eng_rate = ENGAGEMENT_VALUE.get(p, 2.0)
                gross_val = ((p_v / 1000.0) * p_cpm) + (p_e * p_eng_rate)
                net_val = gross_val - 500.0
                roi_val = ((gross_val - 500.0) / 500.0 * 100.0)
                clean_tier = format_dual_tier(p_clas.get("views", "views_0"), p_clas.get("engagement", "engagement_0"))

                comp_rows.append({
                    "Platform": f"{PLATFORM_NAMES.get(p, p)} ({p})",
                    "Classification Tier": clean_tier,
                    "Predicted Views": f"{p_v:,.0f}",
                    "Predicted Engagements": f"{p_e:,.0f}",
                    "Est. Media Value": f"${gross_val:,.2f}",
                    "Net Value ($500 Spend)": f"${net_val:,.2f}",
                    "ROI (%)": f"{'+' if roi_val >= 0 else ''}{roi_val:.1f}%",
                })
            
            st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

    # -----------------------------------------------------------------------
    # TABS 1..N: INDIVIDUAL PLATFORM DEEP-DIVES
    # -----------------------------------------------------------------------
    for i, p_code in enumerate(eval_platforms):
        with tabs[i + 1]:
            p_data = results_by_platform[p_code]
            lin = p_data["lin"]
            clas = p_data["clas"]

            pred_views = max(0.0, float(lin.get("views", 0.0)))
            pred_eng = max(0.0, float(lin.get("engagement", 0.0)))
            views_bin = clas.get("views", "views_0")
            eng_bin = clas.get("engagement", "engagement_0")

            # Parse 0/1 bin index
            v_is_high = views_bin.endswith("1") or ("high" in views_bin.lower())
            e_is_high = eng_bin.endswith("1") or ("high" in eng_bin.lower())
            clean_tier = format_dual_tier(views_bin, eng_bin)

            # Top Platform Header
            p_name = PLATFORM_NAMES.get(p_code, p_code)
            p_logo = PLATFORM_LOGOS.get(p_code, "")

            header_html = (
                f'<div style="background:#FFFFFF; border:1px solid rgba(0,0,0,0.1); border-radius:12px; padding:1.2rem 1.4rem; margin-bottom:1.25rem; display:flex; justify-content:space-between; align-items:center;">'
                f'<div style="display:flex; align-items:center; gap:0.9rem;">'
                f'<div style="transform:scale(1.3); display:flex; align-items:center;">{p_logo}</div>'
                f'<div>'
                f'<div style="font-size:1.2rem; font-weight:700; color:#0D0D0D;">{p_name} Evaluation</div>'
                f'<div style="font-size:0.82rem; color:#555555;">Target Account: <b style="color:#0D0D0D;">{eval_page}</b> · Duration: <b style="color:#0D0D0D;">{duration_sec:.0f}s</b></div>'
                f'</div>'
                f'</div>'
                f'<div style="text-align:right;">'
                f'<div style="font-size:0.75rem; font-weight:600; color:#555555; text-transform:uppercase;">Predicted Tier</div>'
                f'<div style="font-size:1.05rem; font-weight:800; color:#0D0D0D;">{clean_tier}</div>'
                f'</div>'
                f'</div>'
            )
            st.markdown(header_html, unsafe_allow_html=True)

            # ---------------------------------------------------------------
            # PART A: ELEGANT 2x2 PERFORMANCE QUADRANT & BENCHMARK MATRIX
            # ---------------------------------------------------------------
            with st.container(border=True):
                st.markdown(
                    """
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.6rem;">
                        <div>
                            <div style="font-size:0.95rem; font-weight:700; color:#0D0D0D;">Part A: Strategic Performance Matrix</div>
                            <div style="font-size:0.8rem; color:#6B7280;">Each platform is placed at the centre of its SVC classification cell — the exact regression figures live in Part B below.</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Quadrant split lines = predict_clas high/low bin thresholds.
                # Same global yardstick the classification tier uses, so the
                # regression dot and the highlighted tier agree by construction.
                _bin_thr = classifier_bin_thresholds()
                med_views = _bin_thr["views"]
                med_eng = _bin_thr["eng"]

                # Build Sophisticated Continuous Scatter Quadrant Chart
                fig_quad = go.Figure()

                # Determine dynamic axis ranges with padding
                max_v = max(med_views * 2.2, pred_views * 1.3, 40000)
                max_e = max(med_eng * 2.2, pred_eng * 1.3, 2000)

                # Quadrant Soft Background Fills
                # Q4: Bottom-Left (Low Views, Low Engagement)
                fig_quad.add_shape(
                    type="rect", x0=0, x1=med_views, y0=0, y1=med_eng,
                    fillcolor="rgba(244, 63, 94, 0.045)" if (not v_is_high and not e_is_high) else "rgba(244, 63, 94, 0.015)",
                    line=dict(width=0),
                )
                # Q2: Top-Left (Low Views, High Engagement)
                fig_quad.add_shape(
                    type="rect", x0=0, x1=med_views, y0=med_eng, y1=max_e * 1.25,
                    fillcolor="rgba(245, 158, 11, 0.055)" if (not v_is_high and e_is_high) else "rgba(245, 158, 11, 0.015)",
                    line=dict(width=0),
                )
                # Q3: Bottom-Right (High Views, Low Engagement)
                fig_quad.add_shape(
                    type="rect", x0=med_views, x1=max_v * 1.25, y0=0, y1=med_eng,
                    fillcolor="rgba(59, 130, 246, 0.055)" if (v_is_high and not e_is_high) else "rgba(59, 130, 246, 0.015)",
                    line=dict(width=0),
                )
                # Q1: Top-Right (High Views, High Engagement)
                fig_quad.add_shape(
                    type="rect", x0=med_views, x1=max_v * 1.25, y0=med_eng, y1=max_e * 1.25,
                    fillcolor="rgba(16, 185, 129, 0.075)" if (v_is_high and e_is_high) else "rgba(16, 185, 129, 0.015)",
                    line=dict(width=0),
                )

                # Elegant Center Dividing Lines
                fig_quad.add_vline(
                    x=med_views,
                    line=dict(color="rgba(15, 23, 42, 0.28)", width=1.5, dash="dot"),
                )
                fig_quad.add_hline(
                    y=med_eng,
                    line=dict(color="rgba(15, 23, 42, 0.28)", width=1.5, dash="dot"),
                )

                # Quadrant Soft Badges / Labels
                fig_quad.add_annotation(
                    x=med_views * 0.48, y=max_e * 1.08,
                    text="<b>NICHE / DISCUSSION</b><br><span style='font-size:10px; color:#854D0E;'>High Engagement · Below Threshold Views</span>",
                    showarrow=False, font=dict(family="Inter, system-ui, sans-serif", size=10.5, color="#A16207")
                )
                fig_quad.add_annotation(
                    x=med_views + (max_v - med_views) * 0.52, y=max_e * 1.08,
                    text="<b>OPTIMAL VIRAL HIT</b><br><span style='font-size:10px; color:#14532D;'>High Reach · High Engagement</span>",
                    showarrow=False, font=dict(family="Inter, system-ui, sans-serif", size=10.5, color="#15803D")
                )
                fig_quad.add_annotation(
                    x=med_views * 0.48, y=med_eng * 0.16,
                    text="<b>LOW SIGNAL</b><br><span style='font-size:10px; color:#881337;'>Below Threshold Across Both</span>",
                    showarrow=False, font=dict(family="Inter, system-ui, sans-serif", size=10.5, color="#BE123C")
                )
                fig_quad.add_annotation(
                    x=med_views + (max_v - med_views) * 0.52, y=med_eng * 0.16,
                    text="<b>BROAD AWARENESS</b><br><span style='font-size:10px; color:#1E3A8A;'>High Views · Lower Interaction</span>",
                    showarrow=False, font=dict(family="Inter, system-ui, sans-serif", size=10.5, color="#1D4ED8")
                )

                # Add other platform ghost points (positioned by their own tier)
                _clas_map_p = {
                    q: results_by_platform[q]["clas"] for q in eval_platforms
                }
                _spread_p = _cell_spread_offsets(eval_platforms, _clas_map_p)
                for other_p in eval_platforms:
                    if other_p != p_code and other_p in results_by_platform:
                        o_lin = results_by_platform[other_p]["lin"]
                        o_clas = results_by_platform[other_p]["clas"]
                        o_v = max(0.0, float(o_lin.get("views", 0.0)))
                        o_e = max(0.0, float(o_lin.get("engagement", 0.0)))
                        _o_v_hi = _tier_is_high(o_clas.get("views", "views_0"))
                        _o_e_hi = _tier_is_high(o_clas.get("engagement", "engagement_0"))
                        o_x, o_y = _tier_center(
                            _o_v_hi, _o_e_hi, med_views, med_eng, max_v, max_e,
                        )
                        _dx, _dy = _spread_p[other_p]
                        o_x += _dx * (med_views if not _o_v_hi else max_v * 1.25 - med_views)
                        o_y += _dy * (med_eng if not _o_e_hi else max_e * 1.25 - med_eng)
                        fig_quad.add_trace(
                            go.Scatter(
                                x=[o_x],
                                y=[o_y],
                                mode="markers+text",
                                marker=dict(size=14, color="#CBD5E1", line=dict(color="#FFFFFF", width=2)),
                                text=[f"<b>{other_p}</b>"],
                                textposition="top center",
                                textfont=dict(size=9.5, color="#64748B", family="Inter, system-ui, sans-serif"),
                                hoverinfo="text",
                                hovertext=(
                                    f"<b>{PLATFORM_NAMES.get(other_p, other_p)}</b><br>"
                                    f"Tier: <b>{o_clas.get('views')} · {o_clas.get('engagement')}</b><br>"
                                    f"Regression: {o_v:,.0f} views · {o_e:,.0f} engagements"
                                ),
                                name=f"Other: {other_p}",
                                showlegend=False,
                            )
                        )

                # Active Focus Platform Point — placed at its classification cell
                act_marker_color = "#10B981" if (v_is_high and e_is_high) else ("#3B82F6" if v_is_high else ("#F59E0B" if e_is_high else "#F43F5E"))
                act_x, act_y = _tier_center(
                    v_is_high, e_is_high, med_views, med_eng, max_v, max_e
                )
                _adx, _ady = _spread_p[p_code]
                act_x += _adx * (med_views if not v_is_high else max_v * 1.25 - med_views)
                act_y += _ady * (med_eng if not e_is_high else max_e * 1.25 - med_eng)

                # Halo ring effect around active marker
                fig_quad.add_trace(
                    go.Scatter(
                        x=[act_x],
                        y=[act_y],
                        mode="markers",
                        marker=dict(
                            size=34,
                            color=act_marker_color,
                            opacity=0.22,
                        ),
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )

                # Solid Active Marker
                fig_quad.add_trace(
                    go.Scatter(
                        x=[act_x],
                        y=[act_y],
                        mode="markers+text",
                        marker=dict(
                            size=20,
                            color=act_marker_color,
                            symbol="circle",
                            line=dict(color="#FFFFFF", width=2.5),
                        ),
                        text=[f"<b>{p_name}</b>"],
                        textposition="bottom center",
                        textfont=dict(size=11, color="#0F172A", family="Inter, system-ui, sans-serif"),
                        hoverinfo="text",
                        hovertext=(
                            f"<b>★ {p_name} Classification Tier</b><br>"
                            f"Tier: <b>{views_bin} · {eng_bin}</b><br>"
                            f"Regression estimate: {pred_views:,.0f} views · {pred_eng:,.0f} engagements<br>"
                            f"High-bin thresholds: {med_views:,.0f} views · {med_eng:,.0f} eng"
                        ),
                        name=f"Active ({p_name})",
                        showlegend=False,
                    )
                )

                fig_quad.update_layout(
                    height=390,
                    margin=dict(l=50, r=30, t=20, b=45),
                    paper_bgcolor="#FFFFFF",
                    plot_bgcolor="#FAFAFC",
                    hovermode=False,
                    xaxis=dict(
                        title="<b>Views Tier (Low Views ⟵ Split ⟶ High Views)</b>",
                        title_font=dict(size=11, color="#64748B", family="Inter, system-ui, sans-serif"),
                        range=[0, max_v * 1.15],
                        showgrid=False,
                        zeroline=False,
                        fixedrange=True,
                        tickformat=",.0f",
                        tickfont=dict(size=10, color="#64748B"),
                    ),
                    yaxis=dict(
                        title="<b>Engagement Tier (Low Engagement ⟵ Split ⟶ High Engagement)</b>",
                        title_font=dict(size=11, color="#64748B", family="Inter, system-ui, sans-serif"),
                        range=[0, max_e * 1.15],
                        showgrid=False,
                        zeroline=False,
                        fixedrange=True,
                        tickformat=",.0f",
                        tickfont=dict(size=10, color="#64748B"),
                    ),
                )

                st.plotly_chart(
                    fig_quad,
                    use_container_width=True,
                    config={
                        "displayModeBar": False,
                        "staticPlot": True,
                    },
                )

                # Elegant Metric Badges Below Graph
                v_ratio = (pred_views / med_views * 100.0) if med_views > 0 else 100.0
                e_ratio = (pred_eng / med_eng * 100.0) if med_eng > 0 else 100.0
                
                badge_v_color = "#10B981" if v_ratio >= 100 else "#64748B"
                badge_e_color = "#10B981" if e_ratio >= 100 else "#64748B"

                st.markdown(
                    f"""
                    <div style="display:grid; grid-template-columns: 1fr 1fr 1fr; gap:0.6rem; margin-top:0.4rem;">
                        <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px; padding:0.6rem 0.8rem; text-align:center;">
                            <div style="font-size:0.7rem; font-weight:600; text-transform:uppercase; color:#64748B; letter-spacing:0.04em;">Views vs High Split</div>
                            <div style="font-size:1.05rem; font-weight:700; color:{badge_v_color}; margin-top:0.1rem;">{v_ratio:.0f}%</div>
                        </div>
                        <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px; padding:0.6rem 0.8rem; text-align:center;">
                            <div style="font-size:0.7rem; font-weight:600; text-transform:uppercase; color:#64748B; letter-spacing:0.04em;">Engagement vs High Split</div>
                            <div style="font-size:1.05rem; font-weight:700; color:{badge_e_color}; margin-top:0.1rem;">{e_ratio:.0f}%</div>
                        </div>
                        <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px; padding:0.6rem 0.8rem; text-align:center;">
                            <div style="font-size:0.7rem; font-weight:600; text-transform:uppercase; color:#64748B; letter-spacing:0.04em;">Predicted Tier</div>
                            <div style="font-size:1.05rem; font-weight:700; color:#0F172A; margin-top:0.1rem;">{clean_tier}</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # ---------------------------------------------------------------
            # PART B: REGRESSIONS (ACTUAL NUMBERS)
            # ---------------------------------------------------------------
            with st.container(border=True):
                st.markdown('<div style="font-size:0.95rem; font-weight:700; color:#0D0D0D; margin-bottom:0.35rem;">Part B: Regression Predictions (Actual Numbers)</div>', unsafe_allow_html=True)
                st.markdown('<div style="font-size:0.82rem; color:#555555; margin-bottom:0.75rem;">Continuous actual performance estimates predicted via predict_lin (Support Vector Regressor).</div>', unsafe_allow_html=True)

                col_reg1, col_reg2 = st.columns(2, gap="medium")
                with col_reg1:
                    m_views_html = f"""
                    <div class="metric-box">
                        <div class="metric-label">Projected Views</div>
                        <div class="metric-value">{pred_views:,.0f}</div>
                        <div style="font-size:0.78rem; color:#555555; margin-top:0.25rem;">Continuous linear regression prediction</div>
                    </div>
                    """
                    st.markdown(m_views_html, unsafe_allow_html=True)

                with col_reg2:
                    m_eng_html = f"""
                    <div class="metric-box">
                        <div class="metric-label">Projected Engagements</div>
                        <div class="metric-value">{pred_eng:,.0f}</div>
                        <div style="font-size:0.78rem; color:#555555; margin-top:0.25rem;">Likes, comments, shares & saves</div>
                    </div>
                    """
                    st.markdown(m_eng_html, unsafe_allow_html=True)

            # ---------------------------------------------------------------
            # PART C: BUDGET & ROI CALCULATOR
            # ---------------------------------------------------------------
            with st.container(border=True):
                st.markdown('<div style="font-size:0.95rem; font-weight:700; color:#0D0D0D; margin-bottom:0.35rem;">Part C: Production Budget & Value / ROI Calculator</div>', unsafe_allow_html=True)
                st.markdown(
                    f'<div style="font-size:0.82rem; color:#555555; margin-bottom:1rem;">'
                    f'Standard benchmarks for <b>{p_name}</b>: Views CPM = <b>${VIEWS_CPM.get(p_code, 10.0):.2f}</b> / 1k views · Value per Engagement = <b>${ENGAGEMENT_VALUE.get(p_code, 2.0):.2f}</b>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                col_c_in, col_c_calc = st.columns([1, 1.8], gap="medium")

                with col_c_in:
                    budget_input = st.number_input(
                        "Production Budget ($NZD)",
                        min_value=0.0,
                        value=500.0,
                        step=50.0,
                        key=f"budget_input_{p_code}",
                        help="Enter your estimated content creation / production spend.",
                    )

                # Economic Calculation
                cpm = VIEWS_CPM.get(p_code, 10.0)
                eng_val_rate = ENGAGEMENT_VALUE.get(p_code, 2.0)

                views_value = (pred_views / 1000.0) * cpm
                engagement_value = pred_eng * eng_val_rate
                gross_media_value = views_value + engagement_value
                net_value = gross_media_value - budget_input
                roi_pct = ((gross_media_value - budget_input) / budget_input * 100.0) if budget_input > 0 else 0.0

                roi_color = "#009E60" if roi_pct >= 0 else "#D32F2F"
                roi_sign = "+" if roi_pct >= 0 else ""

                with col_c_calc:
                    calc_summary_html = f"""
                    <div style="background:#FFFFFF; border:1px solid rgba(0,0,0,0.08); border-radius:8px; padding:1rem 1.25rem;">
                        <div style="display:flex; justify-content:space-between; margin-bottom:0.4rem; font-size:0.85rem;">
                            <span style="color:#555555;">Views Value ({cpm:.2f} CPM):</span>
                            <b style="color:#0D0D0D;">${views_value:,.2f}</b>
                        </div>
                        <div style="display:flex; justify-content:space-between; margin-bottom:0.4rem; font-size:0.85rem;">
                            <span style="color:#555555;">Engagement Value (${eng_val_rate:.2f}/eng):</span>
                            <b style="color:#0D0D0D;">${engagement_value:,.2f}</b>
                        </div>
                        <div style="display:flex; justify-content:space-between; border-top:1px solid rgba(0,0,0,0.08); padding-top:0.45rem; margin-top:0.45rem; font-size:0.92rem;">
                            <span style="color:#0D0D0D; font-weight:600;">Total Estimated Media Value:</span>
                            <b style="color:#0D0D0D;">${gross_media_value:,.2f}</b>
                        </div>
                        <div style="display:flex; justify-content:space-between; margin-top:0.35rem; font-size:0.92rem;">
                            <span style="color:#555555;">Net Value (Value - Budget):</span>
                            <b style="color:{roi_color};">${net_value:,.2f}</b>
                        </div>
                        <div style="display:flex; justify-content:space-between; margin-top:0.35rem; font-size:1.05rem; font-weight:700;">
                            <span style="color:#0D0D0D;">Estimated ROI:</span>
                            <span style="color:{roi_color};">{roi_sign}{roi_pct:.1f}%</span>
                        </div>
                    </div>
                    """
                    st.markdown(calc_summary_html, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# PERSISTENT HISTORY & EXPORT DRAWER
# ---------------------------------------------------------------------------
history_data = load_prediction_history()
with st.expander(f"📜 Prediction History & Export Log ({len(history_data)} Saved Runs)", expanded=False):
    if not history_data:
        st.info("No saved prediction runs yet. Click '🚀 Run Model Inference' to create your first record.")
    else:
        st.markdown("<div style='font-size:0.85rem; color:#555555; margin-bottom:0.75rem;'>All runs are persisted automatically in <code>prediction_history.json</code> without external database dependencies.</div>", unsafe_allow_html=True)

        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            st.download_button(
                "📥 Download History as JSON",
                data=json.dumps(history_data, indent=2),
                file_name="sm_optimizer_history.json",
                mime="application/json",
                use_container_width=True,
            )
        with col_exp2:
            # Flatten to CSV format
            flat_rows = []
            for r in history_data:
                for p, ev in r.get("evaluations", {}).items():
                    flat_rows.append({
                        "timestamp": r.get("timestamp"),
                        "caption": r.get("caption"),
                        "video_description": r.get("video_description"),
                        "account_page": r.get("account_page"),
                        "duration_seconds": r.get("duration_seconds"),
                        "platform": p,
                        "projected_views": ev.get("projected_views"),
                        "projected_engagement": ev.get("projected_engagement"),
                        "quadrant_views": ev.get("quadrant_views"),
                        "quadrant_engagement": ev.get("quadrant_engagement"),
                    })
            if flat_rows:
                df_export = pd.DataFrame(flat_rows)
                st.download_button(
                    "📥 Download History as CSV",
                    data=df_export.to_csv(index=False),
                    file_name="sm_optimizer_history.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        st.markdown("<div style='height:0.75rem;'></div>", unsafe_allow_html=True)
        # Display recent 5 runs in an interactive dataframe
        if flat_rows:
            st.dataframe(pd.DataFrame(flat_rows).tail(10), use_container_width=True, hide_index=True)