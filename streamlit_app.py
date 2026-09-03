"""Social Media Optimizer — Studio & Prediction Engine.

Architecture:
- Page 1 · Concept: Refined hero composer with preset chips & auto-detection.
- Page 2 · Parameters: Streamlined 2-column layout (Distribution & Setting on left, Content & Entities on right).
- Page 3 · Forecast: Cross-platform score matrix (FB, IG, TT, YT), reach metrics & SQLite persistence.

Flipped / Light Colorway Palette:
- background_primary: #FFFFFF
- background_secondary: #F4F4F6
- text_primary: #0D0D0D
- text_secondary: #555555
- accent_branding: #1A1A1A
- data_success: #009E60
- data_alert: #D32F2F
"""

from __future__ import annotations

import html
import os
import sys
from pathlib import Path
from typing import List, Tuple

import streamlit as st

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.db.database import delete_prediction, get_history, get_prediction, init_db
from src.ml.service import infer_metadata, run_prediction_pipeline

# ---------------------------------------------------------------------------
# Streamlit Page Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Social Media Optimizer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Initialize database schema
init_db()

# ---------------------------------------------------------------------------
# Controlled Vocabularies from notebooks/describe_rob.ipynb & models/reference.md
# ---------------------------------------------------------------------------
PLATFORMS = [
    ("ALL", "All 4 Platforms (FB, IG, TT, YT)"),
    ("FB", "Facebook"),
    ("IG", "Instagram"),
    ("TT", "TikTok"),
    ("YT", "YouTube"),
]
PLATFORM_KEYS = [p[0] for p in PLATFORMS]
PLATFORM_LABELS = dict(PLATFORMS)

# Official Platform Brand SVG Logos
PLATFORM_LOGOS = {
    "FB": """<svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="vertical-align:middle; display:inline-block;"><path d="M24 12.073c0-6.627-5.373-12-12-12s-12 5.373-12 12c0 5.99 4.388 10.954 10.125 11.854v-8.385H7.078v-3.47h3.047V9.43c0-3.007 1.792-4.669 4.533-4.669 1.312 0 2.686.235 2.686.235v2.953H15.83c-1.491 0-1.956.925-1.956 1.874v2.25h3.328l-.532 3.47h-2.796v8.385C19.612 23.027 24 18.062 24 12.073z" fill="#1877F2"/></svg>""",
    "IG": """<svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="vertical-align:middle; display:inline-block;"><radialGradient id="ig-grad" cx="20%" cy="105%" r="120%"><stop offset="0%" stop-color="#fdf497"/><stop offset="5%" stop-color="#fdf497"/><stop offset="45%" stop-color="#fd5949"/><stop offset="60%" stop-color="#d6249f"/><stop offset="90%" stop-color="#285AEB"/></radialGradient><path d="M12 2.163c3.204 0 3.584.012 4.85.07 3.252.148 4.771 1.691 4.919 4.919.058 1.265.069 1.645.069 4.849 0 3.205-.012 3.584-.069 4.849-.149 3.225-1.664 4.771-4.919 4.919-1.266.058-1.644.07-4.85.07-3.204 0-3.584-.012-4.849-.07-3.26-.149-4.771-1.699-4.919-4.92-.058-1.265-.07-1.644-.07-4.849 0-3.204.013-3.583.07-4.849.149-3.227 1.664-4.771 4.919-4.919 1.266-.057 1.645-.069 4.849-.069zm0-2.163c-3.259 0-3.667.014-4.947.072-4.358.2-6.78 2.618-6.98 6.98-.059 1.281-.073 1.689-.073 4.948 0 3.259.014 3.668.072 4.948.2 4.358 2.618 6.78 6.98 6.98 1.281.058 1.689.072 4.948.072 3.259 0 3.668-.014 4.948-.072 4.354-.2 6.782-2.618 6.979-6.98.059-1.28.073-1.689.073-4.948 0-3.259-.014-3.667-.072-4.947-.196-4.354-2.617-6.78-6.979-6.98-1.281-.059-1.69-.073-4.949-.073zm0 5.838c-3.403 0-6.162 2.759-6.162 6.162s2.759 6.163 6.162 6.163 6.162-2.759 6.162-6.163c0-3.403-2.759-6.162-6.162-6.162zm0 10.162c-2.209 0-4-1.79-4-4 0-2.209 1.791-4 4-4s4 1.791 4 4c0 2.21-1.791 4-4 4zm6.406-11.845c-.796 0-1.441.645-1.441 1.44s.645 1.44 1.441 1.44c.795 0 1.439-.645 1.439-1.44s-.644-1.44-1.439-1.44z" fill="url(#ig-grad)"/></svg>""",
    "TT": """<svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="vertical-align:middle; display:inline-block;"><path d="M19.59 6.69a4.83 4.83 0 0 1-3.77-4.25V2h-3.45v13.67a2.89 2.89 0 0 1-2.88 2.89 2.89 2.89 0 0 1-2.89-2.89 2.89 2.89 0 0 1 2.89-2.89c.28 0 .54.04.79.1v-3.5a6.37 6.37 0 0 0-.79-.05A6.34 6.34 0 0 0 3 15.67a6.34 6.34 0 0 0 6.34 6.33 6.34 6.34 0 0 0 6.34-6.33V9.05c1.47 1.05 3.27 1.67 5.22 1.71V7.31c-.44 0-.88-.22-1.31-.62z" fill="#000000"/></svg>""",
    "YT": """<svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="vertical-align:middle; display:inline-block;"><path d="M23.498 6.186a3.016 3.016 0 0 0-2.122-2.136C19.505 3.545 12 3.545 12 3.545s-7.505 0-9.377.505A3.017 3.017 0 0 0 .502 6.186C0 8.07 0 12 0 12s0 3.93.502 5.814a3.016 3.016 0 0 0 2.122 2.136c1.871.505 9.376.505 9.376.505s7.505 0 9.377-.505a3.015 3.015 0 0 0 2.122-2.136C24 15.93 24 12 24 12s0-3.93-.502-5.814zM9.545 15.568V8.432L15.818 12l-6.273 3.568z" fill="#FF0000"/></svg>""",
}


PLATFORM_PAGES = {
    "ALL": ["All Blacks", "Black Ferns", "NZ Sevens", "NZR", "ABXV", "Bunnings NPC", "SRP"],
    "FB": ["All Blacks", "Black Ferns", "NZ Sevens", "NZR", "ABXV", "Bunnings NPC", "SRP"],
    "IG": ["All Blacks", "Black Ferns", "NZ Sevens", "NZR", "ABXV", "Bunnings NPC", "SRP"],
    "TT": ["All Blacks", "Black Ferns", "NZ Sevens", "NZR"],
    "YT": ["All Blacks", "Black Ferns", "NZ Sevens", "NZR", "ABXV"],
}

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

AUDIO_OPTIONS = ["ambient", "voice", "song", "none", "other"]
TEAM_OPTIONS = ["men", "women", "veterans", "maori", "youth"]

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

DEMO_PROMPTS = [
    {
        "label": "Winning Try",
        "desc": "Ardie Savea & Beauden Barrett match winner",
        "text": "Ardie Savea and Beauden Barrett combine for a last-minute winning try against the Springboks in their classic Adidas kit — the stadium erupts in celebration!",
    },
    {
        "label": "Training Drill",
        "desc": "Will Jordan & squad gym banter",
        "text": "Behind the scenes gym session with Will Jordan and Scott Barrett doing intense agility drills in Ineos and Tudor training gear.",
    },
    {
        "label": "Player Tribute",
        "desc": "Sam Cane farewell & legacy reflections",
        "text": "An emotional farewell tribute for Sam Cane: classic career highlights in the black jersey, standing ovation from the crowd, and a heartfelt interview.",
    },
    {
        "label": "Counter Attack",
        "desc": "Caleb Clarke & Damian McKenzie sprint",
        "text": "Caleb Clarke breaks a midfield tackle and offloads to Damian McKenzie, who chips ahead and sprints down the touchline to score under the posts in Ford sponsored match.",
    },
]


# ---------------------------------------------------------------------------
# Custom CSS: Flipped / Clean Light Mode Colorway
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* Smooth Scrolling & Base Canvas */
html {
    scroll-behavior: smooth !important;
}

body {
    background-color: #FFFFFF !important;
    color: #0D0D0D !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    letter-spacing: -0.01em !important;
}

[data-testid="stAppViewContainer"], [data-testid="stMain"], section.main {
    background-color: #FFFFFF !important;
    scroll-behavior: smooth !important;
}

/* Hide Default Streamlit Clutter */
[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"] {
    display: none !important;
}

/* Centered Main Canvas */
.main .block-container, [data-testid="stMainBlockContainer"] {
    max-width: 1040px !important;
    padding-top: 1.5rem !important;
    padding-bottom: 5rem !important;
    padding-left: 1.5rem !important;
    padding-right: 1.5rem !important;
    margin: 0 auto !important;
}

/* Sticky Top Navigation Header */
.top-nav-bar {
    position: sticky;
    top: 0;
    z-index: 999;
    background: rgba(255, 255, 255, 0.94);
    backdrop-filter: blur(12px);
    border-bottom: 1px solid rgba(0, 0, 0, 0.08);
    padding: 0.85rem 0;
    margin-bottom: 2.5rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.nav-brand {
    font-size: 0.95rem;
    font-weight: 700;
    color: #0D0D0D;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.nav-brand-badge {
    font-size: 0.7rem;
    font-weight: 600;
    color: #0D0D0D;
    background: #F4F4F6;
    border: 1px solid #CCCCCC;
    padding: 0.15rem 0.5rem;
    border-radius: 4px;
}

.nav-steps {
    display: flex;
    gap: 0.4rem;
}

.nav-step-item {
    font-size: 0.82rem;
    font-weight: 500;
    color: #555555 !important;
    text-decoration: none !important;
    padding: 0.35rem 0.75rem;
    border-radius: 6px;
    transition: all 0.15s ease;
}

.nav-step-item:hover {
    color: #0D0D0D !important;
    background: #F4F4F6;
}

/* Section Header Typography */
.section-header {
    margin-bottom: 1.8rem;
}

.section-tag {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #555555;
    margin-bottom: 0.35rem;
}

.section-title {
    font-size: 1.85rem;
    font-weight: 700;
    letter-spacing: -0.025em;
    color: #0D0D0D;
    line-height: 1.25;
    margin-bottom: 0.4rem;
}

.section-desc {
    font-size: 0.92rem;
    color: #555555;
    line-height: 1.5;
    max-width: 680px;
}

/* Streamlit Container Borders - Clean Card Style */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: #F4F4F6 !important;
    border: 1px solid rgba(0, 0, 0, 0.08) !important;
    border-radius: 12px !important;
    padding: 1.25rem 1.4rem !important;
    margin-bottom: 1rem !important;
    box-shadow: none !important;
}

.card-title {
    font-size: 0.9rem;
    font-weight: 600;
    color: #0D0D0D;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* Native Input & Widget Styling */
div[data-baseweb="input"], div[data-baseweb="select"], div[data-baseweb="textarea"] {
    background-color: #FFFFFF !important;
    border: 1px solid rgba(0, 0, 0, 0.15) !important;
    border-radius: 8px !important;
    color: #0D0D0D !important;
}

div[data-baseweb="input"]:focus-within, div[data-baseweb="select"]:focus-within, div[data-baseweb="textarea"]:focus-within {
    border-color: #0D0D0D !important;
}

/* Streamlit Multiselect Tags */
span[data-baseweb="tag"] {
    background-color: #E5E5EA !important;
    border: 1px solid rgba(0, 0, 0, 0.1) !important;
    border-radius: 6px !important;
    color: #0D0D0D !important;
}

/* Validation Alert */
.val-box-invalid {
    background: rgba(211, 47, 47, 0.08);
    border: 1px solid #D32F2F;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    color: #B71C1C;
    font-size: 0.85rem;
    line-height: 1.5;
    margin-bottom: 1rem;
}

.val-box-valid {
    background: rgba(0, 158, 96, 0.08);
    border: 1px solid #009E60;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    color: #007043;
    font-size: 0.85rem;
    margin-bottom: 1rem;
}

/* Scorecard Hero */
.scorecard-hero {
    background: #F4F4F6;
    border: 1px solid rgba(0, 0, 0, 0.1);
    border-radius: 12px;
    padding: 1.5rem 1.8rem;
    margin-bottom: 1.25rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.scorecard-number {
    font-size: 3.2rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    line-height: 1;
    margin: 0.25rem 0;
}

.score-success { color: #009E60; }
.score-alert   { color: #D32F2F; }
.score-neutral { color: #555555; }

.status-badge {
    display: inline-block;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    padding: 0.2rem 0.6rem;
    border-radius: 4px;
}
.status-green { background: rgba(0, 158, 96, 0.12); color: #007043; border: 1px solid #009E60; }
.status-red   { background: rgba(211, 47, 47, 0.12); color: #B71C1C; border: 1px solid #D32F2F; }
.status-gray  { background: rgba(0, 0, 0, 0.08); color: #555555; border: 1px solid #CCCCCC; }

/* Streamlit Native Tabs Styling */
button[data-baseweb="tab"] {
    background-color: #F4F4F6 !important;
    border: 1px solid rgba(0, 0, 0, 0.08) !important;
    border-radius: 8px 8px 0 0 !important;
    padding: 0.65rem 1.25rem !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    color: #555555 !important;
    transition: all 0.15s ease !important;
    margin-right: 0.35rem !important;
}

button[data-baseweb="tab"]:hover {
    color: #0D0D0D !important;
    background-color: #EAEAEA !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    background-color: #FFFFFF !important;
    border-color: rgba(0, 0, 0, 0.15) !important;
    border-bottom: 2px solid #0D0D0D !important;
    color: #0D0D0D !important;
}

div[data-baseweb="tab-list"] {
    gap: 4px !important;
    border-bottom: 1px solid rgba(0, 0, 0, 0.1) !important;
    margin-bottom: 1.2rem !important;
}

div[data-baseweb="tab-panel"] {
    padding-top: 0.5rem !important;
}


/* Button & Action Link Styles */
.nav-action-primary {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    background: #0D0D0D;
    color: #FFFFFF !important;
    text-decoration: none !important;
    font-size: 0.9rem;
    font-weight: 600;
    padding: 0.65rem 1.4rem;
    border-radius: 8px;
    transition: background 0.15s ease;
    width: 100%;
    text-align: center;
}

.nav-action-primary:hover {
    background: #262626;
}

.nav-action-secondary {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    background: #E5E5EA;
    border: 1px solid rgba(0, 0, 0, 0.1);
    color: #0D0D0D !important;
    text-decoration: none !important;
    font-size: 0.9rem;
    font-weight: 500;
    padding: 0.65rem 1.2rem;
    border-radius: 8px;
    transition: background 0.15s ease;
    width: 100%;
    text-align: center;
}

.nav-action-secondary:hover {
    background: #D1D1D6;
}

div.stButton > button {
    border-radius: 8px !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    padding: 0.5rem 1rem !important;
    background-color: #FFFFFF !important;
    color: #0D0D0D !important;
    border: 1px solid rgba(0, 0, 0, 0.14) !important;
    transition: all 0.15s ease !important;
}

div.stButton > button:hover {
    border-color: #0D0D0D !important;
    background-color: #F4F4F6 !important;
}

div.stButton > button[kind="primary"] {
    background-color: #0D0D0D !important;
    border: none !important;
    color: #FFFFFF !important;
}

div.stButton > button[kind="primary"]:hover {
    background-color: #262626 !important;
}
</style>

<!-- Top Sticky Header -->
<div class="top-nav-bar">
    <div class="nav-brand">
        <span>🏉 SMO Studio</span>
        <span class="nav-brand-badge">PRO</span>
    </div>
    <div class="nav-steps">
        <a href="#section-1" target="_self" class="nav-step-item">01 · Concept</a>
        <a href="#section-2" target="_self" class="nav-step-item">02 · Parameters</a>
        <a href="#section-3" target="_self" class="nav-step-item">03 · Forecast</a>
    </div>
</div>

<!-- Suppress Streamlit Cache-Clearing Keyboard Shortcuts (Cmd+C, Ctrl+C) -->
<script>
(function() {
    const handleKeyShortcuts = function(e) {
        if ((e.metaKey || e.ctrlKey) && ['c', 'C', 'v', 'V', 'x', 'X', 'a', 'A', 'z', 'Z'].includes(e.key)) {
            e.stopImmediatePropagation();
            return;
        }
        const tag = document.activeElement ? document.activeElement.tagName.toLowerCase() : '';
        if (tag === 'input' || tag === 'textarea') {
            e.stopImmediatePropagation();
        }
    };
    window.addEventListener('keydown', handleKeyShortcuts, true);
    window.addEventListener('keyup', handleKeyShortcuts, true);
    window.addEventListener('keypress', handleKeyShortcuts, true);
    if (window.parent && window.parent !== window) {
        try {
            window.parent.addEventListener('keydown', handleKeyShortcuts, true);
            window.parent.addEventListener('keyup', handleKeyShortcuts, true);
            window.parent.addEventListener('keypress', handleKeyShortcuts, true);
        } catch(err) {}
    }
})();
</script>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Callbacks for Safe State Updates (Before Widget Instantiation)
# ---------------------------------------------------------------------------
DEFAULT_DESC = "A male rugby player sprints downfield, breaks a tackle and scores a try under the posts."

if "video_description" not in st.session_state:
    st.session_state.video_description = DEFAULT_DESC

if "metadata" not in st.session_state:
    st.session_state.metadata = infer_metadata(st.session_state.video_description)

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

if "active_view_platform" not in st.session_state:
    st.session_state.active_view_platform = None



def sync_metadata_from_description() -> None:
    """Auto-generate and synchronize all metadata fields from the current video description."""
    desc = st.session_state.get("page1_desc_textarea", st.session_state.video_description)
    st.session_state.video_description = desc

    meta = infer_metadata(desc)
    st.session_state.metadata = meta

    # Sync all Page 2 form widget keys
    st.session_state.input_title = meta.get("title", desc[:60])
    st.session_state.input_platform = "ALL"
    st.session_state.input_page = meta.get("page", "All Blacks")
    st.session_state.input_duration = float(meta.get("duration_seconds", 20.0))
    st.session_state.input_themes = [t for t in (meta.get("content_theme") or []) if t in THEME_OPTIONS] or ["rugby_skills"]
    st.session_state.input_formats = [f for f in (meta.get("format_access") or []) if f in FORMAT_OPTIONS] or ["highlight"]
    st.session_state.input_tones = [t for t in (meta.get("tone") or []) if t in TONE_OPTIONS] or ["excitement"]
    st.session_state.input_people = [p for p in meta.get("people", ["all blacks"]) if p in PEOPLE_OPTIONS] or ["all blacks"]
    st.session_state.input_brands = [b for b in meta.get("brands", ["adidas"]) if b in BRAND_OPTIONS] or ["adidas"]
    st.session_state.input_context = meta.get("context", ["match day"])[0] if meta.get("context") else "match day"
    st.session_state.input_team = meta.get("overall_team", ["men"])[0] if meta.get("overall_team") else "men"
    st.session_state.input_audio = meta.get("audio_format", ["ambient"])[0] if meta.get("audio_format") else "ambient"


def apply_demo_preset(prompt_text: str) -> None:
    """Callback executed before widgets instantiate to update session state with a demo preset."""
    st.session_state.video_description = prompt_text
    st.session_state.page1_desc_textarea = prompt_text

    meta = infer_metadata(prompt_text)
    st.session_state.metadata = meta

    st.session_state.input_title = meta.get("title", prompt_text[:60])
    st.session_state.input_platform = "ALL"
    st.session_state.input_page = meta.get("page", "All Blacks")
    st.session_state.input_duration = float(meta.get("duration_seconds", 20.0))
    st.session_state.input_themes = [t for t in (meta.get("content_theme") or []) if t in THEME_OPTIONS] or ["try"]
    st.session_state.input_formats = [f for f in meta.get("format_access", []) if f in FORMAT_OPTIONS] or ["highlight"]
    st.session_state.input_tones = [t for t in (meta.get("tone") or []) if t in TONE_OPTIONS] or ["excitement"]
    st.session_state.input_people = [p for p in meta.get("people", ["all blacks"]) if p in PEOPLE_OPTIONS] or ["all blacks"]
    st.session_state.input_brands = [b for b in meta.get("brands", ["adidas"]) if b in BRAND_OPTIONS] or ["adidas"]
    st.session_state.input_context = meta.get("context", ["match day"])[0] if meta.get("context") else "match day"
    st.session_state.input_team = meta.get("overall_team", ["men"])[0] if meta.get("overall_team") else "men"
    st.session_state.input_audio = meta.get("audio_format", ["ambient"])[0] if meta.get("audio_format") else "ambient"


def load_history_record(record_id: int) -> None:
    """Callback executed to load a past run from SQLite database."""
    full_rec = get_prediction(record_id)
    if full_rec and full_rec.get("full_payload"):
        st.session_state.prediction_result = full_rec.get("full_payload")
        if full_rec.get("metadata"):
            meta = full_rec.get("metadata")
            st.session_state.metadata = meta
            desc = full_rec.get("description", meta.get("description", ""))
            st.session_state.video_description = desc
            st.session_state.page1_desc_textarea = desc
            st.session_state.input_title = meta.get("title", "")
            st.session_state.input_platform = meta.get("platform", "ALL")
            st.session_state.input_page = meta.get("page", "All Blacks")
            st.session_state.input_duration = float(meta.get("duration_seconds", 20.0))
            st.session_state.input_themes = [t for t in (meta.get("content_theme") or []) if t in THEME_OPTIONS] or ["try"]
            st.session_state.input_formats = [f for f in meta.get("format_access", []) if f in FORMAT_OPTIONS] or ["highlight"]
            st.session_state.input_tones = [t for t in (meta.get("tone") or []) if t in TONE_OPTIONS] or ["excitement"]
            st.session_state.input_people = [p for p in meta.get("people", ["all blacks"]) if p in PEOPLE_OPTIONS] or ["all blacks"]
            st.session_state.input_brands = [b for b in meta.get("brands", ["adidas"]) if b in BRAND_OPTIONS] or ["adidas"]
            st.session_state.active_view_platform = None



# ---------------------------------------------------------------------------
# Validation Helper
# ---------------------------------------------------------------------------
def validate_inputs(
    desc: str, title: str, themes: List[str], formats: List[str], tones: List[str]
) -> Tuple[bool, List[str]]:
    """Verify that all required fields for model prediction are filled."""
    missing = []
    if not desc or len(desc.strip()) < 5:
        missing.append("Video Description on Page 1 (at least 5 characters)")
    if not title or len(title.strip()) < 2:
        missing.append("Caption / Post Title on Page 2")
    if not themes:
        missing.append("At least 1 Content Theme on Page 2")
    if not formats:
        missing.append("At least 1 Format / Access tag on Page 2")
    if not tones:
        missing.append("At least 1 Tone tag on Page 2")

    is_valid = len(missing) == 0
    return is_valid, missing


# ===========================================================================
# PAGE 1: CONCEPT STUDIO
# ===========================================================================
st.markdown("<div id='section-1'></div>", unsafe_allow_html=True)
container_p1 = st.container(key="section-1")
with container_p1:
    st.markdown(
        """
        <div class="section-header">
            <div class="section-tag">01 · Concept Studio</div>
            <div class="section-title">Describe Your Video Idea</div>
            <div class="section-desc">Type your post concept in plain English. The model auto-detects rugby entities, format, tone, and duration, then evaluates reach across all platforms.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Composer Card
    with st.container(border=True):
        desc_input = st.text_area(
            label="Video Concept & Storyline",
            value=st.session_state.video_description,
            height=120,
            placeholder="e.g. A male rugby player sprints downfield, breaks a tackle, grounds the ball for a try, and the team celebrates with the crowd...",
            key="page1_desc_textarea",
            on_change=sync_metadata_from_description,
            label_visibility="collapsed",
        )

        col_act1, col_act2 = st.columns([1, 1])
        with col_act1:
            st.button(
                "⚡ Auto-Detect Attributes",
                key="btn_autogen_meta",
                on_click=sync_metadata_from_description,
                use_container_width=True,
            )
        with col_act2:
            st.markdown(
                '<a href="#section-2" target="_self" class="nav-action-primary">Continue to Parameters →</a>',
                unsafe_allow_html=True,
            )

    # Preset Chips
    st.markdown('<div style="font-size:0.8rem; font-weight:600; color:#555555; margin-top:1.2rem; margin-bottom:0.6rem;">Preset Examples</div>', unsafe_allow_html=True)
    col_demo1, col_demo2, col_demo3, col_demo4 = st.columns(4)
    demo_cols = [col_demo1, col_demo2, col_demo3, col_demo4]
    for i, demo in enumerate(DEMO_PROMPTS):
        with demo_cols[i]:
            st.button(
                f"{demo['label']}\n— {demo['desc']}",
                key=f"demo_btn_{i}",
                on_click=apply_demo_preset,
                args=(demo["text"],),
                use_container_width=True,
            )


# ===========================================================================
# PAGE 2: PARAMETERS & METADATA (Streamlined 2-Column Symmetrical Layout)
# ===========================================================================
st.markdown("<div id='section-2'></div>", unsafe_allow_html=True)
container_p2 = st.container(key="section-2")
with container_p2:
    st.markdown(
        """
        <div class="section-header">
            <div class="section-tag">02 · Parameters & Metadata</div>
            <div class="section-title">Refine Video Attributes</div>
            <div class="section-desc">Auto-extracted from your concept description. By default evaluates across all 4 platforms simultaneously. You can tweak any setting before running predictions.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    current_meta = st.session_state.metadata

    col_left, col_right = st.columns(2, gap="medium")

    with col_left:
        # Card 1: Distribution & Setting
        with st.container(border=True):
            st.markdown('<div class="card-title">📱 Distribution & Setting</div>', unsafe_allow_html=True)

            meta_title = st.text_input(
                "Post Caption / Title",
                value=current_meta.get("title") or (st.session_state.video_description[:60] if st.session_state.video_description else ""),
                key="input_title",
                help="The primary caption or headline.",
            )

            col_p1, col_p2 = st.columns(2)
            with col_p1:
                curr_plat = current_meta.get("platform", "ALL")
                plat_idx = PLATFORM_KEYS.index(curr_plat) if curr_plat in PLATFORM_KEYS else 0
                meta_platform = st.selectbox(
                    "Target Platform",
                    options=PLATFORM_KEYS,
                    index=plat_idx,
                    format_func=lambda x: f"{PLATFORM_LABELS.get(x, x)}",
                    key="input_platform",
                )

            with col_p2:
                available_pages = PLATFORM_PAGES.get(meta_platform, PLATFORM_PAGES["ALL"])
                curr_page = current_meta.get("page", "All Blacks")
                page_idx = available_pages.index(curr_page) if curr_page in available_pages else 0

                meta_page = st.selectbox(
                    "Account / Channel",
                    options=available_pages,
                    index=page_idx,
                    key="input_page",
                )

            meta_duration = st.slider(
                "Duration (seconds)",
                min_value=5.0,
                max_value=120.0,
                value=float(current_meta.get("duration_seconds", 20.0)),
                step=1.0,
                key="input_duration",
            )

            # Match Context & Team Tier & Audio
            curr_ctx = current_meta.get("context", ["match day"])[0] if current_meta.get("context") else "match day"
            ctx_idx = CONTEXT_OPTIONS.index(curr_ctx) if curr_ctx in CONTEXT_OPTIONS else 0
            meta_context = st.selectbox(
                "Match Context",
                options=CONTEXT_OPTIONS,
                index=ctx_idx,
                key="input_context",
                help="Setting or occasion from describe_rob schema.",
            )

            col_s1, col_s2 = st.columns(2)
            with col_s1:
                curr_team = current_meta.get("overall_team", ["men"])[0] if current_meta.get("overall_team") else "men"
                team_idx = TEAM_OPTIONS.index(curr_team) if curr_team in TEAM_OPTIONS else 0
                meta_team = st.selectbox(
                    "Team Tier",
                    options=TEAM_OPTIONS,
                    index=team_idx,
                    key="input_team",
                )
            with col_s2:
                curr_aud = current_meta.get("audio_format", ["ambient"])[0] if current_meta.get("audio_format") else "ambient"
                aud_idx = AUDIO_OPTIONS.index(curr_aud) if curr_aud in AUDIO_OPTIONS else 0
                meta_audio = st.selectbox(
                    "Audio Format",
                    options=AUDIO_OPTIONS,
                    index=aud_idx,
                    key="input_audio",
                )

    with col_right:
        # Card 2: Content & Entities
        with st.container(border=True):
            st.markdown('<div class="card-title">🎯 Content & Entities</div>', unsafe_allow_html=True)

            inferred_themes = [t for t in (current_meta.get("content_theme") or []) if t in THEME_OPTIONS]
            meta_themes = st.multiselect(
                "Content Themes",
                options=THEME_OPTIONS,
                default=inferred_themes or ["rugby_skills"],
                key="input_themes",
            )

            inferred_formats = [f for f in (current_meta.get("format_access") or []) if f in FORMAT_OPTIONS]
            meta_formats = st.multiselect(
                "Format & Access",
                options=FORMAT_OPTIONS,
                default=inferred_formats or ["highlight"],
                key="input_formats",
            )

            inferred_tones = [t for t in (current_meta.get("tone") or []) if t in TONE_OPTIONS]
            meta_tones = st.multiselect(
                "Emotional Tone",
                options=TONE_OPTIONS,
                default=inferred_tones or ["excitement"],
                key="input_tones",
            )

            # People Multi-Dropdown / Search
            inferred_people = [p for p in (current_meta.get("people") or []) if p in PEOPLE_OPTIONS]
            meta_people = st.multiselect(
                "People / Players",
                options=PEOPLE_OPTIONS,
                default=inferred_people or ["all blacks"],
                key="input_people",
                help="Search and select featured players or teams.",
            )

            # Brands Multi-Dropdown / Search
            inferred_brands = [b for b in (current_meta.get("brands") or []) if b in BRAND_OPTIONS]
            meta_brands = st.multiselect(
                "Brands / Sponsors",
                options=BRAND_OPTIONS,
                default=inferred_brands or ["adidas"],
                key="input_brands",
                help="Search and select sponsors or featured brands.",
            )

    # Validation Status
    is_valid, missing_fields = validate_inputs(
        st.session_state.video_description,
        meta_title,
        meta_themes,
        meta_formats,
        meta_tones,
    )

    if not is_valid:
        missing_list_str = " · ".join(missing_fields)
        st.markdown(
            f"<div class='val-box-invalid'><b>Required inputs missing:</b> {missing_list_str}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div class='val-box-valid'>Ready to evaluate <b>{PLATFORM_LABELS.get(meta_platform, meta_platform)}</b> on <b>{meta_page}</b>.</div>",
            unsafe_allow_html=True,
        )

    # Action Row
    col_nav1, col_nav2 = st.columns([1, 2], gap="medium")
    with col_nav1:
        st.markdown(
            '<a href="#section-1" target="_self" class="nav-action-secondary">← Back to Concept</a>',
            unsafe_allow_html=True,
        )

    with col_nav2:
        btn_label = "Generate Multi-Platform Score →" if meta_platform == "ALL" else f"Generate Score for {meta_platform} →"
        gen_btn = st.button(
            btn_label,
            key="btn_run_predict",
            type="primary",
            disabled=not is_valid,
            use_container_width=True,
        )
        if gen_btn and is_valid:
            payload = {
                "title": meta_title,
                "description": st.session_state.video_description,
                "platform": meta_platform,
                "page": meta_page,
                "duration_seconds": float(meta_duration),
                "year": 2025,
                "category_l0": "No Hashtag",
                "category_l1": "No Hashtag",
                "category_l2": "No Hashtag",
                "content_theme": meta_themes,
                "content_themes": meta_themes,
                "format_access": meta_formats,
                "tone": meta_tones,
                "tones": meta_tones,
                "people": meta_people if meta_people else ["all blacks"],
                "brands": meta_brands if meta_brands else ["adidas"],
                "event": [meta_page.lower()],
                "context": [meta_context],
                "overall_team": [meta_team],
                "audio_format": [meta_audio],
                "cost": 0.0,
                "expected_rpm": 3.0,
                "expected_cpm": 5.0,
            }

            st.session_state.metadata = payload

            with st.spinner("Calculating performance across platforms..."):
                result = run_prediction_pipeline(payload, save_to_db=True)
                st.session_state.prediction_result = result
                st.session_state.selected_view_platform = result.get("best_platform", "FB") if payload["platform"] == "ALL" else payload["platform"]



# ===========================================================================
# PAGE 3: PERFORMANCE FORECAST & BENCHMARKS
# ===========================================================================
st.markdown("<div id='section-3'></div>", unsafe_allow_html=True)
container_p3 = st.container(key="section-3")
with container_p3:
    st.markdown(
        """
        <div class="section-header">
            <div class="section-tag">03 · Performance Forecast</div>
            <div class="section-title">Model Verdict & Cross-Platform Metrics</div>
            <div class="section-desc">Comprehensive reach projections, engagement probabilities, and platform ranking based on historical rugby performance.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    result = st.session_state.prediction_result

    if not result:
        with st.container(border=True):
            st.markdown(
                "<div style='text-align:center; padding:2rem 1rem;'>"
                "<div style='font-size:1.1rem; font-weight:600; color:#0D0D0D; margin-bottom:0.35rem;'>Forecast Locked</div>"
                "<div style='font-size:0.88rem; color:#555555; max-width:440px; margin:0 auto 1.25rem auto; line-height:1.5;'>"
                "Complete your video concept and metadata above, then click <b>'Generate Score'</b> to unlock cross-platform predictions."
                "</div>"
                "</div>",
                unsafe_allow_html=True,
            )
            col_lk1, col_lk2, col_lk3 = st.columns([1, 1.2, 1])
            with col_lk2:
                st.markdown(
                    '<a href="#section-2" target="_self" class="nav-action-primary">Complete Parameters →</a>',
                    unsafe_allow_html=True,
                )
    else:
        saved_id = result.get("saved_id")
        is_all = result.get("is_all_platforms", True)
        best_platform = result.get("best_platform", "FB")
        leaderboard = result.get("platform_leaderboard", [])

        go_score = result.get("go_score", 0.0)
        verdict = result.get("verdict", "borderline").lower()
        verdict_msg = result.get("verdict_message", "Model evaluated your concept.")

        if go_score >= 65:
            score_class = "score-success"
            tag_class = "status-green"
            verdict_label = "Recommended · Strong Signal"
        elif go_score >= 45:
            score_class = "score-neutral"
            tag_class = "status-gray"
            verdict_label = "Borderline · Optimize Tags"
        else:
            score_class = "score-alert"
            tag_class = "status-red"
            verdict_label = "Low Signal · Review Concept"

        # Hero Scorecard
        hero_title = "Overall Multi-Platform Score" if is_all else f"{result.get('selected_platform')} Performance Score"
        hero_html = (
            f'<div class="scorecard-hero">'
            f'<div>'
            f'<div style="font-size:0.8rem; font-weight:600; color:#555555; text-transform:uppercase; letter-spacing:0.05em;">{hero_title}</div>'
            f'<div class="scorecard-number {score_class}">{go_score}<span style="font-size:1.4rem; font-weight:500; color:#888888;">/100</span></div>'
            f'<div class="status-badge {tag_class}">{verdict_label}</div>'
            f'</div>'
            f'<div style="max-width:520px; border-left:1px solid rgba(0,0,0,0.08); padding-left:1.5rem;">'
            f'<div style="font-size:0.88rem; font-weight:600; color:#0D0D0D; margin-bottom:0.25rem;">Executive Summary</div>'
            f'<div style="font-size:0.85rem; color:#555555; line-height:1.5;">{html.escape(verdict_msg)}</div>'
            f'<div style="font-size:0.82rem; color:#0D0D0D; margin-top:0.5rem; font-weight:500;">Optimal Platform: <b>{best_platform}</b> ({PLATFORM_LABELS.get(best_platform, best_platform)})</div>'
            f'</div>'
            f'</div>'
        )
        st.markdown(hero_html, unsafe_allow_html=True)

        # 4-Platform Breakdown & Deep-Dive Tabs (Instant Client-Side Tab Switching)
        st.markdown('<div style="font-size:0.88rem; font-weight:600; color:#0D0D0D; margin-bottom:0.75rem;">Cross-Platform Performance Matrix</div>', unsafe_allow_html=True)

        platforms_map = result.get("platforms", {})
        tab_fb, tab_ig, tab_tt, tab_yt = st.tabs([
            f"Facebook ({platforms_map.get('FB', {}).get('go_score', 0.0):.1f})",
            f"Instagram ({platforms_map.get('IG', {}).get('go_score', 0.0):.1f})",
            f"TikTok ({platforms_map.get('TT', {}).get('go_score', 0.0):.1f})",
            f"YouTube ({platforms_map.get('YT', {}).get('go_score', 0.0):.1f})",
        ])

        tab_mapping = [
            ("FB", "Facebook", tab_fb),
            ("IG", "Instagram", tab_ig),
            ("TT", "TikTok", tab_tt),
            ("YT", "YouTube", tab_yt),
        ]

        for p_code, p_name, tab_ctx in tab_mapping:
            with tab_ctx:
                p_data = platforms_map.get(p_code, {})
                p_score = p_data.get("go_score", 0.0)
                p_views = (p_data.get("estimates") or {}).get("views", 0)
                p_eng = (p_data.get("estimates") or {}).get("engagement", 0)
                p_vp = (p_data.get("views") or {}).get("probability", 0.5)
                p_ep = (p_data.get("engagement") or {}).get("probability", 0.5)
                p_is_best = (p_code == best_platform)
                p_logo = PLATFORM_LOGOS.get(p_code, "")
                p_fit_exp = p_data.get("fit_explanation", "")

                score_color = "#009E60" if p_score >= 65 else ("#555555" if p_score >= 45 else "#D32F2F")
                badge_html = "<span style='font-size:0.72rem; font-weight:600; color:#007043; background:rgba(0,158,96,0.12); padding:0.2rem 0.6rem; border-radius:4px;'>★ BEST FIT PLATFORM</span>" if p_is_best else ""

                # Top Platform Header Card
                header_card_html = (
                    f'<div style="background:#FFFFFF; border:1px solid rgba(0,0,0,0.1); border-radius:12px; padding:1.2rem 1.4rem; margin-bottom:1.25rem; display:flex; justify-content:space-between; align-items:center; box-shadow:0 2px 8px rgba(0,0,0,0.03);">'
                    f'<div style="display:flex; align-items:center; gap:0.9rem;">'
                    f'<div style="transform:scale(1.3); display:flex; align-items:center;">{p_logo}</div>'
                    f'<div>'
                    f'<div style="display:flex; align-items:center; gap:0.6rem;">'
                    f'<span style="font-size:1.15rem; font-weight:700; color:#0D0D0D;">{p_name}</span>'
                    f'{badge_html}'
                    f'</div>'
                    f'<div style="font-size:0.8rem; color:#555555; margin-top:0.2rem;">Reach probability: <b style="color:#0D0D0D;">{p_vp*100:.0f}%</b> · Engagement probability: <b style="color:#0D0D0D;">{p_ep*100:.0f}%</b></div>'
                    f'</div>'
                    f'</div>'
                    f'<div style="text-align:right;">'
                    f'<div style="font-size:0.72rem; font-weight:600; color:#555555; text-transform:uppercase;">Platform Score</div>'
                    f'<div style="font-size:2.2rem; font-weight:800; color:{score_color}; line-height:1;">{p_score:.1f}<span style="font-size:1rem; font-weight:500; color:#888888;">/100</span></div>'
                    f'</div>'
                    f'</div>'
                )
                st.markdown(header_card_html, unsafe_allow_html=True)

                # Lower Metrics: 50/50 Symmetrical Columns
                col_res1, col_res2 = st.columns(2, gap="medium")

                with col_res1:
                    with st.container(border=True):
                        st.markdown(
                            f'<div class="card-title">📈 Detailed Metrics: {p_name}</div>',
                            unsafe_allow_html=True,
                        )

                        col_m1, col_m2 = st.columns(2)
                        with col_m1:
                            m1_html = (
                                f'<div style="background:#FFFFFF; padding:0.85rem; border-radius:8px; border:1px solid rgba(0,0,0,0.08);">'
                                f'<div style="font-size:0.75rem; color:#555555;">Projected Views</div>'
                                f'<div style="font-size:1.35rem; font-weight:700; color:#0D0D0D; margin:0.15rem 0;">{p_views:,.0f}</div>'
                                f'<div style="font-size:0.75rem; color:#009E60;">P(High Views) = {p_vp*100:.0f}%</div>'
                                f'</div>'
                            )
                            st.markdown(m1_html, unsafe_allow_html=True)
                        with col_m2:
                            m2_html = (
                                f'<div style="background:#FFFFFF; padding:0.85rem; border-radius:8px; border:1px solid rgba(0,0,0,0.08);">'
                                f'<div style="font-size:0.75rem; color:#555555;">Projected Engagements</div>'
                                f'<div style="font-size:1.35rem; font-weight:700; color:#0D0D0D; margin:0.15rem 0;">{p_eng:,.0f}</div>'
                                f'<div style="font-size:0.75rem; color:#555555;">P(High Eng) = {p_ep*100:.0f}%</div>'
                                f'</div>'
                            )
                            st.markdown(m2_html, unsafe_allow_html=True)

                        # Fit Explanation / Why Good or Bad Fit
                        if p_fit_exp:
                            st.markdown(
                                f'<div style="margin-top:0.85rem; background:#F4F4F6; border-left:3px solid #0D0D0D; border-radius:4px; padding:0.75rem 0.9rem;">'
                                f'<div style="font-size:0.75rem; font-weight:700; text-transform:uppercase; color:#0D0D0D; margin-bottom:0.25rem;">💡 Why This Idea Fits / Misses ({p_name})</div>'
                                f'<div style="font-size:0.82rem; color:#333333; line-height:1.5;">{html.escape(p_fit_exp)}</div>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

                    # Peer Benchmarks
                    similar_posts = p_data.get("similar") or result.get("similar") or []
                    if similar_posts:
                        with st.container(border=True):
                            st.markdown(f'<div class="card-title">🔍 Historical Peer Benchmarks ({p_name})</div>', unsafe_allow_html=True)
                            for peer in similar_posts[:3]:
                                peer_title_safe = html.escape(peer.get('title', 'Untitled')[:65])
                                peer_plat_safe = peer.get('platform', p_code)
                                peer_page_safe = html.escape(peer.get('page', ''))
                                peer_views = peer.get('views', 0)
                                peer_eng = peer.get('engagement', 0)
                                peer_logo = PLATFORM_LOGOS.get(peer_plat_safe, "")

                                peer_html = (
                                    f'<div style="background:#FFFFFF; border:1px solid rgba(0,0,0,0.08); border-radius:8px; padding:0.7rem 0.85rem; margin-bottom:0.5rem;">'
                                    f'<div style="font-size:0.82rem; font-weight:500; color:#0D0D0D; margin-bottom:0.2rem;">"{peer_title_safe}"</div>'
                                    f'<div style="font-size:0.75rem; color:#555555; display:flex; align-items:center; gap:0.8rem;">'
                                    f'<span style="display:inline-flex; align-items:center; gap:0.25rem;">{peer_logo} <b>{peer_plat_safe}</b> · {peer_page_safe}</span>'
                                    f'<span>👁️ {peer_views:,.0f} views</span>'
                                    f'<span>❤️ {peer_eng:,.0f} eng</span>'
                                    f'</div>'
                                    f'</div>'
                                )
                                st.markdown(peer_html, unsafe_allow_html=True)

                with col_res2:
                    with st.container(border=True):
                        st.markdown('<div class="card-title">🏆 Platform Ranking & Fit</div>', unsafe_allow_html=True)

                        for row in leaderboard:
                            row_p_code = row["platform"]
                            row_p_name = PLATFORM_LABELS.get(row_p_code, row_p_code)
                            row_is_best = (row_p_code == best_platform)
                            row_is_current = (row_p_code == p_code)
                            row_logo = PLATFORM_LOGOS.get(row_p_code, "")
                            
                            if row_is_current:
                                bg_style = "background:#FFFFFF; border:1.5px solid #0D0D0D; box-shadow:0 2px 8px rgba(0,0,0,0.05);"
                            elif row_is_best:
                                bg_style = "background:#FFFFFF; border:1px solid #009E60;"
                            else:
                                bg_style = "background:#FFFFFF; border:1px solid rgba(0,0,0,0.08);"

                            row_score_color = "#009E60" if row["go_score"] >= 65 else ("#555555" if row["go_score"] >= 45 else "#D32F2F")
                            row_badge_label = " ★ BEST FIT" if row_is_best else (" · ACTIVE TAB" if row_is_current else "")

                            row_html = (
                                f'<div style="{bg_style} border-radius:8px; padding:0.75rem 0.9rem; display:flex; justify-content:space-between; align-items:center; margin-bottom:0.5rem;">'
                                f'<div style="display:flex; align-items:center; gap:0.6rem;">'
                                f'{row_logo}'
                                f'<div>'
                                f'<div style="font-size:0.85rem; font-weight:600; color:#0D0D0D;">{row_p_name}<span style="font-size:0.72rem; color:#007043; font-weight:600;">{row_badge_label}</span></div>'
                                f'<div style="font-size:0.72rem; color:#555555;">Reach P: {row.get("views_p", 0)*100:.0f}% · Eng P: {row.get("eng_p", 0)*100:.0f}%</div>'
                                f'</div>'
                                f'</div>'
                                f'<div style="font-size:1.25rem; font-weight:700; color:{row_score_color};">'
                                f'{row["go_score"]:.1f}'
                                f'</div>'
                                f'</div>'
                            )
                            st.markdown(row_html, unsafe_allow_html=True)




        # Database History Drawer
        with st.expander("📜 Recent Prediction History (SQLite)", expanded=False):
            history_rows = get_history(limit=8)
            if not history_rows:
                st.info("No saved records in SQLite yet.")
            else:
                for h in history_rows:
                    col_h1, col_h2, col_h3, col_h4 = st.columns([1, 4, 2, 1.2])
                    with col_h1:
                        st.markdown(f"**#{h['id']}**")
                    with col_h2:
                        st.markdown(f"**{html.escape(h.get('title') or h.get('description', '')[:45])}**")
                        st.caption(f"{h.get('platform')} · {h.get('page')} · {h.get('duration_seconds', 20):.0f}s")
                    with col_h3:
                        st.markdown(f"Score: **{h.get('go_score', 0):.1f}**")
                        st.caption(f"Views: {h.get('views_pred', 0):,.0f}")
                    with col_h4:
                        st.button(
                            "Load",
                            key=f"btn_load_h_{h['id']}",
                            on_click=load_history_record,
                            args=(h["id"],),
                        )

        # Bottom Actions
        col_end1, col_end2 = st.columns(2, gap="medium")
        with col_end1:
            st.markdown(
                '<a href="#section-1" target="_self" class="nav-action-secondary">← New Concept (Page 1)</a>',
                unsafe_allow_html=True,
            )
        with col_end2:
            st.markdown(
                '<a href="#section-2" target="_self" class="nav-action-secondary">← Edit Parameters (Page 2)</a>',
                unsafe_allow_html=True,
            )