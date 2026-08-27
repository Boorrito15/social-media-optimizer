import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import streamlit as st

from utils.config import gcs_processed_bucket
from utils.gcs import list_existing_objects

st.set_page_config(
    page_title="Media Pipeline — Live Monitor",
    page_icon="🍏",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -----------------------------------------------------------------------------
# Apple Human Interface Inspired CSS
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=SF+Pro+Display:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", "Helvetica Neue", sans-serif;
        color: #f5f5f7;
    }
    
    .stApp {
        background: radial-gradient(circle at 50% 0%, #1c1c1e 0%, #000000 75%);
    }

    /* Apple Glass Card */
    .apple-card {
        background: rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(30px);
        -webkit-backdrop-filter: blur(30px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 22px;
        padding: 22px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .apple-card:hover {
        border-color: rgba(255, 255, 255, 0.16);
    }

    .apple-metric-title {
        font-size: 0.8rem;
        font-weight: 500;
        letter-spacing: 0.04em;
        color: #86868b;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    
    .apple-metric-val {
        font-size: 2.1rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        color: #f5f5f7;
    }

    .apple-metric-sub {
        font-size: 0.85rem;
        color: #86868b;
        margin-top: 4px;
    }

    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(48, 209, 88, 0.12);
        color: #30d158;
        border: 1px solid rgba(48, 209, 88, 0.25);
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: -0.01em;
    }

    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #30d158;
        box-shadow: 0 0 12px #30d158;
    }

    /* Custom Streamlit Progress Bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #2997ff 0%, #af52de 50%, #ff2d55 100%);
        border-radius: 10px;
    }
    .stProgress > div > div > div {
        background-color: rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        height: 8px;
    }

    /* Hide standard header/footer */
    #MainMenu, footer, header {visibility: hidden;}
    .block-container {padding-top: 2rem; padding-bottom: 2rem;}
</style>
""", unsafe_allow_html=True)

TOTAL_TARGETS = {
    "facebook": 4578,
    "instagram": 4875,
    "tiktok": 2137,
    "youtube": 1552,
}

bucket = gcs_processed_bucket()

if "start_time" not in st.session_state:
    st.session_state["start_time"] = time.time()
if "initial_counts" not in st.session_state:
    st.session_state["initial_counts"] = None

@st.cache_data(ttl=3)
def fetch_data():
    objs = list_existing_objects(bucket, prefix="videos/")
    counts = {
        "facebook": sum(1 for o in objs if "videos/facebook/" in o),
        "instagram": sum(1 for o in objs if "videos/instagram/" in o),
        "tiktok": sum(1 for o in objs if "videos/tiktok/" in o),
        "youtube": sum(1 for o in objs if "videos/youtube/" in o),
    }
    return counts, objs

counts, all_objs = fetch_data()

if st.session_state["initial_counts"] is None:
    st.session_state["initial_counts"] = counts.copy()

elapsed = max(1.0, time.time() - st.session_state["start_time"])
initial_meta = st.session_state["initial_counts"]["facebook"] + st.session_state["initial_counts"]["instagram"]
current_meta = counts["facebook"] + counts["instagram"]
meta_delta = current_meta - initial_meta
target_meta = TOTAL_TARGETS["facebook"] + TOTAL_TARGETS["instagram"]
meta_remaining = max(0, target_meta - current_meta)

vpm = (meta_delta / elapsed) * 60.0
vph = vpm * 60.0

if vpm > 0.05:
    mins_left = meta_remaining / vpm
    eta_dt = datetime.now() + timedelta(minutes=mins_left)
    eta_main = eta_dt.strftime('%H:%M')
    eta_sub = f"in ~{int(mins_left//60)}h {int(mins_left%60)}m"
else:
    eta_main = "Calculating"
    eta_sub = "speed..."

# -----------------------------------------------------------------------------
# Top Navigation Bar (Apple Style)
# -----------------------------------------------------------------------------
nav_col1, nav_col2 = st.columns([3, 1])
with nav_col1:
    st.markdown("""
    <div style="display: flex; align-items: baseline; gap: 14px; margin-bottom: 2px;">
        <h1 style="font-size: 2.1rem; font-weight: 700; letter-spacing: -0.04em; margin: 0; color: #ffffff;">
            Pipeline Overview
        </h1>
        <span style="font-size: 0.85rem; color: #86868b; font-weight: 500;">
            Branch: <span style="color: #2997ff; font-family: monospace;">meta_layer_scr</span>
        </span>
    </div>
    <div style="font-size: 0.9rem; color: #86868b; margin-bottom: 20px;">
        Automated short-form video scraping, 480p H.264 transcode & GCS persistence.
    </div>
    """, unsafe_allow_html=True)

with nav_col2:
    st.markdown("""
    <div style="text-align: right; padding-top: 6px;">
        <div class="status-pill">
            <span class="status-dot"></span>
            <span>2 Workers Live (10 Threads)</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Key Metric KPI Tiles (Glass Cards)
# -----------------------------------------------------------------------------
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    meta_pct = (current_meta / target_meta) * 100
    st.markdown(f"""
    <div class="apple-card">
        <div class="apple-metric-title">Total Meta Ingestion</div>
        <div class="apple-metric-val">{current_meta:,}</div>
        <div class="apple-metric-sub">{meta_pct:.1f}% of {target_meta:,} videos</div>
    </div>
    """, unsafe_allow_html=True)

with kpi2:
    st.markdown(f"""
    <div class="apple-card">
        <div class="apple-metric-title">Current Speed</div>
        <div class="apple-metric-val" style="color: #2997ff;">{vpm:.1f} <span style="font-size: 1.1rem; font-weight: 500;">/min</span></div>
        <div class="apple-metric-sub">~{int(vph):,} videos / hour</div>
    </div>
    """, unsafe_allow_html=True)

with kpi3:
    st.markdown(f"""
    <div class="apple-card">
        <div class="apple-metric-title">Estimated Time (ETA)</div>
        <div class="apple-metric-val" style="color: #30d158; font-size: 1.8rem;">{eta_main}</div>
        <div class="apple-metric-sub">{eta_sub}</div>
    </div>
    """, unsafe_allow_html=True)

with kpi4:
    st.markdown(f"""
    <div class="apple-card">
        <div class="apple-metric-title">GCS Target Bucket</div>
        <div class="apple-metric-val" style="font-size: 1.25rem; color: #af52de; word-break: break-all;">{bucket}</div>
        <div class="apple-metric-sub">asia-southeast2 • standard</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# -----------------------------------------------------------------------------
# Overall Progress Segment
# -----------------------------------------------------------------------------
st.markdown(f"""
<div class="apple-card" style="margin-bottom: 20px;">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
        <span style="font-weight: 600; font-size: 1rem; letter-spacing: -0.02em;">Meta Layer Progress (Facebook + Instagram)</span>
        <span style="font-weight: 700; color: #2997ff; font-size: 1.05rem;">{current_meta:,} / {target_meta:,} ({meta_pct:.1f}%)</span>
    </div>
</div>
""", unsafe_allow_html=True)
st.progress(min(1.0, current_meta / target_meta))

st.write("")

# -----------------------------------------------------------------------------
# Platform Deep Dive Cards
# -----------------------------------------------------------------------------
p_col1, p_col2 = st.columns(2)

with p_col1:
    fb_pct = (counts["facebook"] / TOTAL_TARGETS["facebook"]) * 100
    st.markdown(f"""
    <div class="apple-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <div style="width: 32px; height: 32px; border-radius: 8px; background: rgba(24, 119, 242, 0.15); display: flex; align-items: center; justify-content: center; font-weight: bold; color: #1877F2;">
                    f
                </div>
                <div>
                    <div style="font-weight: 600; font-size: 1rem; color: #f5f5f7;">Facebook Reels</div>
                    <div style="font-size: 0.75rem; color: #86868b;">5 Parallel Concurrency Threads</div>
                </div>
            </div>
            <span style="font-weight: 700; font-size: 1.2rem; color: #1877F2;">{fb_pct:.1f}%</span>
        </div>
        <div style="font-size: 1.7rem; font-weight: 700; letter-spacing: -0.03em; margin-bottom: 2px;">
            {counts['facebook']:,} <span style="font-size: 0.9rem; color: #86868b; font-weight: 400;">/ {TOTAL_TARGETS['facebook']:,}</span>
        </div>
        <div style="font-size: 0.8rem; color: #86868b; margin-bottom: 12px;">
            {TOTAL_TARGETS['facebook'] - counts['facebook']:,} videos remaining
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.progress(min(1.0, counts["facebook"] / TOTAL_TARGETS["facebook"]))

with p_col2:
    ig_pct = (counts["instagram"] / TOTAL_TARGETS["instagram"]) * 100
    st.markdown(f"""
    <div class="apple-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <div style="width: 32px; height: 32px; border-radius: 8px; background: rgba(225, 48, 108, 0.15); display: flex; align-items: center; justify-content: center; font-size: 1.1rem; color: #E1306C;">
                    📸
                </div>
                <div>
                    <div style="font-weight: 600; font-size: 1rem; color: #f5f5f7;">Instagram Reels</div>
                    <div style="font-size: 0.75rem; color: #86868b;">5 Parallel Concurrency Threads</div>
                </div>
            </div>
            <span style="font-weight: 700; font-size: 1.2rem; color: #E1306C;">{ig_pct:.1f}%</span>
        </div>
        <div style="font-size: 1.7rem; font-weight: 700; letter-spacing: -0.03em; margin-bottom: 2px;">
            {counts['instagram']:,} <span style="font-size: 0.9rem; color: #86868b; font-weight: 400;">/ {TOTAL_TARGETS['instagram']:,}</span>
        </div>
        <div style="font-size: 0.8rem; color: #86868b; margin-bottom: 12px;">
            {TOTAL_TARGETS['instagram'] - counts['instagram']:,} videos remaining
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.progress(min(1.0, counts["instagram"] / TOTAL_TARGETS["instagram"]))

st.write("")

# -----------------------------------------------------------------------------
# Clean Data Table & Recent Uploads
# -----------------------------------------------------------------------------
t_col1, t_col2 = st.columns([1, 1])

with t_col1:
    st.markdown("""<div class="apple-metric-title">All Platforms Status</div>""", unsafe_allow_html=True)
    table_df = pd.DataFrame([
        {"Platform": "Facebook", "Scraped": f"{counts['facebook']:,}", "Target": f"{TOTAL_TARGETS['facebook']:,}", "Status": f"{fb_pct:.1f}%"},
        {"Platform": "Instagram", "Scraped": f"{counts['instagram']:,}", "Target": f"{TOTAL_TARGETS['instagram']:,}", "Status": f"{ig_pct:.1f}%"},
        {"Platform": "TikTok", "Scraped": f"{counts['tiktok']:,}", "Target": f"{TOTAL_TARGETS['tiktok']:,}", "Status": "93.4%"},
        {"Platform": "YouTube", "Scraped": f"{counts['youtube']:,}", "Target": f"{TOTAL_TARGETS['youtube']:,}", "Status": "98.8%"},
    ])
    st.dataframe(table_df, use_container_width=True, hide_index=True)

with t_col2:
    st.markdown("""<div class="apple-metric-title">Recent GCS 480p Deliverables</div>""", unsafe_allow_html=True)
    recent_objs = [o for o in sorted(all_objs, reverse=True) if o.endswith(".mp4")][:5]
    if recent_objs:
        recent_df = pd.DataFrame({
            "Object Key": recent_objs,
            "Transcode": ["H.264 480p"] * len(recent_objs)
        })
        st.dataframe(recent_df, use_container_width=True, hide_index=True)

st.markdown(f"""
<div style="text-align: center; color: #86868b; font-size: 0.75rem; margin-top: 24px;">
    Updated {datetime.now().strftime('%H:%M:%S')} • Auto-refreshes every 3 seconds
</div>
""", unsafe_allow_html=True)

time.sleep(3)
st.rerun()
