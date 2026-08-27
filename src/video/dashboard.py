import sys
import time
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
from utils.config import gcs_processed_bucket
from utils.gcs import list_existing_objects

st.set_page_config(
    page_title="Pipeline Command Center",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -----------------------------------------------------------------------------
# Apple Human Interface Dark CSS Injection
# -----------------------------------------------------------------------------
st.markdown(
    """
<style>
    /* System Font & Background */
    @import url('https://fonts.cdnfonts.com/css/sf-pro-display');
    
    .stApp {
        background-color: #000000;
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", "Helvetica Neue", sans-serif;
        color: #F5F5F7;
    }

    /* Prevent Screen Dimming & Reload Flicker */
    [data-testid="stStatusWidget"] { display: none !important; }
    .stApp[data-test-script-state="running"] > div:first-child { opacity: 1 !important; filter: none !important; }
    div[data-testid="stAppViewBlockContainer"] { opacity: 1 !important; filter: none !important; }

    /* Hide standard Streamlit header & margins */
    header[data-testid="stHeader"] { background: transparent; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1200px; }

    /* Apple Glassmorphism Card */
    .apple-card {
        background: rgba(28, 28, 30, 0.65);
        backdrop-filter: blur(25px) saturate(190%);
        -webkit-backdrop-filter: blur(25px) saturate(190%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 18px;
        padding: 20px 22px;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.4);
        margin-bottom: 16px;
        transition: border-color 0.2s ease, transform 0.2s ease;
    }
    .apple-card:hover {
        border-color: rgba(255, 255, 255, 0.16);
    }

    /* Typography Utilities */
    .subhead {
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: rgba(235, 235, 245, 0.6);
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 32px;
        font-weight: 700;
        letter-spacing: -0.03em;
        color: #FFFFFF;
        line-height: 1.1;
    }
    .metric-caption {
        font-size: 13px;
        color: rgba(235, 235, 245, 0.4);
        margin-top: 6px;
    }

    /* Status Pill */
    .apple-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 12px;
        font-weight: 500;
        padding: 5px 12px;
        border-radius: 9999px;
        background: rgba(48, 209, 88, 0.12);
        color: #30D158;
        border: 1px solid rgba(48, 209, 88, 0.22);
    }
    .status-dot {
        width: 6px;
        height: 6px;
        background-color: #30D158;
        border-radius: 50%;
        box-shadow: 0 0 8px #30D158;
    }

    /* Code Pill */
    .code-pill {
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.08);
        color: #0A84FF;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 12px;
        padding: 2px 7px;
        border-radius: 6px;
    }

    /* Apple-Style Progress Bar */
    .progress-track {
        height: 6px;
        border-radius: 9999px;
        background: rgba(255, 255, 255, 0.08);
        overflow: hidden;
        margin-top: 12px;
    }
    .progress-fill {
        height: 100%;
        border-radius: 9999px;
        background: #0A84FF;
        transition: width 0.6s cubic-bezier(0.16, 1, 0.3, 1);
    }
    .progress-fill.meta {
        background: linear-gradient(90deg, #0A84FF 0%, #BF5AF2 100%);
    }

    /* Table Styling */
    .apple-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
    }
    .apple-table th {
        text-align: left;
        color: rgba(235, 235, 245, 0.5);
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        padding: 10px 12px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }
    .apple-table td {
        padding: 12px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.04);
        color: #FFFFFF;
    }
    .apple-table tr:last-child td { border-bottom: none; }
    
    .table-tag {
        font-size: 11px;
        font-weight: 500;
        padding: 3px 8px;
        border-radius: 6px;
        background: rgba(48, 209, 88, 0.12);
        color: #30D158;
    }
</style>
""",
    unsafe_allow_html=True,
)

TOTAL_TARGETS = {
    "facebook": 4578,
    "instagram": 4875,
    "tiktok": 2137,
    "youtube": 1552,
}

bucket = gcs_processed_bucket()

# Session state for rolling history
if "history" not in st.session_state:
    st.session_state["history"] = deque(maxlen=200)

@st.cache_data(ttl=2)
def fetch_data():
    objs = list_existing_objects(bucket, prefix="videos/")
    counts = {
        "facebook": sum(1 for o in objs if "videos/facebook/" in o),
        "instagram": sum(1 for o in objs if "videos/instagram/" in o),
        "tiktok": sum(1 for o in objs if "videos/tiktok/" in o),
        "youtube": sum(1 for o in objs if "videos/youtube/" in o),
    }
    recent = [o.replace("videos/", "") for o in sorted(objs, reverse=True) if o.endswith(".mp4")][:5]
    return counts, objs, recent

counts, all_objs, recent_files = fetch_data()

now = time.time()
current_meta = counts["facebook"] + counts["instagram"]
target_meta = TOTAL_TARGETS["facebook"] + TOTAL_TARGETS["instagram"]
meta_remaining = max(0, target_meta - current_meta)
meta_pct = (current_meta / target_meta) * 100

fb_count = counts["facebook"]
fb_pct = (fb_count / TOTAL_TARGETS["facebook"]) * 100
fb_rem = TOTAL_TARGETS["facebook"] - fb_count

ig_count = counts["instagram"]
ig_pct = (ig_count / TOTAL_TARGETS["instagram"]) * 100
ig_rem = TOTAL_TARGETS["instagram"] - ig_count

# Record history point
hist = st.session_state["history"]
if not hist or hist[-1][1] != current_meta:
    hist.append((now, current_meta))

# Rolling 30-download speed & ETA
target_window = 30
ref_time, ref_count = hist[0]
for t_hist, c_hist in reversed(hist):
    if (current_meta - c_hist) >= target_window:
        ref_time, ref_count = t_hist, c_hist
        break

delta_count = current_meta - ref_count
delta_time = max(1.0, now - ref_time)

if delta_count > 0 and delta_time > 1.0:
    vps = delta_count / delta_time
    vpm = vps * 60.0
    vph = int(vps * 3600.0)
    seconds_left = meta_remaining / vps
    eta_dt = datetime.now() + timedelta(seconds=seconds_left)
    hours = int(seconds_left // 3600)
    minutes = int((seconds_left % 3600) // 60)
    eta_val = eta_dt.strftime("%H:%M")
    eta_cap = f"in ~{hours}h {minutes}m (last {delta_count} vids)"
else:
    vpm = 0.0
    vph = 0
    eta_val = "--:--"
    eta_cap = "Calibrating speed..."

# -----------------------------------------------------------------------------
# Header Section
# -----------------------------------------------------------------------------
header_col1, header_col2 = st.columns([4, 1])
with header_col1:
    st.markdown(
        """
    <div style="margin-bottom: 24px;">
        <h1 style="font-size: 28px; font-weight: 700; letter-spacing: -0.03em; margin: 0; color: #FFFFFF;">Pipeline Command Center</h1>
        <p style="color: rgba(235, 235, 245, 0.6); font-size: 14px; margin-top: 4px;">
            Short-form video scraping, 480p H.264 transcode & GCS sync on <span class="code-pill">meta_layer_scr</span>
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

with header_col2:
    st.markdown(
        """
    <div style="display: flex; justify-content: flex-end; align-items: center; height: 100%;">
        <div class="apple-badge">
            <div class="status-dot"></div>
            2 Workers Live (10 Threads)
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------------------------
# Top Metrics Row
# -----------------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(
        f"""
    <div class="apple-card">
        <div class="subhead">Total Meta Ingestion</div>
        <div class="metric-value">{current_meta:,}</div>
        <div class="metric-caption">{meta_pct:.1f}% of {target_meta:,} videos</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        f"""
    <div class="apple-card">
        <div class="subhead">Current Speed</div>
        <div class="metric-value" style="color: #0A84FF;">{vpm:.1f} <span style="font-size: 18px; font-weight: 500; color: rgba(235,235,245,0.6);">/min</span></div>
        <div class="metric-caption">~{vph:,} videos / hour</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        f"""
    <div class="apple-card">
        <div class="subhead">ETA (Last 30 Vids)</div>
        <div class="metric-value" style="color: #30D158;">{eta_val}</div>
        <div class="metric-caption">{eta_cap}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with c4:
    st.markdown(
        f"""
    <div class="apple-card">
        <div class="subhead">GCS Target Bucket</div>
        <div class="metric-value" style="font-size: 18px; font-weight: 600; color: #BF5AF2; padding-top: 6px;">{bucket}</div>
        <div class="metric-caption">asia-southeast2 • standard</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------------------------
# Overall Meta Layer Progress
# -----------------------------------------------------------------------------
st.markdown(
    f"""
<div class="apple-card">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <span style="font-weight: 600; font-size: 14px;">Overall Meta Layer Progress</span>
        <span style="font-size: 13px; color: #0A84FF; font-weight: 600;">{current_meta:,} / {target_meta:,} ({meta_pct:.1f}%)</span>
    </div>
    <div class="progress-track">
        <div class="progress-fill meta" style="width: {min(100.0, meta_pct):.1f}%;"></div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Source Details Grid (FB & IG)
# -----------------------------------------------------------------------------
col_fb, col_ig = st.columns(2)

with col_fb:
    st.markdown(
        f"""
    <div class="apple-card">
        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
            <div>
                <div style="font-weight: 600; font-size: 15px;">Facebook Reels</div>
                <div style="font-size: 12px; color: rgba(235, 235, 245, 0.4);">5 Parallel Concurrency Threads</div>
            </div>
            <span style="font-size: 16px; font-weight: 700; color: #0A84FF;">{fb_pct:.1f}%</span>
        </div>
        <div style="margin-top: 14px;">
            <span class="metric-value">{fb_count:,}</span>
            <span style="color: rgba(235, 235, 245, 0.4); font-size: 14px;"> / {TOTAL_TARGETS['facebook']:,}</span>
        </div>
        <div style="font-size: 12px; color: rgba(235, 235, 245, 0.4); margin-top: 2px;">{fb_rem:,} videos remaining</div>
        <div class="progress-track">
            <div class="progress-fill" style="width: {min(100.0, fb_pct):.1f}%; background: #0A84FF;"></div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col_ig:
    st.markdown(
        f"""
    <div class="apple-card">
        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
            <div>
                <div style="font-weight: 600; font-size: 15px;">Instagram Reels</div>
                <div style="font-size: 12px; color: rgba(235, 235, 245, 0.4);">5 Parallel Concurrency Threads</div>
            </div>
            <span style="font-size: 16px; font-weight: 700; color: #FF375F;">{ig_pct:.1f}%</span>
        </div>
        <div style="margin-top: 14px;">
            <span class="metric-value">{ig_count:,}</span>
            <span style="color: rgba(235, 235, 245, 0.4); font-size: 14px;"> / {TOTAL_TARGETS['instagram']:,}</span>
        </div>
        <div style="font-size: 12px; color: rgba(235, 235, 245, 0.4); margin-top: 2px;">{ig_rem:,} videos remaining</div>
        <div class="progress-track">
            <div class="progress-fill" style="width: {min(100.0, ig_pct):.1f}%; background: #FF375F;"></div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------------------------
# Tables Row
# -----------------------------------------------------------------------------
t_col1, t_col2 = st.columns(2)

with t_col1:
    tt_pct = (counts['tiktok'] / TOTAL_TARGETS['tiktok']) * 100
    yt_pct = (counts['youtube'] / TOTAL_TARGETS['youtube']) * 100
    st.markdown(
        f"""
    <div class="apple-card">
        <div class="subhead" style="margin-bottom: 12px;">All Platforms Status</div>
        <table class="apple-table">
            <thead>
                <tr>
                    <th>Platform</th>
                    <th>Uploaded</th>
                    <th>Target</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>Facebook</td>
                    <td>{fb_count:,}</td>
                    <td style="color: rgba(235,235,245,0.6);">{TOTAL_TARGETS['facebook']:,}</td>
                    <td style="color: #0A84FF;">{fb_pct:.1f}%</td>
                </tr>
                <tr>
                    <td>Instagram</td>
                    <td>{ig_count:,}</td>
                    <td style="color: rgba(235,235,245,0.6);">{TOTAL_TARGETS['instagram']:,}</td>
                    <td style="color: #FF375F;">{ig_pct:.1f}%</td>
                </tr>
                <tr>
                    <td>TikTok</td>
                    <td>{counts['tiktok']:,}</td>
                    <td style="color: rgba(235,235,245,0.6);">{TOTAL_TARGETS['tiktok']:,}</td>
                    <td style="color: #30D158;">{tt_pct:.1f}%</td>
                </tr>
                <tr>
                    <td>YouTube</td>
                    <td>{counts['youtube']:,}</td>
                    <td style="color: rgba(235,235,245,0.6);">{TOTAL_TARGETS['youtube']:,}</td>
                    <td style="color: #30D158;">{yt_pct:.1f}%</td>
                </tr>
            </tbody>
        </table>
    </div>
    """,
        unsafe_allow_html=True,
    )

with t_col2:
    recent_rows = "".join([
        f'<tr><td style="font-family: ui-monospace, SFMono-Regular, monospace; font-size: 12px; color: rgba(235,235,245,0.8);">{r}</td><td style="text-align: right;"><span class="table-tag">480p Ready</span></td></tr>'
        for r in recent_files
    ]) if recent_files else '<tr><td colspan="2" style="color: rgba(235,235,245,0.4);">Loading uploads...</td></tr>'

    st.markdown(
        f"""
    <div class="apple-card">
        <div class="subhead" style="margin-bottom: 12px;">Recent GCS 480p Deliverables</div>
        <table class="apple-table">
            <thead>
                <tr>
                    <th>GCS Object Key</th>
                    <th style="text-align: right;">Format</th>
                </tr>
            </thead>
            <tbody>
                {recent_rows}
            </tbody>
        </table>
    </div>
    """,
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown(
    f"""
<div style="text-align: center; font-size: 11px; color: rgba(235, 235, 245, 0.3); margin-top: 20px; letter-spacing: 0.02em;">
    Ultra-responsive Rolling 30-Download ETA Algorithm • Zero-Reload Fluid Updates • Last Ping: {datetime.now().strftime('%H:%M:%S')}
</div>
""",
    unsafe_allow_html=True,
)

time.sleep(2)
st.rerun()
