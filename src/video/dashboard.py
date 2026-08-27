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
    page_title="MinionsScout — Pipeline Center",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;700&display=swap');
    
    .stApp {
        background-color: #F7F5F0;
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        color: #111111;
    }

    [data-testid="stStatusWidget"] { display: none !important; }
    .stApp[data-test-script-state="running"] > div:first-child { opacity: 1 !important; filter: none !important; }
    div[data-testid="stAppViewBlockContainer"] { opacity: 1 !important; filter: none !important; }

    header[data-testid="stHeader"] { background: transparent; }
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1180px; }

    .hero-banner {
        background: linear-gradient(135deg, #E02424 0%, #B91C1C 40%, #7F1D1D 100%);
        border-radius: 24px;
        padding: 32px 36px;
        color: #FFFFFF;
        margin-bottom: 20px;
        box-shadow: 0 16px 36px rgba(224, 36, 36, 0.22);
    }

    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 32px;
        font-weight: 700;
        letter-spacing: -0.03em;
        line-height: 1.1;
        margin-bottom: 6px;
    }

    .editorial-card {
        background: #FFFFFF;
        border: 1px solid rgba(17, 17, 17, 0.08);
        border-radius: 20px;
        padding: 22px 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.03);
        margin-bottom: 16px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .editorial-card:hover {
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.06);
    }

    .subhead {
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #737373;
        margin-bottom: 6px;
    }
    .metric-value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 34px;
        font-weight: 700;
        letter-spacing: -0.03em;
        color: #111111;
        line-height: 1.1;
    }
    .metric-value.crimson {
        color: #E02424;
    }
    .metric-caption {
        font-size: 13px;
        color: #737373;
        margin-top: 6px;
        font-weight: 500;
    }

    .progress-track {
        height: 8px;
        border-radius: 9999px;
        background: rgba(17, 17, 17, 0.08);
        overflow: hidden;
        margin-top: 12px;
    }
    .progress-fill {
        height: 100%;
        border-radius: 9999px;
        background: linear-gradient(90deg, #E02424 0%, #FF5A5F 100%);
    }

    .apple-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
    }
    .apple-table th {
        text-align: left;
        color: #737373;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        padding: 10px 12px;
        border-bottom: 1px solid rgba(17, 17, 17, 0.08);
    }
    .apple-table td {
        padding: 12px;
        border-bottom: 1px solid rgba(17, 17, 17, 0.04);
        color: #111111;
        font-weight: 600;
    }
    .apple-table tr:last-child td { border-bottom: none; }
    
    .table-tag {
        font-size: 11px;
        font-weight: 600;
        padding: 3px 8px;
        border-radius: 6px;
        background: rgba(224, 36, 36, 0.1);
        color: #E02424;
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

if "history" not in st.session_state:
    st.session_state["history"] = deque(maxlen=300)
if "smooth_vpm" not in st.session_state:
    st.session_state["smooth_vpm"] = None
if "smooth_seconds_left" not in st.session_state:
    st.session_state["smooth_seconds_left"] = None

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
hist.append((now, current_meta))

# EMA smoothing calculation over last 3 minutes / 30 downloads
lookback_sec = 180.0
min_downloads = 25

ref_time, ref_count = hist[0]
for t_hist, c_hist in hist:
    if (now - t_hist) <= lookback_sec and (current_meta - c_hist) >= min_downloads:
        ref_time, ref_count = t_hist, c_hist
        break

delta_count = current_meta - ref_count
delta_time = max(1.0, now - ref_time)

if delta_count > 0 and delta_time > 5.0:
    raw_vps = delta_count / delta_time
    raw_vpm = raw_vps * 60.0
    
    alpha = 0.15
    if st.session_state["smooth_vpm"] is None:
        st.session_state["smooth_vpm"] = raw_vpm
    else:
        st.session_state["smooth_vpm"] = (alpha * raw_vpm) + ((1.0 - alpha) * st.session_state["smooth_vpm"])
    
    smooth_vpm = st.session_state["smooth_vpm"]
    smooth_vps = max(0.01, smooth_vpm / 60.0)
    raw_seconds_left = meta_remaining / smooth_vps
    
    if st.session_state["smooth_seconds_left"] is None:
        st.session_state["smooth_seconds_left"] = raw_seconds_left
    else:
        st.session_state["smooth_seconds_left"] = (0.10 * raw_seconds_left) + (0.90 * st.session_state["smooth_seconds_left"])
    
    smooth_seconds = st.session_state["smooth_seconds_left"]
    eta_dt = datetime.now() + timedelta(seconds=smooth_seconds)
    
    hours = int(smooth_seconds // 3600)
    minutes = int((smooth_seconds % 3600) // 60)
    
    eta_val = eta_dt.strftime("%H:%M")
    eta_cap = f"in ~{hours}h {minutes}m (EMA stabilized)"
    vpm_val = round(smooth_vpm, 1)
    vph_val = int(smooth_vpm * 60.0)
else:
    vpm_val = 0.0
    vph_val = 0
    eta_val = "--:--"
    eta_cap = "Sampling downloads..."

# -----------------------------------------------------------------------------
# Hero Banner
# -----------------------------------------------------------------------------
st.markdown(
    """
<div class="hero-banner">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
        <div style="display: flex; align-items: center; gap: 10px;">
            <span style="font-size: 22px;">❋</span>
            <span style="font-family: 'Space Grotesk', sans-serif; font-size: 22px; font-weight: 700; letter-spacing: 0.05em; text-transform: uppercase;">MinionsScout</span>
        </div>
        <span style="background: rgba(255,255,255,0.2); padding: 5px 12px; border-radius: 9999px; font-size: 11px; font-weight: 700; letter-spacing: 0.04em;">
            ● 2 WORKERS LIVE (10 THREADS)
        </span>
    </div>
    <div class="hero-title">Make an impact.</div>
    <div style="font-size: 14px; opacity: 0.9;">
        Short-form video scraping, 480p H.264 transcode & GCS persistence on <code style="background: rgba(0,0,0,0.2); padding: 2px 6px; border-radius: 4px; color: #fff;">meta_layer_scr</code>
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
    <div class="editorial-card">
        <div class="subhead">Total Meta Ingestion</div>
        <div class="metric-value crimson">{current_meta:,}</div>
        <div class="metric-caption">{meta_pct:.1f}% of {target_meta:,} videos</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        f"""
    <div class="editorial-card">
        <div class="subhead">Stabilized Speed</div>
        <div class="metric-value">{vpm_val:.1f} <span style="font-size: 18px; font-weight: 500; color: #737373;">/min</span></div>
        <div class="metric-caption">~{vph_val:,} videos / hour</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        f"""
    <div class="editorial-card">
        <div class="subhead">EMA Smoothed ETA</div>
        <div class="metric-value crimson">{eta_val}</div>
        <div class="metric-caption">{eta_cap}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with c4:
    st.markdown(
        f"""
    <div class="editorial-card">
        <div class="subhead">GCS Target Bucket</div>
        <div class="metric-value" style="font-size: 18px; font-weight: 700; color: #111111; padding-top: 6px;">{bucket}</div>
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
<div class="editorial-card" style="background: #111111; color: #FFFFFF;">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <span style="font-weight: 700; font-family: 'Space Grotesk', sans-serif; font-size: 15px;">Overall Meta Layer Progress</span>
        <span style="font-size: 14px; color: #FF5A5F; font-weight: 700; font-family: 'Space Grotesk', sans-serif;">{current_meta:,} / {target_meta:,} ({meta_pct:.1f}%)</span>
    </div>
    <div class="progress-track" style="background: rgba(255,255,255,0.15);">
        <div class="progress-fill" style="width: {min(100.0, meta_pct):.1f}%;"></div>
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
    <div class="editorial-card">
        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
            <div>
                <div style="font-weight: 700; font-family: 'Space Grotesk', sans-serif; font-size: 16px;">Facebook Reels</div>
                <div style="font-size: 12px; color: #737373;">5 Parallel Concurrency Threads</div>
            </div>
            <span style="font-size: 12px; font-weight: 700; background: rgba(224,36,36,0.1); color: #E02424; padding: 3px 8px; border-radius: 9999px;">{fb_pct:.1f}%</span>
        </div>
        <div style="margin-top: 14px;">
            <span class="metric-value">{fb_count:,}</span>
            <span style="color: #737373; font-size: 14px; font-weight: 500;"> / {TOTAL_TARGETS['facebook']:,}</span>
        </div>
        <div style="font-size: 12px; color: #737373; margin-top: 2px;">{fb_rem:,} videos remaining</div>
        <div class="progress-track">
            <div class="progress-fill" style="width: {min(100.0, fb_pct):.1f}%; background: #111111;"></div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col_ig:
    st.markdown(
        f"""
    <div class="editorial-card">
        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
            <div>
                <div style="font-weight: 700; font-family: 'Space Grotesk', sans-serif; font-size: 16px;">Instagram Reels</div>
                <div style="font-size: 12px; color: #737373;">5 Parallel Concurrency Threads</div>
            </div>
            <span style="font-size: 12px; font-weight: 700; background: rgba(224,36,36,0.1); color: #E02424; padding: 3px 8px; border-radius: 9999px;">{ig_pct:.1f}%</span>
        </div>
        <div style="margin-top: 14px;">
            <span class="metric-value">{ig_count:,}</span>
            <span style="color: #737373; font-size: 14px; font-weight: 500;"> / {TOTAL_TARGETS['instagram']:,}</span>
        </div>
        <div style="font-size: 12px; color: #737373; margin-top: 2px;">{ig_rem:,} videos remaining</div>
        <div class="progress-track">
            <div class="progress-fill" style="width: {min(100.0, ig_pct):.1f}%;"></div>
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
    <div class="editorial-card">
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
                    <td style="color: #737373;">{TOTAL_TARGETS['facebook']:,}</td>
                    <td style="color: #E02424;">{fb_pct:.1f}%</td>
                </tr>
                <tr>
                    <td>Instagram</td>
                    <td>{ig_count:,}</td>
                    <td style="color: #737373;">{TOTAL_TARGETS['instagram']:,}</td>
                    <td style="color: #E02424;">{ig_pct:.1f}%</td>
                </tr>
                <tr>
                    <td>TikTok</td>
                    <td>{counts['tiktok']:,}</td>
                    <td style="color: #737373;">{TOTAL_TARGETS['tiktok']:,}</td>
                    <td style="color: #111111;">{tt_pct:.1f}%</td>
                </tr>
                <tr>
                    <td>YouTube</td>
                    <td>{counts['youtube']:,}</td>
                    <td style="color: #737373;">{TOTAL_TARGETS['youtube']:,}</td>
                    <td style="color: #111111;">{yt_pct:.1f}%</td>
                </tr>
            </tbody>
        </table>
    </div>
    """,
        unsafe_allow_html=True,
    )

with t_col2:
    recent_rows = "".join([
        f'<tr><td style="font-family: monospace; font-size: 12px; color: #444444;">{r}</td><td style="text-align: right;"><span class="table-tag">480p Ready</span></td></tr>'
        for r in recent_files
    ]) if recent_files else '<tr><td colspan="2" style="color: #737373;">Loading uploads...</td></tr>'

    st.markdown(
        f"""
    <div class="editorial-card">
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
<div style="text-align: center; font-size: 11px; color: #888888; margin-top: 20px; font-weight: 500;">
    MinionsScout • Exponential Moving Average (EMA) Stabilized Speed & ETA • Last Ping: {datetime.now().strftime('%H:%M:%S')}
</div>
""",
    unsafe_allow_html=True,
)

time.sleep(2)
st.rerun()
