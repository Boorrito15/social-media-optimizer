"""Interactive Live Dashboard for Video Scraper Progress (Streamlit)."""

import time
from datetime import datetime
import pandas as pd
import streamlit as st

from utils.config import gcs_processed_bucket
from utils.gcs import list_existing_objects

st.set_page_config(
    page_title="Social Media Optimizer — Live Scraper Monitor",
    page_icon="🎬",
    layout="wide",
)

st.title("🎬 Social Media Video Pipeline Monitor")
st.caption("Live tracking of short-form video scraping, 480p transcoding, and GCS ingestion.")

TOTAL_TARGETS = {
    "facebook": 4578,
    "instagram": 4875,
    "tiktok": 2137,
    "youtube": 1552,
}

bucket = gcs_processed_bucket()

@st.cache_data(ttl=5)
def fetch_gcs_counts():
    objs = list_existing_objects(bucket, prefix="videos/")
    counts = {
        "facebook": sum(1 for o in objs if "videos/facebook/" in o),
        "instagram": sum(1 for o in objs if "videos/instagram/" in o),
        "tiktok": sum(1 for o in objs if "videos/tiktok/" in o),
        "youtube": sum(1 for o in objs if "videos/youtube/" in o),
    }
    return counts, objs

counts, all_objs = fetch_gcs_counts()

total_meta = counts["facebook"] + counts["instagram"]
meta_target = TOTAL_TARGETS["facebook"] + TOTAL_TARGETS["instagram"]
meta_pct = total_meta / meta_target

col1, col2, col3, col4 = st.columns(4)
col1.metric("🔵 Facebook Reels", f"{counts[facebook]:,} / {TOTAL_TARGETS[facebook]:,}", f"{counts[facebook]/TOTAL_TARGETS[facebook]*100:.1f}%")
col2.metric("🟣 Instagram Reels", f"{counts[instagram]:,} / {TOTAL_TARGETS[instagram]:,}", f"{counts[instagram]/TOTAL_TARGETS[instagram]*100:.1f}%")
col3.metric("📊 Total Meta Ingested", f"{total_meta:,} / {meta_target:,}", f"{meta_pct*100:.1f}%")
col4.metric("☁️ GCS Bucket", bucket)

st.write("### 📈 Overall Meta Layer Progress")
st.progress(min(1.0, meta_pct))

col_fb, col_ig = st.columns(2)
with col_fb:
    st.write("#### 🔵 Facebook Progress")
    fb_pct = min(1.0, counts["facebook"] / TOTAL_TARGETS["facebook"])
    st.progress(fb_pct)
    st.write(f"**{counts[facebook]:,}** of **{TOTAL_TARGETS[facebook]:,}** videos uploaded ({fb_pct*100:.1f}%)")

with col_ig:
    st.write("#### 🟣 Instagram Progress")
    ig_pct = min(1.0, counts["instagram"] / TOTAL_TARGETS["instagram"])
    st.progress(ig_pct)
    st.write(f"**{counts[instagram]:,}** of **{TOTAL_TARGETS[instagram]:,}** videos uploaded ({ig_pct*100:.1f}%)")

st.divider()

st.write("### 📂 Platform Breakdown")
df_summary = pd.DataFrame([
    {"Platform": "Facebook", "Scraped in GCS": counts["facebook"], "Target Total": TOTAL_TARGETS["facebook"], "Completion": f"{counts[facebook]/TOTAL_TARGETS[facebook]*100:.1f}%"},
    {"Platform": "Instagram", "Scraped in GCS": counts["instagram"], "Target Total": TOTAL_TARGETS["instagram"], "Completion": f"{counts[instagram]/TOTAL_TARGETS[instagram]*100:.1f}%"},
    {"Platform": "TikTok", "Scraped in GCS": counts["tiktok"], "Target Total": TOTAL_TARGETS["tiktok"], "Completion": f"{counts[tiktok]/TOTAL_TARGETS[tiktok]*100:.1f}%"},
    {"Platform": "YouTube", "Scraped in GCS": counts["youtube"], "Target Total": TOTAL_TARGETS["youtube"], "Completion": f"{counts[youtube]/TOTAL_TARGETS[youtube]*100:.1f}%"},
])
st.dataframe(df_summary, use_container_width=True)

st.caption(f"Last refreshed: {datetime.now().strftime(%Y-%m-%d %H:%M:%S)} (Auto-refreshes periodically)")
time.sleep(5)
st.rerun()