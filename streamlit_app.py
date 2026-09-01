"""Streamlit UI for the Social Media Optimizer — a clean, Gemini-style landing.

Describe a video idea in words; every other piece of metadata is
auto-inferred from the description, then the app predicts whether to make it,
how it will perform, and a demo revenue figure.

Tabs:
  * ✨ Analyzer  — the landing page / single input.
  * 📊 Explore   — dataset exploration & findings.

The UI talks to the FastAPI backend (src/api/app.py). Run:
    ./run.sh all      # API on :8000, UI on :8501
"""

from __future__ import annotations

import os

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import requests
import streamlit as st

API_BASE = os.environ.get("SMO_API", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="Social Media Optimizer",
    page_icon="🏉",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

_CSS = """
<style>
    .block-container { max-width: 900px; padding-top: 1rem; }
    .hero { text-align: center; padding: 2rem 1rem 0.5rem; }
    .hero h1 { font-size: 2.4rem; font-weight: 600; margin-bottom: .1rem;
               background: linear-gradient(90deg,#10263f,#2b6cb0);
               -webkit-background-clip: text; -webkit-text-fill-color: transparent;}
    .hero .sub { color: #5f6368; font-size: 1.05rem; }
    .analyze-box { border: 1px solid #e0e0e0; border-radius: 28px; padding: 4px;
                   box-shadow: 0 2px 12px rgba(0,0,0,.06); }
    .verdict-banner { border-radius: 16px; padding: 1.1rem 1.4rem; color: #fff;
                      font-weight: 700; font-size: 1.5rem; }
    .verdict-msg { margin-top: .4rem; font-size: .95rem; font-weight: 400; opacity:.96; }
    .pill { display:inline-block; background:#f3f4f6; border:1px solid #e5e7eb;
            border-radius:999px; padding:2px 10px; margin:1px 2px; font-size:.8rem; color:#374151;}
    .metric-tile { background:#fafafa; border:1px solid #eee; border-radius:16px; padding:1rem; }
    [data-testid="stSidebar"] { background:#fbfbfb; }
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------


def _post(path: str, payload: dict):
    try:
        r = requests.post(f"{API_BASE}{path}", json=payload, timeout=90)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Could not reach the API at {API_BASE}{path}: {e}")
        st.info("Start it with  ./run.sh api")
        return None


@st.cache_data(ttl=300)
def _get_peers():
    try:
        r = requests.get(f"{API_BASE}/explore/peers", timeout=60)
        if r.status_code == 200:
            return r.json().get("peers", [])
    except requests.RequestException:
        pass
    return []


# ---------------------------------------------------------------------------
# Persistence helpers — single source of truth for the description text
# ---------------------------------------------------------------------------


def _ensure_desc_state():
    # Seed the description widget from `desc` only on first mount. From then on
    # the widget's own session-state (key="desc_input") is authoritative.
    if "desc_input" not in st.session_state:
        st.session_state["desc_input"] = st.session_state.get("desc", "")


def _set_desc(text: str, write_widget: bool = True):
    st.session_state["desc"] = text
    if write_widget:
        st.session_state["desc_input"] = text


def _current_desc() -> str:
    return str(st.session_state.get("desc_input", "")).strip()


# ---------------------------------------------------------------------------
# Result rendering
# ---------------------------------------------------------------------------


def _score_gauge(score: float):
    color = "#0f9d58" if score >= 65 else ("#f0a500" if score >= 45 else "#d0453b")
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": "/100", "font": {"size": 40}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#bbb"},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 45], "color": "#fdecea"},
                    {"range": [45, 65], "color": "#fff3d6"},
                    {"range": [65, 100], "color": "#e6f4ea"},
                ],
            },
        )
    )
    fig.update_layout(height=230, margin=dict(t=15, b=0, l=10, r=10))
    return fig


def render_results(d: dict):
    score = d["go_score"]
    vc = {"make": "#0f9d58", "borderline": "#f0a500", "skip": "#d0453b"}
    color = vc.get(d["verdict"], "#666")
    emoji = {"make": "✅", "borderline": "⚠️", "skip": "🚫"}.get(d["verdict"], "?")

    # --- Verdict banner ---
    st.markdown(
        f"<div class='verdict-banner' style='background:{color}'>"
        f"{emoji} {d['verdict'].upper()}"
        f"<div class='verdict-msg'>{d['verdict_message']}</div></div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns([1.1, 1, 1])
    with c1:
        st.plotly_chart(_score_gauge(score), width="stretch")
    est = d["estimates"]
    with c2:
        _metric_tile(
            "Typical views",
            f"{est['views']:,}",
            f"P25–P75: {est['views_range'][0]:,} – {est['views_range'][1]:,}",
            f"{d['views']['probability']*100:.0f}% likely HIGH",
        )
        _metric_tile(
            "Typical engagement",
            f"{est['engagement']:,}",
            f"P25–P75: {est['eng_range'][0]:,} – {est['eng_range'][1]:,}",
            f"{d['engagement']['probability']*100:.0f}% likely HIGH",
        )

    m = d["money"]
    with c3:
        _metric_tile("💵 Demo revenue", f"${m['revenue']:,.0f}",
                     "views × RPM/1000", f"Net ${m['net']:,.0f} after boost cost")
        roi = m.get("roi_percent")
        _metric_tile("ROI (demo)", f"{roi}%" if roi is not None else "n/a",
                     f"Boost cost ${m['boost_cost']:,.0f}",
                     f"Cost ${d.get('inferred', {}).get('cost', 0):,.0f}",
                     accent=True)

    # --- Auto-inferred metadata ---
    inf = d.get("inferred") or {}
    with st.expander("Auto-generated metadata — how the model read your description"):
        c = st.columns(4)
        c[0].markdown("**Platform**")
        c[0].write(inf.get("platform", "—"))
        c[1].markdown("**Brand/page**")
        c[1].write(inf.get("page", "—"))
        c[2].markdown("**Duration**")
        c[2].write(f"{inf.get('duration_seconds', '—')}s")
        c[3].markdown("**Suggested title**")
        c[3].write(inf.get("title", "—") or "—")
        st.markdown("**Theme(s)**", unsafe_allow_html=True)
        st.markdown(
            "".join(f"<span class='pill'>{t}</span>" for t in (inf.get("content_themes") or []))
            or "_none inferred_", unsafe_allow_html=True)
        st.markdown("**Format / access**", unsafe_allow_html=True)
        st.markdown(
            "".join(f"<span class='pill'>{t}</span>" for t in (inf.get("format_access") or []))
            or "_none inferred_", unsafe_allow_html=True)

    # --- Money disclaimer ---
    st.caption(
        "⚠️ Money is a demo: `cost_nzd` is empty in the data (0 / 11,306 rows), "
        "so revenue = views × RPM/1000 is a rule-of-thumb, not a trained model."
    )

    # --- Similar videos ---
    st.markdown("#### 🔎 Similar posts from history")
    sim = d.get("similar") or []
    if not sim:
        st.info("No similar posts indexed yet.")
    for s in sim[:4]:
        st.markdown(
            f"- **{s['title'] or '(untitled)'}** · `{s['platform']}/{s['page']}` · "
            f"**{s['views']:,.0f}** views · **{s['engagement']:,.0f}** engagement"
            f"  \n  _{s['description'][:120]}…_"
        )


def _metric_tile(label, value, sub, note, accent=False):
    st.markdown(
        f"<div class='metric-tile' style='margin-bottom:10px'>"
        f"<div style='font-size:.8rem;color:#5f6368'>{label}</div>"
        f"<div style='font-size:1.6rem;font-weight:700;"
        f"{'color:#2f6fed;' if accent else 'color:#202124;'}'>{value}</div>"
        f"<div style='font-size:.8rem;color:#80868b'>{sub}</div>"
        f"<div style='font-size:.75rem;color:#9aa0a6;margin-top:2px'>{note}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Analyzer tab (landing)
# ---------------------------------------------------------------------------


def tab_analyser():
    _ensure_desc_state()
    # Hero
    st.markdown(
        "<div class='hero'><h1>🏉 Social Media Optimizer</h1>"
        "<div class='sub'>Describe a video idea — everything else is inferred.</div></div>",
        unsafe_allow_html=True,
    )

    # Search-style input, Gemini-like
    with st.container():
        st.markdown("<div class='analyze-box'>", unsafe_allow_html=True)
        st.text_area(
            "Describe your video idea",
            height=70,
            placeholder="e.g. A huge try in the final minute that breaks the deadlock as the crowd erupts…",
            label_visibility="collapsed",
            key="desc_input",
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # Keep a mirror so Clear/`_set_desc` works even though the widget owns it.
        desc = _current_desc()

        # Buttons row: Analyze + Clear
        b1, b2, b3 = st.columns([1, 1, 3])
        analyse = b1.button("✨ Analyse", type="primary",
                            width="stretch", disabled=not desc)
        if b2.button("Clear", width="stretch"):
            _set_desc("")
            st.session_state.pop("last_result", None)
            st.rerun()
        b3.caption("")

    if analyse and desc:
        with st.spinner("Analysing your idea…"):
            d = _post("/predict", {"description": desc})
        if d:
            _set_desc(desc, write_widget=False)
            st.session_state["last_result"] = d
            st.markdown("---")
            render_results(d)
    elif "last_result" in st.session_state and desc:
        st.markdown("---")
        render_results(st.session_state["last_result"])


# ---------------------------------------------------------------------------
# Explore tab
# ---------------------------------------------------------------------------


def tab_explore():
    st.title("📊 Explore & Findings")
    st.caption(
        "A representative sample of historical posts from the model index — "
        "dive into what drove views and engagement."
    )

    try:
        df_full = pd.read_csv("data/processed/processed.csv")
    except Exception:
        df_full = pd.DataFrame()

    peers = _get_peers()
    if not peers:
        st.warning("No peer data available yet. Is the API running? (./run.sh api)")
        return

    df = pd.DataFrame(peers)
    st.markdown(f"**{len(df)} representative posts**")

    c1, c2, c3 = st.columns(3)
    if not df_full.empty:
        c1.metric("Median views", f"{df_full['views'].median():,.0f}")
        c2.metric("Median engagement", f"{df_full['engagement'].median():,.0f}")
        c3.metric("Platforms", ", ".join(sorted(df_full["platform"].dropna().unique().tolist())))

    col1, col2 = st.columns(2)
    hover_col = "content" if "content" in df else ("title" if "title" in df else "page")
    with col1:
        if "views" in df and "engagement" in df:
            hover = [c for c in [hover_col, "page"] if c in df]
            fig = px.scatter(
                df, x="views", y="engagement", color="platform",
                hover_data=hover, log_x=True, log_y=True,
                title="Views vs. Engagement (log scale)",
            )
            st.plotly_chart(fig, width="stretch")
    with col2:
        if "views" in df:
            fig = px.box(df, x="platform", y="views", log_y=True,
                         title="Views by platform")
            st.plotly_chart(fig, width="stretch")

    st.markdown("#### Sample posts")
    label_col = "content" if "content" in df else "title"
    cols = [c for c in [label_col, "page", "platform", "views", "engagement", "url"] if c in df]
    st.dataframe(df[cols].copy() if cols else df, width="stretch")


# ---------------------------------------------------------------------------
# app
# ---------------------------------------------------------------------------


def main():
    if "desc" not in st.session_state:
        st.session_state["desc"] = ""
    tab = st.tabs(["✨ Analyzer", "📊 Explore"])
    with tab[0]:
        tab_analyser()
    with tab[1]:
        tab_explore()


if __name__ == "__main__":
    main()
