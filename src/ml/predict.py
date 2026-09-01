"""Prediction logic used by the FastAPI app and reused by the Streamlit preview.

Loads data/models/bundle.joblib (feature pipeline, classifiers, regressors,
bin thresholds) and data/models/similar.joblib (semantic embeddings + peers).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
from scipy.spatial.distance import cdist

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BUNDLE_PATH = os.path.join(PROJECT_ROOT, "data", "models", "bundle.joblib")
SIM_PATH = os.path.join(PROJECT_ROOT, "data", "models", "similar.joblib")

_bundle: Optional[dict] = None
_similar: Optional[dict] = None
_embedder = None


def load():
    global _bundle, _similar
    if _bundle is None:
        _bundle = joblib.load(BUNDLE_PATH)
        if os.path.exists(SIM_PATH):
            _similar = joblib.load(SIM_PATH)
    return _bundle


def _get_embedder():
    global _embedder
    if _embedder is None and _similar is not None:
        from sentence_transformers import SentenceTransformer

        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


# ---------------------------------------------------------------------------
# Core prediction
# ---------------------------------------------------------------------------


def build_raw(row: dict) -> dict:
    """Normalize an arbitrary request dict (from API or UI) into pipeline input."""
    themes = row.get("content_themes") or []
    formats = row.get("format_access") or []
    tones = row.get("tones") or []
    if isinstance(themes, str):
        themes = [themes]
    if isinstance(formats, str):
        formats = [formats]
    if isinstance(tones, str):
        tones = [tones]
    return {
        "title": str(row.get("title", "") or ""),
        "description": str(row.get("description", "") or ""),
        "platform": str(row.get("platform", "FB") or "FB"),
        "page": str(row.get("page", "All Blacks") or "All Blacks"),
        "year": int(row.get("year", 2025) or 2025),
        "category_l0": str(row.get("category_l0") or "No Hashtag"),
        "category_l1": str(row.get("category_l1") or "No Hashtag"),
        "category_l2": str(row.get("category_l2") or "No Hashtag"),
        "duration_seconds": float(row.get("duration_seconds", 20) or 20),
        "content_theme": themes,
        "format_access": formats,
        "tone": tones,
        "cost": float(row.get("cost", 0) or 0),
        "expected_rpm": float(row.get("expected_rpm", 3.0) or 3.0),
        "expected_cpm": float(row.get("expected_cpm", 5.0) or 5.0),
    }


def predict_raw(raw: dict) -> dict:
    load()
    b = _bundle
    pipe = b["pipe"]
    x = pipe.transform_row(raw)

    p_views = float(b["clf_views"].predict_proba(x.reshape(1, -1))[0][1])
    p_eng = float(b["clf_eng"].predict_proba(x.reshape(1, -1))[0][1])

    v_high = p_views >= 0.5
    e_high = p_eng >= 0.5

    # Continuous estimates from the regressors (these respond to the input),
    # softly anchored to the historical median of the predicted bucket so an
    # outlier regression value can't show something absurd.
    cond = b.get("conditional_stats", {})
    v_stats = cond.get("views_high" if v_high else "views_low", {})
    e_stats = cond.get("eng_high" if e_high else "eng_low", {})
    bucket_v = v_stats.get("median") or b.get("views_median", 0)
    bucket_e = e_stats.get("median") or b.get("eng_median", 0)

    raw_views = float(np.expm1(b["reg_views"].predict(x.reshape(1, -1))[0]))
    raw_eng = float(np.expm1(b["reg_eng"].predict(x.reshape(1, -1))[0]))

    def _anchor(pred: float, bucket: float, w: float = 0.35) -> float:
        # blend the regression point-prediction toward the bucket median to
        # dampen tails, then keep it within a sane +/- factor of the bucket
        return max(0.25 * bucket, min(4.0 * bucket, (1 - w) * pred + w * bucket))

    views_est = _anchor(raw_views, bucket_v)
    eng_est = _anchor(raw_eng, bucket_e)

    # Range = anchored estimate ± half the bucket's IQR
    v_p25, v_p75 = (v_stats.get("p25") or 0), (v_stats.get("p75") or 0)
    e_p25, e_p75 = (e_stats.get("p25") or 0), (e_stats.get("p75") or 0)
    v_half = (v_p75 - v_p25) / 2
    e_half = (e_p75 - e_p25) / 2

    go_score = round(100.0 * (0.55 * p_views + 0.45 * p_eng), 1)

    verdict, message = _verdict(go_score, p_views, p_eng)

    # Input-strength / confidence: very thin inputs should not over-claim.
    signal = int(len(raw.get("description") or "")) + int(len(raw.get("title") or ""))
    signal += 40 * len(raw.get("content_theme") or [])
    signal += 40 * len(raw.get("format_access") or [])
    if signal >= 120:
        confidence = "high"
        confidence_note = "Good descriptive signal for the model."
    elif signal >= 50:
        confidence = "medium"
        confidence_note = "Decent signal — more descriptive text/themes would tighten the estimate."
    else:
        confidence = "low"
        confidence_note = "Very little input — the result leans on platform/page defaults. Add a description and themes for a trustworthy read."

    # Money (DEMO): ad revenue from views via RPM, minus paid-boost cost via CPM
    rpm = raw["expected_rpm"]
    cpm = raw["expected_cpm"]
    revenue = views_est * rpm / 1000.0
    boost_cost = (views_est / 1000.0) * cpm * 0.4  # hypothetical 40% paid reach
    net = revenue - boost_cost - raw["cost"]
    roi = (net / raw["cost"] * 100.0) if raw["cost"] and raw["cost"] > 0 else float("nan")

    similar = _similar_videos(raw, k=5)

    return {
        "go_score": go_score,
        "verdict": verdict,
        "verdict_message": message,
        "confidence": confidence,
        "confidence_note": confidence_note,
        "views": {
            "target": "views",
            "label": "High" if v_high else "Low",
            "probability": round(p_views, 4),
            "is_high": v_high,
        },
        "engagement": {
            "target": "engagement",
            "label": "High" if e_high else "Low",
            "probability": round(p_eng, 4),
            "is_high": e_high,
        },
        "estimates": {
            "views": round(views_est),
            "engagement": round(eng_est),
            "views_range": [round(max(0, views_est - v_half)), round(views_est + v_half)],
            "eng_range": [round(max(0, eng_est - e_half)), round(eng_est + e_half)],
        },
        "money": {
            "revenue": round(revenue, 2),
            "boost_cost": round(boost_cost, 2),
            "net": round(net, 2),
            "roi_percent": (round(roi, 1) if not np.isnan(roi) else None),
            "is_demo": True,
        },
        "similar": similar,
        "model_metrics": b["metrics"],
    }


def _verdict(go: float, pv: float, pe: float) -> tuple:
    if go >= 65:
        return "make", (
            "Strong signal: likely to outperform the median in both reach and "
            "engagement. Green-light."
        )
    if go >= 45:
        return "borderline", (
            "Mixed signal. Some risk — adjust format/theme/production before "
            "committing budget."
        )
    return "skip", (
        "Weak signal relative to historical posts. Low predicted reach and "
        "engagement; reconsider the concept or run a cheaper test."
    )


def _similar_videos(raw: dict, k: int = 5) -> list:
    if _similar is None:
        return []
    emb = _get_embedder()
    if emb is None:
        return []
    text = f"{raw['title']} {raw['description']}".strip()
    if not text:
        return []
    q = emb.encode([text])[0]
    all_emb = _similar["embeddings"]
    dists = cdist([q], all_emb)[0]
    order = np.argsort(dists)[:k]
    rows = _similar["rows"]
    out = []
    for i in order:
        r = rows[int(i)]
        out.append(
            {
                "title": str(r.get("content", "")),
                "description": str(r.get("description", "")),
                "platform": str(r.get("platform", "")),
                "page": str(r.get("page", "")),
                "views": float(r.get("views", 0) or 0),
                "engagement": float(r.get("engagement", 0) or 0),
                "url": str(r.get("url", "")),
                "distance": float(dists[int(i)]),
            }
        )
    return out


def peers_for_explore() -> List[dict]:
    load()
    if _similar is None:
        return []
    return _similar["peers"]
