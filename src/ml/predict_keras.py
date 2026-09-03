"""Keras model prediction logic for the FastAPI / Streamlit serving path.

Loads:
  - bundle_keras.joblib (FeaturePipeline + scaler + metadata + conditional stats)
  - keras_model.keras    (the 2-head Keras model)

Follows the same predict_raw() contract as src/ml/predict.py so the API can
drop it in as a drop-in replacement or gated alternative.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import joblib
import numpy as np
from scipy.spatial.distance import cdist

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BUNDLE_PATH = os.path.join(PROJECT_ROOT, "data", "models", "bundle_keras.joblib")
MODEL_PATH = os.path.join(PROJECT_ROOT, "data", "models", "keras_model.keras")
SIM_PATH = os.path.join(PROJECT_ROOT, "data", "models", "similar.joblib")

_bundle: Optional[dict] = None
_model = None
_similar: Optional[dict] = None
_embedder = None


def load():
    global _bundle, _model, _similar
    if _bundle is None and os.path.exists(BUNDLE_PATH) and os.path.exists(MODEL_PATH):
        import keras  # defer import — not everyone needs it

        _bundle = joblib.load(BUNDLE_PATH)
        _model = keras.models.load_model(MODEL_PATH)
        if os.path.exists(SIM_PATH):
            _similar = joblib.load(SIM_PATH)
    return _bundle is not None


def is_available() -> bool:
    return _bundle is not None and _model is not None


def _get_embedder():
    global _embedder
    if _embedder is None and _similar is not None:
        from sentence_transformers import SentenceTransformer

        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def build_raw(row: dict) -> dict:
    """Normalize an arbitrary request dict into pipeline input (same as predict.py)."""
    themes = row.get("content_theme") or row.get("content_themes") or []
    formats = row.get("format_access") or []
    tones = row.get("tone") or row.get("tones") or []
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
        "year": str(row.get("year", 2025) or 2025),
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
        # Extra fields for full feature pipeline matching describe_rob.ipynb & reference.md
        "people": row.get("people") or [],
        "brands": row.get("brands") or [],
        "event": row.get("event") or row.get("events") or [],
        "context": row.get("context") or [],
        "overall_team": row.get("overall_team") or [],
        "audio_format": row.get("audio_format") or [],
    }



def _infer_hashtags_mentions_emojis(raw: dict) -> tuple:
    """Extract hashtag/mention/emoji tokens from the description text."""
    import regex
    from src.ml.features import HASHTAG_PATTERN, MENTION_PATTERN, EMOJI_PATTERN

    text = f"{raw['title']} {raw['description']}".strip()
    hashtags = [m.casefold() for m in HASHTAG_PATTERN.findall(text)]
    mentions = [m.casefold().rstrip(".") for m in MENTION_PATTERN.findall(text)]

    def _all_emojis(t):
        return EMOJI_PATTERN.findall(t)

    emojis = _all_emojis(text)
    return hashtags, mentions, emojis


def predict_raw(raw: dict) -> dict:
    """Predict using the Keras 2-head model.

    Returns the same contract dict as predict.py's predict_raw().
    """
    if not is_available():
        raise RuntimeError("Keras model not loaded — call load() first")

    b = _bundle
    pipe = b["pipe"]
    scaler_X = b["scaler_X"]
    numeric_cols = b.get("numeric_cols", [])

    # Build base row for the pipeline
    text_content = f"{raw['title']} {raw['description']}".strip()
    pipe_row = {
        "platform": raw["platform"],
        "page": raw["page"],
        "year": str(raw["year"]),
        "category_l0": raw["category_l0"],
        "category_l1": raw["category_l1"],
        "category_l2": raw["category_l2"],
        "content": text_content,  # FeaturePipeline expects 'content' column for hashtag/mention/emoji extraction
        "description_json": _build_description_json(raw, pipe._json_fields),
        "duration_seconds": raw["duration_seconds"],
    }

    # Infer hashtags, mentions, emojis from description text
    hashtags, mentions, emojis = _infer_hashtags_mentions_emojis(raw)
    # These are already captured in the description_json counts via the FeaturePipeline
    pipe_row["n_hashtags"] = len(hashtags)
    pipe_row["n_mentions"] = len(mentions)
    pipe_row["n_emojis"] = len(emojis)

    # Transform
    import pandas as pd

    df = pd.DataFrame([pipe_row])
    X_sparse = pipe.transform(df)
    x = X_sparse.toarray().astype(np.float32)[0]

    # Scale numeric columns
    if numeric_cols:
        numeric_indices = [i for i, c in enumerate(pipe.base_columns) if c in numeric_cols]
        binary_indices = [i for i in range(x.shape[0]) if i not in numeric_indices]
        if numeric_indices:
            x_num = scaler_X.transform(x[numeric_indices].reshape(1, -1))[0]
            x[numeric_indices] = x_num

# Platform baseline view and engagement distributions
PLATFORM_BASELINES = {
    "IG": {"views_mult": 1.15, "eng_mult": 1.45, "views_base": 380000, "eng_base": 22000},
    "TT": {"views_mult": 0.90, "eng_mult": 0.95, "views_base": 290000, "eng_base": 14000},
    "FB": {"views_mult": 1.25, "eng_mult": 0.45, "views_base": 420000, "eng_base": 6500},
    "YT": {"views_mult": 0.60, "eng_mult": 0.35, "views_base": 180000, "eng_base": 4200},
}


def _calc_platform_compatibility(
    platform: str,
    duration: float,
    themes: list,
    formats: list,
    tones: list,
    audios: list,
) -> tuple[float, float]:
    """Calculate platform-specific view and engagement compatibility modifiers."""
    p = platform.upper()
    dur_mod = 0.0
    fmt_mod = 0.0

    all_tags = set(themes + formats)

    if p == "TT":
        # TikTok: strong preference for ultra-short (<20s), high-energy, candid/challenges
        if duration <= 18.0:
            dur_mod += 0.08
        elif duration <= 30.0:
            dur_mod += 0.02
        elif duration > 50.0:
            dur_mod -= 0.20
        elif duration > 35.0:
            dur_mod -= 0.10

        if any(f in ("candid clip", "challenges", "reaction", "highlight", "rugby_skills") for f in all_tags):
            fmt_mod += 0.07
        if any(f in ("interview", "archive", "ticket sales", "announcement", "player story") for f in all_tags):
            fmt_mod -= 0.14
        if any(a in ("song", "music", "ambient") for a in audios):
            fmt_mod += 0.04
        if any(t in ("excitement", "humour", "lighthearted") for t in tones):
            fmt_mod += 0.04

    elif p == "IG":
        # Instagram: Reels sweet spot 12s-35s, high visual quality, highlights, celebrations
        if 12.0 <= duration <= 35.0:
            dur_mod += 0.06
        elif duration > 60.0:
            dur_mod -= 0.12

        if any(f in ("highlight", "celebration", "rugby_skills", "behind-the-scenes", "candid clip") for f in all_tags):
            fmt_mod += 0.08
        if any(t in ("excitement", "pride", "wholesome") for t in tones):
            fmt_mod += 0.05

    elif p == "YT":
        # YouTube: rewards longer duration (30s-90s+), depth, interviews, analysis, archival montages
        if duration >= 35.0:
            dur_mod += 0.12
        elif duration <= 14.0:
            dur_mod -= 0.10

        if any(f in ("interview", "behind-the-scenes", "montage", "player story", "archive", "training") for f in all_tags):
            fmt_mod += 0.12
        if any(a == "commentary" for a in audios):
            fmt_mod += 0.06
        if any(f in ("ticket sales", "promotion") for f in all_tags):
            fmt_mod -= 0.06

    elif p == "FB":
        # Facebook: broad duration tolerance (15s-60s), high performance on historic rugby rivalry, haka, classic tries
        if 15.0 <= duration <= 60.0:
            dur_mod += 0.04
        if any(f in ("haka", "rivalry", "archive", "try", "celebration", "player story") for f in all_tags):
            fmt_mod += 0.09
        if any(t in ("pride", "nostalgia", "tension", "solemn") for t in tones):
            fmt_mod += 0.06

    return dur_mod, fmt_mod


def _explain_platform_fit(
    platform: str,
    go_score: float,
    duration: float,
    themes: list,
    formats: list,
    tones: list,
    audios: list,
    people: list,
) -> str:
    """Generate detailed human-understandable explanation of why this idea fits or misses on this platform."""
    p = platform.upper()
    all_tags = set(themes + formats)
    reasons_pos = []
    reasons_neg = []

    if p == "IG":
        if 12.0 <= duration <= 35.0:
            reasons_pos.append(f"Optimal Reels runtime ({duration:.0f}s within the 12s–35s sweet spot)")
        elif duration > 50.0:
            reasons_neg.append(f"Duration of {duration:.0f}s exceeds ideal short-form Reels engagement window")
        
        if any(f in ("highlight", "celebration", "rugby_skills") for f in all_tags):
            reasons_pos.append("High visual appeal and fast-paced highlight mechanics match Instagram viewer behaviour")
        if any(t in ("excitement", "pride", "wholesome") for t in tones):
            reasons_pos.append("High-energy positive emotion drives saves and story shares")
        if "none" in audios:
            reasons_neg.append("Silent/no audio penalizes discovery in Instagram Reels feed")

    elif p == "TT":
        if duration <= 20.0:
            reasons_pos.append(f"Snappy {duration:.0f}s pacing captures high initial completion rate on TikTok For You feed")
        elif duration > 35.0:
            reasons_neg.append(f"Duration of {duration:.0f}s is longer than TikTok's high-retention threshold")
        
        if any(f in ("candid clip", "challenges", "reaction", "rugby_skills") for f in all_tags):
            reasons_pos.append("Authentic, candid content and raw skills excel with TikTok algorithms")
        if any(a in ("song", "music", "ambient") for a in audios):
            reasons_pos.append("Active audio/music backing strongly boosts algorithmic distribution")
        if any(f in ("interview", "announcement", "ticket sales") for f in all_tags):
            reasons_neg.append("Formal broadcast or commercial tone underperforms casual UGC on TikTok")

    elif p == "FB":
        if 15.0 <= duration <= 60.0:
            reasons_pos.append(f"Balanced {duration:.0f}s video duration matches Facebook feed viewing patterns")
        if any(f in ("haka", "rivalry", "archive", "try", "player story") for f in all_tags):
            reasons_pos.append("Historic rivalry, Haka, and heritage storylines drive massive comment debates and community shares on Facebook")
        if any(t in ("pride", "nostalgia", "tension") for t in tones):
            reasons_pos.append("Heritage pride and rivalry tension resonate strongly with core rugby fans")

    elif p == "YT":
        if duration >= 35.0:
            reasons_pos.append(f"Extended {duration:.0f}s duration rewards YouTube watch time algorithms")
        elif duration <= 15.0:
            reasons_neg.append(f"Short {duration:.0f}s clip limits long-form watch time unless packaged strictly as Shorts")
        
        if any(f in ("interview", "behind-the-scenes", "montage", "player story", "archive") for f in all_tags):
            reasons_pos.append("In-depth storytelling and behind-the-scenes access generate high subscriber retention on YouTube")
        if any(a in ("voice", "commentary") for a in audios):
            reasons_pos.append("Verbal commentary and dialogue provide strong context for search indexing")

    if any(pe in ("all blacks", "black ferns", "beauden barrett", "ardie savea", "will jordan") for pe in people):
        reasons_pos.append("Marquee player/team recognition acts as a proven click-through catalyst")

    if go_score >= 65:
        summary = "Strong strategic fit: " + " · ".join(reasons_pos[:2])
    elif go_score >= 45:
        pos_txt = " · ".join(reasons_pos[:1]) if reasons_pos else "Standard baseline appeal"
        neg_txt = " · ".join(reasons_neg[:1]) if reasons_neg else "Needs tighter tag optimization"
        summary = f"Moderate fit: {pos_txt}. Opportunity: {neg_txt}"
    else:
        neg_txt = " · ".join(reasons_neg[:2]) if reasons_neg else "Lacks distinct platform hook and high-energy tags"
        summary = f"Weak fit: {neg_txt}"

    return summary


def predict_raw(raw: dict) -> dict:

    """Predict using the Keras 2-head model with platform-specific scaling.

    Returns the same contract dict as predict.py's predict_raw().
    """
    if not is_available():
        raise RuntimeError("Keras model not loaded — call load() first")

    b = _bundle
    pipe = b["pipe"]
    scaler_X = b["scaler_X"]
    numeric_cols = b.get("numeric_cols", [])

    # Build base row for the pipeline
    text_content = f"{raw['title']} {raw['description']}".strip()
    pipe_row = {
        "platform": raw["platform"],
        "page": raw["page"],
        "year": str(raw["year"]),
        "category_l0": raw["category_l0"],
        "category_l1": raw["category_l1"],
        "category_l2": raw["category_l2"],
        "content": text_content,
        "description_json": _build_description_json(raw, pipe._json_fields),
        "duration_seconds": raw["duration_seconds"],
    }

    hashtags, mentions, emojis = _infer_hashtags_mentions_emojis(raw)
    pipe_row["n_hashtags"] = len(hashtags)
    pipe_row["n_mentions"] = len(mentions)
    pipe_row["n_emojis"] = len(emojis)

    import pandas as pd

    df = pd.DataFrame([pipe_row])
    X_sparse = pipe.transform(df)
    x = X_sparse.toarray().astype(np.float32)[0]

    # Scale numeric columns
    if numeric_cols:
        numeric_indices = [i for i, c in enumerate(pipe.base_columns) if c in numeric_cols]
        if numeric_indices:
            x_num = scaler_X.transform(x[numeric_indices].reshape(1, -1))[0]
            x[numeric_indices] = x_num

    # Predict via Keras
    pred_out = _model.predict(x.reshape(1, -1), verbose=0)
    proba_views = pred_out[0][0]  # shape (1, 2)
    proba_eng = pred_out[1][0]  # shape (1, 2)

    # Extract all inputs for continuous scoring
    themes = raw.get("content_theme") or raw.get("content_themes") or []
    formats = raw.get("format_access") or []
    tones = raw.get("tone") or raw.get("tones") or []
    audios = raw.get("audio_format") or []
    people = raw.get("people") or []
    brands = raw.get("brands") or []
    plat = str(raw.get("platform", "FB")).upper()
    desc_text = str(raw.get("description", ""))

    # 1. Platform & duration compatibility curve
    dur_mod, fmt_mod = _calc_platform_compatibility(
        plat, float(raw.get("duration_seconds", 20.0)), themes, formats, tones, audios
    )

    # 2. Text richness & description story depth
    text_len = len(desc_text.strip())
    if text_len >= 140:
        story_mod = 0.05
    elif text_len >= 60:
        story_mod = 0.02
    elif text_len >= 25:
        story_mod = 0.0
    else:
        story_mod = -0.06

    # 3. Entity & Star Power modifier
    entity_mod = 0.0
    if any(p in ("all blacks", "black ferns", "beauden barrett", "ardie savea", "will jordan", "rieo ioane") for p in people):
        entity_mod += 0.04
    if any(b in ("adidas", "ineos", "altrad", "red bull") for b in brands):
        entity_mod += 0.02

    # 4. Content Theme & Format Synergy
    theme_mod = 0.0
    if any(t in ("try", "celebration", "rugby_skills", "haka") for t in themes):
        theme_mod += 0.06
    if any(t in ("training", "challenges") for t in themes):
        theme_mod += 0.02
    if any(t in ("player story", "rivalry") for t in themes):
        theme_mod += 0.03
    if any(f in ("announcement", "squad naming") for f in formats + themes):
        theme_mod -= 0.04
    if any(f in ("ticket sales", "promotion") for f in formats):
        theme_mod -= 0.08
    if "non rugby related" in themes:
        theme_mod -= 0.42

    # 5. Tone Modifier
    tone_mod = 0.0
    if any(t in ("excitement", "pride") for t in tones):
        tone_mod += 0.04
    elif any(t in ("humour", "wholesome", "tension") for t in tones):
        tone_mod += 0.02
    elif any(t in ("lighthearted",) for t in tones):
        tone_mod += 0.01
    elif any(t in ("solemn", "sadness") for t in tones):
        if not any(t in ("try", "celebration", "haka") for t in themes):
            tone_mod -= 0.22
        else:
            tone_mod -= 0.05

    # 6. Audio Format Modifier
    audio_mod = 0.0
    if any(a in ("ambient", "song", "voice") for a in audios):
        audio_mod += 0.02
    elif "none" in audios:
        audio_mod -= 0.12

    # Combine continuous signals with model probability
    p_views = float(np.clip(proba_views[1] + dur_mod + fmt_mod + story_mod + entity_mod + theme_mod + audio_mod, 0.05, 0.98))
    p_eng = float(np.clip(proba_eng[1] + dur_mod * 0.8 + fmt_mod * 1.1 + story_mod * 0.8 + entity_mod * 1.2 + theme_mod + tone_mod + audio_mod, 0.05, 0.98))

    v_high = p_views >= 0.5
    e_high = p_eng >= 0.5

    # Conditional stats for anchoring estimates
    cond = b.get("conditional_stats", {})
    v_stats = cond.get("views_high" if v_high else "views_low", {})
    e_stats = cond.get("eng_high" if e_high else "eng_low", {})
    bucket_v = v_stats.get("median") or b.get("views_median", 0)
    bucket_e = e_stats.get("median") or b.get("eng_median", 0)

    # Scale estimates by platform-specific baseline and probability
    plat_base = PLATFORM_BASELINES.get(plat, PLATFORM_BASELINES["FB"])
    views_est = int(bucket_v * plat_base["views_mult"] * (p_views / 0.75))
    eng_est = int(bucket_e * plat_base["eng_mult"] * (p_eng / 0.75))

    # Go-score (continuous 0-100 spectrum)
    go_score = round(100.0 * (0.55 * p_views + 0.45 * p_eng), 1)

    verdict, message = _verdict(go_score, p_views, p_eng)




    # Input-strength / confidence
    signal = int(len(raw.get("description") or "")) + int(len(raw.get("title") or ""))
    signal += 40 * len(raw.get("content_theme") or [])
    signal += 40 * len(raw.get("format_access") or [])
    if signal >= 120:
        confidence = "high"
        confidence_note = "Good descriptive signal for the model."
    elif signal >= 50:
        confidence = "medium"
        confidence_note = "Decent signal — more text/themes would tighten the estimate."
    else:
        confidence = "low"
        confidence_note = "Very little input — result leans on defaults. Add description and themes for a trustworthy read."

    # Money (demo)
    rpm = raw.get("expected_rpm", 3.0)
    cpm = raw.get("expected_cpm", 5.0)
    revenue = views_est * rpm / 1000.0
    boost_cost = (views_est / 1000.0) * cpm * 0.4
    net = revenue - boost_cost - raw.get("cost", 0)
    roi = (net / raw["cost"] * 100.0) if raw.get("cost", 0) and raw["cost"] > 0 else float("nan")

    similar = _similar_videos(raw, k=5)

    fit_explanation = _explain_platform_fit(
        plat, go_score, float(raw.get("duration_seconds", 20.0)), themes, formats, tones, audios, people
    )

    return {
        "go_score": go_score,
        "verdict": verdict,
        "verdict_message": message,
        "fit_explanation": fit_explanation,
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
        "model_metrics": b.get("metrics", {}),
        "model_type": "keras_2head",
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
        out.append({
            "title": str(r.get("content", "")),
            "description": str(r.get("description", "")),
            "platform": str(r.get("platform", "")),
            "page": str(r.get("page", "")),
            "views": float(r.get("views", 0) or 0),
            "engagement": float(r.get("engagement", 0) or 0),
            "url": str(r.get("url", "")),
            "distance": float(dists[int(i)]),
        })
    return out


def peers_for_explore() -> list:
    if not is_available():
        return []
    if _similar is None:
        return []
    return _similar["peers"]


def _build_description_json(row: dict, fields: list) -> str:
    """Build description_json string from user-supplied fields matching describe_rob & reference.md."""
    import json

    payload: dict = {
        "play_by_play": str(row.get("description", "") or ""),
        "content_theme": row.get("content_theme") or row.get("content_themes") or [],
        "format_access": row.get("format_access") or [],
        "people": row.get("people") or [],
        "brands": row.get("brands") or [],
        "event": row.get("event") or row.get("events") or [],
        "tone": row.get("tone") or row.get("tones") or [],
        "context": row.get("context") or [],
        "overall_team": row.get("overall_team") or ["men"],
        "audio_format": row.get("audio_format") or ["ambient"],
    }
    return json.dumps(payload)



# ---------------------------------------------------------------------------
# Cross-platform prediction
# ---------------------------------------------------------------------------

PLATFORMS = ["FB", "IG", "TT", "YT"]


def predict_multi(raw: dict) -> dict:
    """Score the same idea across all platforms.

    Returns the base platform result plus a `platforms` map, a
    `platform_leaderboard`, a `best_platform` recommendation,
    and feature-level explanations.
    """
    base = predict_with_explain(raw)
    per_platform = {}
    for plat in PLATFORMS:
        row = dict(raw)
        row["platform"] = plat
        per_platform[plat] = predict_raw(row)

    # Find best platform by go_score
    scores = {p: per_platform[p]["go_score"] for p in PLATFORMS}
    best = max(scores, key=scores.get)

    # Build a leaderboard
    leaderboard = sorted(
        [{"platform": p, "go_score": per_platform[p]["go_score"],
          "verdict": per_platform[p]["verdict"],
          "views_p": per_platform[p]["views"]["probability"],
          "eng_p": per_platform[p]["engagement"]["probability"],
          "estimates": per_platform[p]["estimates"]}
         for p in PLATFORMS],
        key=lambda x: x["go_score"], reverse=True,
    )

    base["platforms"] = {p: per_platform[p] for p in PLATFORMS}
    base["platform_leaderboard"] = leaderboard
    base["best_platform"] = best
    return base


# ---------------------------------------------------------------------------
# Explainability — feature-level contributions
# ---------------------------------------------------------------------------


def _get_feature_contributions(raw: dict) -> dict:
    """Compute which feature groups drive the prediction up or down.

    Uses a fast perturbation approach: for each feature group, measure the
    delta between the current prediction and a baseline with that group zeroed.
    """
    if not is_available():
        return {}

    b = _bundle
    pipe = b["pipe"]
    scaler_X = b["scaler_X"]
    numeric_cols = b.get("numeric_cols", [])

    # Build the input row
    text_content = f"{raw['title']} {raw['description']}".strip()
    pipe_row = {
        "platform": raw["platform"],
        "page": raw["page"],
        "year": str(raw["year"]),
        "category_l0": raw["category_l0"],
        "category_l1": raw["category_l1"],
        "category_l2": raw["category_l2"],
        "content": text_content,
        "description_json": _build_description_json(raw, pipe._json_fields),
        "duration_seconds": raw["duration_seconds"],
    }
    hashtags, mentions, emojis = _infer_hashtags_mentions_emojis(raw)
    pipe_row["n_hashtags"] = len(hashtags)
    pipe_row["n_mentions"] = len(mentions)
    pipe_row["n_emojis"] = len(emojis)

    import pandas as pd
    df = pd.DataFrame([pipe_row])
    X_sparse = pipe.transform(df)
    x = X_sparse.toarray().astype(np.float32)[0]

    if numeric_cols:
        numeric_indices = [i for i, c in enumerate(pipe.base_columns) if c in numeric_cols]
        if numeric_indices:
            x_num = scaler_X.transform(x[numeric_indices].reshape(1, -1))[0]
            x[numeric_indices] = x_num

    # Current prediction
    current_out = _model.predict(x.reshape(1, -1), verbose=0)
    current_views = float(current_out[0][0][1])
    current_eng = float(current_out[1][0][1])

    # Define feature group indices
    n_cat = len(pipe.cat_columns)
    n_json = len(pipe.json_vocab)
    n_base = len(pipe.base_columns)
    cat_indices = list(range(0, n_cat))
    json_indices = list(range(n_cat, n_cat + n_json))
    base_indices = list(range(n_cat + n_json, n_cat + n_json + n_base))

    def _group_contrib(indices):
        x_ablate = x.copy()
        x_ablate[indices] = 0.0
        ablate_out = _model.predict(x_ablate.reshape(1, -1), verbose=0)
        dv = current_views - float(ablate_out[0][0][1])
        de = current_eng - float(ablate_out[1][0][1])
        return dv, de

    # Per-group ablation
    groups_v = {}
    groups_e = {}
    for name, indices, col_names in [
        ("platform/page", cat_indices, pipe.cat_columns),
        ("themes/formats", json_indices, pipe.json_columns),
        ("duration/counts", base_indices, pipe.base_columns),
    ]:
        dv, de = _group_contrib(indices)
        groups_v[name] = {"total": round(dv, 4), "active": len(indices)}
        groups_e[name] = {"total": round(de, 4), "active": len(indices)}

    # Also check which specific features in themes/formats are active
    active_themes = []
    for idx, col_name in zip(json_indices, pipe.json_columns):
        if abs(x[idx]) > 0.001:
            label = col_name.replace("json_", "").replace("__", ": ")
            active_themes.append({"feature": label, "value": 1.0})
    active_cats = []
    for idx, col_name in zip(cat_indices, pipe.cat_columns):
        if abs(x[idx]) > 0.001:
            active_cats.append({"feature": col_name, "value": 1.0})

    return {
        "views": {
            "groups": groups_v,
            "current_prob": round(current_views, 4),
            "active_themes": active_themes,
            "active_categories": active_cats,
        },
        "engagement": {
            "groups": groups_e,
            "current_prob": round(current_eng, 4),
            "active_themes": active_themes,
            "active_categories": active_cats,
        },
    }


def predict_with_explain(raw: dict) -> dict:
    """Predict and return the standard result plus feature-level explanations."""
    result = predict_raw(raw)
    result["explanation"] = _get_feature_contributions(raw)
    return result