"""Unified prediction and evaluation service for the Streamlit UI and API."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from src.db.database import save_prediction
from src.ml import infer as inferlib
from src.ml import predict as predict_xgb
from src.ml import predict_keras

logger = logging.getLogger(__name__)

_MODELS_INITIALIZED = False
ALL_PLATFORMS = ["FB", "IG", "TT", "YT"]


def init_models() -> None:
    """Load prediction models once in memory."""
    global _MODELS_INITIALIZED
    if not _MODELS_INITIALIZED:
        try:
            predict_keras.load()
        except Exception as e:
            logger.warning("Could not load Keras model: %s", e)
        try:
            predict_xgb.load()
        except Exception as e:
            logger.warning("Could not load XGBoost bundle: %s", e)
        _MODELS_INITIALIZED = True


def infer_metadata(description: str) -> Dict[str, Any]:
    """Auto-infer initial metadata from free-text video description."""
    meta = inferlib.infer(description)
    # Default to ALL platforms
    meta["platform"] = "ALL"
    return meta


def _predict_single(norm_row: Dict[str, Any]) -> Dict[str, Any]:
    """Predict for a single concrete platform (FB, IG, TT, YT)."""
    if predict_keras.is_available():
        try:
            return predict_keras.predict_raw(norm_row)
        except Exception as e:
            logger.warning("Keras predict_raw failed: %s, falling back to XGB", e)
    return predict_xgb.predict_raw(norm_row)


def run_prediction_pipeline(
    raw_input: Dict[str, Any], save_to_db: bool = True
) -> Dict[str, Any]:
    """Execute model prediction across single or all platforms, peer benchmarks, and SQLite persistence."""
    init_models()

    target_platform = raw_input.get("platform", "ALL")
    is_all = target_platform in ("ALL", "", None)

    # Normalize base row using builder
    if predict_keras.is_available():
        base_norm = predict_keras.build_raw(raw_input)
    else:
        base_norm = predict_xgb.build_raw(raw_input)

    # Evaluate across all 4 platforms
    platforms_results: Dict[str, Dict[str, Any]] = {}
    leaderboard = []

    for p in ALL_PLATFORMS:
        p_row = dict(base_norm)
        p_row["platform"] = p
        p_res = _predict_single(p_row)
        platforms_results[p] = p_res
        leaderboard.append({
            "platform": p,
            "go_score": p_res.get("go_score", 50.0),
            "verdict": p_res.get("verdict", "borderline"),
            "verdict_message": p_res.get("verdict_message", ""),
            "fit_explanation": p_res.get("fit_explanation", ""),
            "views_p": (p_res.get("views") or {}).get("probability", 0.5),
            "eng_p": (p_res.get("engagement") or {}).get("probability", 0.5),
            "estimates": p_res.get("estimates", {}),
            "similar": p_res.get("similar", []),
        })


    leaderboard.sort(key=lambda x: x["go_score"], reverse=True)
    best_plat = leaderboard[0]["platform"] if leaderboard else "FB"

    if is_all:
        # Cross-platform composite view
        best_res = platforms_results[best_plat]
        avg_score = round(sum(r["go_score"] for r in leaderboard) / max(1, len(leaderboard)), 1)
        
        # Primary result incorporates composite scores + all platform details
        result = dict(best_res)
        result["is_all_platforms"] = True
        result["selected_platform"] = "ALL"
        result["composite_go_score"] = avg_score
        result["best_platform"] = best_plat
        result["platforms"] = platforms_results
        result["platform_leaderboard"] = leaderboard
    else:
        # User specified a single platform
        selected_res = platforms_results.get(target_platform, platforms_results[best_plat])
        result = dict(selected_res)
        result["is_all_platforms"] = False
        result["selected_platform"] = target_platform
        result["best_platform"] = best_plat
        result["platforms"] = platforms_results
        result["platform_leaderboard"] = leaderboard

    # Save to SQLite database if requested
    saved_id = None
    if save_to_db:
        try:
            saved_id = save_prediction(base_norm, result)
            result["saved_id"] = saved_id
        except Exception as e:
            logger.error("Failed to save prediction to DB: %s", e)
            result["saved_id"] = None

    return result
