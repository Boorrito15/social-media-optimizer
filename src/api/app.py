"""FastAPI application serving the SMO prediction API.

Run:
    .venv/bin/uvicorn src.api.app:app --reload --port 8000

Endpoints:
    GET  /health
    POST /infer        — turn a free-text description into full metadata
    POST /predict      — verdict, via LLM frame-by-frame projection when the
                         frame-by-frame model is trained (falls back to compact)
    GET  /explore/peers
    GET  /schema
"""

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    InferResponse,
    PredictRequest,
    PredictResponse,
)
from src.ml import infer as inferlib
from src.ml import predict
from src.ml import predict_keras

# The frame-by-frame model branch (feature/frame-by-frame-model) adds
# `predict_fbf`; this UI/API branch should keep running even when that module
# is absent, so the import is optional.
try:
    from src.ml import predict_fbf as _predict_fbf
except Exception:  # noqa: BLE001
    _predict_fbf = None

# Prefer Keras over XGBoost when the Keras bundle is present (Path A)
_KERAS_AVAILABLE = False

app = FastAPI(
    title="Social Media Optimizer API",
    version="0.3.1",
    description="Predict whether an All Blacks short-form video will perform. "
    "Give it a free-text description; it is projected into a frame-by-frame "
    "scene breakdown (LLM) and scored.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_FBF_AVAILABLE = _predict_fbf is not None and os.path.exists(_predict_fbf.BUNDLE_FBF)


@app.on_event("startup")
def _load() -> None:
    global _KERAS_AVAILABLE

    # Prefer Keras model (Path A) — it achieves 82-84% accuracy
    predict_keras.load()
    _KERAS_AVAILABLE = predict_keras.is_available()

    # Fallback: XGBoost compact model
    predict.load()

    if _FBF_AVAILABLE:
        _predict_fbf._load()
    _ = predict._get_embedder() if predict._similar is not None else None


@app.get("/health")
def health() -> dict:
    if _KERAS_AVAILABLE:
        return {"status": "ok", "mode": "keras_2head",
                "models_loaded": True,
                "accuracy": predict_keras._bundle.get("metrics", {}) if predict_keras._bundle else None}
    if _FBF_AVAILABLE:
        return {"status": "ok", "mode": "frame-by-frame",
                "models_loaded": _predict_fbf._bundle is not None}
    return {"status": "ok", "mode": "compact",
            "models_loaded": predict._bundle is not None}


@app.post("/infer", response_model=InferResponse)
def infer_endpoint(req: PredictRequest) -> dict:
    meta = inferlib.infer(req.description or "")
    return {"description": req.description or "", "metadata": meta}


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(req: PredictRequest) -> dict:
    desc = (req.description or "").strip()

    # Keras 2-head model (Path A) — best accuracy
    if _KERAS_AVAILABLE:
        raw = predict_keras.build_raw(req.model_dump())
        meta = inferlib.infer(desc)
        _fill_auto_fields(req, raw, meta)
        result = predict_keras.predict_with_explain(raw)
        result["inferred"] = meta
        result["model_type"] = "keras_2head"
        return result

    raw = predict.build_raw(req.model_dump())

    if _FBF_AVAILABLE and desc:
        try:
            # general idea -> frame-by-frame breakdown via the LLM layer
            meta_signal = inferlib.infer(desc)
            breakdown = _predict_fbf.project_description(
                desc,
                metadata={
                    "content_theme": meta_signal.get("content_themes") or [],
                    "format_access": meta_signal.get("format_access") or [],
                    "tone": meta_signal.get("tones") or [],
                },
            )
            result = _predict_fbf.predict_raw_bf(breakdown, raw)
            result["inferred"] = breakdown
            result["projection"] = {"layer": "gemini-frame-by-frame",
                                    "notes": "general description -> frame-by-frame"}
            return result
        except Exception as e:  # noqa: BLE001
            # fall back to compact model if the LLM layer fails
            result = _compact_predict(req, raw)
            result["projection"] = {"layer": "compact-fallback",
                                    "notes": f"LLM layer failed ({e}); used raw text"}
            return result

    result = _compact_predict(req, raw)
    return result


@app.post("/predict/multi", response_model=PredictResponse)
def predict_multi_endpoint(req: PredictRequest) -> dict:
    """Score the same idea across all platforms.

    Returns the base result plus:
      - platform_leaderboard: sorted list of per-platform scores
      - best_platform: the platform with the highest go_score
      - platforms: detailed per-platform results
    """
    desc = (req.description or "").strip()

    if _KERAS_AVAILABLE:
        raw = predict_keras.build_raw(req.model_dump())
        meta = inferlib.infer(desc)
        _fill_auto_fields(req, raw, meta)
        result = predict_keras.predict_multi(raw)
        result["inferred"] = meta
        result["model_type"] = "keras_2head"
        return result

    # Fallback: use the single-platform predict and just vary the platform field
    result = predict_endpoint(req)
    return result


def _fill_auto_fields(req: PredictRequest, raw: dict, meta: dict) -> None:
    """Fill fields the user left at defaults with inferred values."""
    if not req.auto:
        return
    default_req = PredictRequest()
    for field in ("title", "platform", "page", "duration_seconds",
                  "content_themes", "format_access", "tones"):
        default_val = getattr(default_req, field)
        current = getattr(req, field)
        if current == default_val or (isinstance(current, list) and not current):
            raw[field] = meta.get(field, default_val)


def _compact_predict(req: PredictRequest, raw: dict) -> dict:
    meta = inferlib.infer(req.description or "")
    _fill_auto_fields(req, raw, meta)
    result = predict.predict_raw(raw)
    result["inferred"] = meta
    return result


@app.get("/explore/peers")
def peers() -> dict:
    ps = predict.peers_for_explore()
    return {"count": len(ps), "peers": ps}


@app.get("/schema")
def schema_info() -> dict:
    return {
        "platforms": ["FB", "IG", "TT", "YT"],
        "pages": [
            "All Blacks", "Black Ferns", "NZ Sevens", "NZR",
            "ABXV", "Bunnings NPC", "SRP",
        ],
    }

