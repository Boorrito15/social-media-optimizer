"""Tests for SQLite database persistence, multi-platform execution, and UI prediction service."""

import pytest
from src.db.database import (
    delete_prediction,
    get_history,
    get_prediction,
    init_db,
    save_prediction,
)
from src.ml.service import infer_metadata, run_prediction_pipeline


def test_sqlite_db_lifecycle():
    init_db()

    req = {
        "title": "Test Winning Try",
        "description": "A player breaks the line and scores under the posts.",
        "platform": "IG",
        "page": "All Blacks",
        "duration_seconds": 18.0,
        "content_theme": ["try", "celebration"],
        "format_access": ["highlight"],
        "tone": ["excitement"],
    }
    res = {
        "go_score": 82.5,
        "verdict": "make",
        "verdict_message": "Strong greenlight post.",
        "confidence": "high",
        "estimates": {"views": 185000, "engagement": 4200},
        "views": {"probability": 0.85},
        "engagement": {"probability": 0.79},
    }

    pred_id = save_prediction(req, res)
    assert pred_id is not None
    assert pred_id > 0

    # Verify JSON file was written into data/predictions
    from src.db.database import DB_DIR
    predictions_json_dir = DB_DIR / "predictions"
    assert predictions_json_dir.exists()
    json_files = list(predictions_json_dir.glob("prediction_*.json"))
    assert len(json_files) > 0

    history = get_history(limit=5)
    assert len(history) > 0

    assert any(h["id"] == pred_id for h in history)

    record = get_prediction(pred_id)
    assert record is not None
    assert record["go_score"] == 82.5
    assert record["platform"] == "IG"
    assert record["page"] == "All Blacks"
    assert record["full_payload"]["verdict"] == "make"

    deleted = delete_prediction(pred_id)
    assert deleted is True

    record_after = get_prediction(pred_id)
    assert record_after is None


def test_infer_metadata_good_vs_terrible():
    # Good rugby prompt with named players and sponsors
    good_desc = "Ardie Savea and Beauden Barrett combine for a try against Springboks in Adidas and Ineos gear."
    meta_good = infer_metadata(good_desc)
    assert "title" in meta_good
    assert meta_good["platform"] == "ALL"
    assert meta_good["page"] == "All Blacks"
    assert "try" in meta_good["content_theme"]
    assert "ardie savea" in meta_good["people"]
    assert "beauden barrett" in meta_good["people"]
    assert "springboks" in meta_good["people"]
    assert "adidas" in meta_good["brands"]
    assert "ineos" in meta_good["brands"]

    # Terrible / non-rugby prompt
    bad_desc = "A terrible blurry camera with static noise and nothing happens."
    meta_bad = infer_metadata(bad_desc)
    assert "non rugby related" in meta_bad["content_theme"]
    assert "solemn" in meta_bad["tone"]



def test_prediction_score_discrimination():
    good_input = {
        "title": "Unbelievable Try 🏉",
        "description": "A male rugby player sprints downfield, breaks a tackle and scores a try under the posts.",
        "platform": "ALL",
        "page": "All Blacks",
        "duration_seconds": 15.0,
        "content_theme": ["try", "celebration"],
        "format_access": ["highlight"],
        "tone": ["excitement"],
    }
    good_res = run_prediction_pipeline(good_input, save_to_db=False)

    bad_input = {
        "title": "Boring test video",
        "description": "A terrible blurry camera with static noise and nothing happens.",
        "platform": "ALL",
        "page": "All Blacks",
        "duration_seconds": 15.0,
        "content_theme": ["non rugby related"],
        "format_access": ["candid clip"],
        "tone": ["solemn"],
    }
    bad_res = run_prediction_pipeline(bad_input, save_to_db=False)

    # Score discrimination check
    assert good_res["go_score"] > bad_res["go_score"]
    assert bad_res["go_score"] < 40.0
    assert bad_res["verdict"] in ("skip", "borderline")


def test_cross_platform_score_differentiation():
    # Short fast highlight should score highest on IG / TT vs YT
    short_highlight = {
        "title": "Insane Step & Try",
        "description": "Quick 12s burst breaking two tackles for a score under the posts.",
        "platform": "ALL",
        "page": "All Blacks",
        "duration_seconds": 12.0,
        "content_theme": ["try", "rugby_skills"],
        "format_access": ["highlight"],
        "tone": ["excitement"],
        "audio_format": ["music"],
    }
    res = run_prediction_pipeline(short_highlight, save_to_db=False)
    plat_scores = {p: res["platforms"][p]["go_score"] for p in ["FB", "IG", "TT", "YT"]}
    
    # Verify scores are differentiated across platforms
    unique_scores = set(plat_scores.values())
    assert len(unique_scores) >= 3, f"Expected distinct scores across platforms, got {plat_scores}"
    
    # Verify platform-specific expected views/engagement are realistically scaled
    ig_est = res["platforms"]["IG"]["estimates"]
    yt_est = res["platforms"]["YT"]["estimates"]
    assert ig_est["engagement"] > yt_est["engagement"]

    # Verify fit_explanation is present in both single platform results and leaderboard
    for p in ["FB", "IG", "TT", "YT"]:
        assert "fit_explanation" in res["platforms"][p]
        assert len(res["platforms"][p]["fit_explanation"]) > 0

    for row in res["platform_leaderboard"]:
        assert "fit_explanation" in row
        assert len(row["fit_explanation"]) > 0

