"""SQLite database persistence layer for Social Media Optimizer prediction runs."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB_DIR = PROJECT_ROOT / "data"
DB_PATH = DB_DIR / "smo.db"


def get_connection() -> sqlite3.Connection:
    """Return a connection to the SQLite database, ensuring tables exist."""
    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Initialize the SQLite schema."""
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                description TEXT NOT NULL,
                title TEXT,
                platform TEXT,
                page TEXT,
                duration_seconds REAL,
                go_score REAL,
                verdict TEXT,
                verdict_message TEXT,
                views_pred REAL,
                engagement_pred REAL,
                views_prob REAL,
                engagement_prob REAL,
                metadata_json TEXT,
                full_payload_json TEXT
            )
            """
        )
        conn.commit()


def save_prediction(req_data: Dict[str, Any], res_data: Dict[str, Any]) -> int:
    """Save a single prediction run and its full result payload into SQLite."""
    init_db()

    now_iso = datetime.now(timezone.utc).isoformat()
    description = req_data.get("description", "")
    title = req_data.get("title", "")
    platform = req_data.get("platform", "FB")
    page = req_data.get("page", "All Blacks")
    duration = float(req_data.get("duration_seconds", 20.0) or 20.0)

    go_score = float(res_data.get("go_score", 0.0) or 0.0)
    verdict = str(res_data.get("verdict", "Neutral"))
    verdict_message = str(res_data.get("verdict_message", ""))

    views_pred = float(
        (res_data.get("estimates") or {}).get("views")
        or (res_data.get("views") or {}).get("probability", 0) * 100000
    )
    eng_pred = float(
        (res_data.get("estimates") or {}).get("engagement")
        or (res_data.get("engagement") or {}).get("probability", 0) * 5000
    )
    views_prob = float((res_data.get("views") or {}).get("probability", 0.5))
    eng_prob = float((res_data.get("engagement") or {}).get("probability", 0.5))


    meta_json = json.dumps(req_data, ensure_ascii=False)
    payload_json = json.dumps(res_data, ensure_ascii=False)

    # Save to dedicated predictions JSON directory in the codebase
    predictions_json_dir = DB_DIR / "predictions"
    predictions_json_dir.mkdir(parents=True, exist_ok=True)

    
    timestamp_slug = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    json_filename = f"prediction_{timestamp_slug}_{platform.lower()}.json"
    json_path = predictions_json_dir / json_filename

    export_payload = {
        "timestamp": now_iso,
        "input": req_data,
        "prediction": res_data,
    }
    
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(export_payload, f, indent=2, ensure_ascii=False)
    except Exception as e:
        # Non-blocking if file write encounters issues
        pass

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO predictions (
                created_at, description, title, platform, page,
                duration_seconds, go_score, verdict, verdict_message,
                views_pred, engagement_pred, views_prob, engagement_prob,
                metadata_json, full_payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                now_iso,
                description,
                title,
                platform,
                page,
                duration,
                go_score,
                verdict,
                verdict_message,
                views_pred,
                eng_pred,
                views_prob,
                eng_prob,
                meta_json,
                payload_json,
            ),
        )
        conn.commit()
        return cursor.lastrowid



def get_history(limit: int = 20) -> List[Dict[str, Any]]:
    """Retrieve recent predictions ordered from newest to oldest."""
    init_db()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, created_at, description, title, platform, page,
                   duration_seconds, go_score, verdict, views_pred, engagement_pred
            FROM predictions
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cursor.fetchall()
        return [dict(r) for r in rows]


def get_prediction(prediction_id: int) -> Optional[Dict[str, Any]]:
    """Retrieve a single prediction by its ID including full payload."""
    init_db()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT * FROM predictions WHERE id = ?
            """,
            (prediction_id,),
        )
        row = cursor.fetchone()
        if not row:
            return None
        res = dict(row)
        if res.get("metadata_json"):
            try:
                res["metadata"] = json.loads(res["metadata_json"])
            except Exception:
                res["metadata"] = {}
        if res.get("full_payload_json"):
            try:
                res["full_payload"] = json.loads(res["full_payload_json"])
            except Exception:
                res["full_payload"] = {}
        return res


def delete_prediction(prediction_id: int) -> bool:
    """Delete a prediction by its ID."""
    init_db()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM predictions WHERE id = ?", (prediction_id,))
        conn.commit()
        return cursor.rowcount > 0
