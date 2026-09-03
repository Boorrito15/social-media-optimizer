"""Pydantic request / response models for the prediction API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# Canonical option lists surfaced by the API & UI
PLATFORMS = ["FB", "IG", "TT", "YT"]
PAGES = [
    "All Blacks",
    "Black Ferns",
    "NZ Sevens",
    "NZR",
    "ABXV",
    "Bunnings NPC",
    "SRP",
]

JSON_HELPERS = {
    "content_theme": [
        "rugby_skills", "try", "training", "celebration", "challenges",
        "player story", "kick", "tackle", "haka", "rivalry", "non rugby related",
    ],
    "format_access": [
        "highlight", "candid clip", "behind-the-scenes", "interview",
        "announcement", "promotion", "montage", "reaction", "ticket sales", "archive",
    ],
    "tone": [
        "exciting", "emotional", "celebratory", "informative", "playful",
        "serious", "competitive", "inspiring",
    ],
}


class PredictRequest(BaseModel):
    title: str = Field(default="", description="Post/caption title text.")
    description: str = Field(
        default="",
        description="Play-by-play / story description of the video.",
    )
    platform: str = Field(default="FB")
    page: str = Field(default="All Blacks")
    year: int = Field(default=2025)
    category_l0: Optional[str] = "No Hashtag"
    category_l1: Optional[str] = "No Hashtag"
    category_l2: Optional[str] = "No Hashtag"
    duration_seconds: float = Field(default=20, gt=0, le=3600)
    content_themes: List[str] = Field(default_factory=list)
    format_access: List[str] = Field(default_factory=list)
    tones: List[str] = Field(default_factory=list)
    # Cost / effort for demo ROI computation
    cost: Optional[float] = Field(default=0, ge=0, description="Production cost (optional, demo).")
    expected_rpm: Optional[float] = Field(
        default=3.0, ge=0, description="Advertiser revenue per 1,000 views (demo)."
    )
    expected_cpm: Optional[float] = Field(
        default=5.0, ge=0, description="Cost per 1,000 impressions for paid boost (demo)."
    )
    # Auto-infer: when true, any field left at its default is filled from the
    # description text (the UI sends only the description and relies on this).
    auto: bool = Field(
        default=True,
        description="Fill missing metadata from the description (platform, page, themes, etc.).",
    )


class TargetPrediction(BaseModel):
    target: str
    label: str
    probability: float
    is_high: bool


class SimilarVideo(BaseModel):
    title: str
    description: str
    platform: str
    page: str
    views: float
    engagement: float
    url: str = ""
    distance: float


class PlatformResult(BaseModel):
    platform: str
    go_score: float
    verdict: str
    views_p: float
    eng_p: float
    estimates: Dict[str, Any]


class PredictResponse(BaseModel):
    go_score: float
    verdict: str
    verdict_message: str
    confidence: str
    confidence_note: str
    inferred: Dict[str, Any] = Field(default_factory=dict, description="Auto-generated metadata.")
    views: TargetPrediction
    engagement: TargetPrediction
    estimates: Dict[str, Any]
    money: Dict[str, Any]
    similar: List[SimilarVideo]
    model_metrics: Dict[str, float]
    model_type: str = Field(default="", description="Which model served this prediction.")
    platform_leaderboard: List[PlatformResult] = Field(default_factory=list, description="Per-platform scores.")
    best_platform: str = Field(default="", description="Platform with highest go_score.")
    explanation: Dict[str, Any] = Field(default_factory=dict, description="Feature-level explanation.")


class InferResponse(BaseModel):
    description: str
    metadata: Dict[str, Any]
