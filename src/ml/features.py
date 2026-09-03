"""Feature pipeline shared by training and prediction.

Reproduces the categorical / text / semantic encodings used in the
analysis notebooks, but in a single trainable pipeline so that a raw
user input (platform, page, description, title, duration...) is mapped
onto exactly the same feature columns the models were trained on.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterator, List, Optional, Sequence

import numpy as np
import pandas as pd
import regex
from sklearn.preprocessing import OneHotEncoder

# ---------------------------------------------------------------------------
# Vocabulary-ish constants mirroring Notebook feature names
# ---------------------------------------------------------------------------

JSON_LIST_FIELDS = [
    "content_theme",
    "format_access",
    "people",
    "brands",
    "event",
    "tone",
    "context",
    "overall_team",
    "audio_format",
]

CATEGORICAL_FIELDS = ["platform", "page", "year", "category_l0", "category_l1", "category_l2"]

# Regexes taken from the notebook (rob.ipynb)
HASHTAG_PATTERN = regex.compile(r"(?<![\p{L}\p{N}_])#([\p{L}\p{N}_]+)")
MENTION_PATTERN = regex.compile(r"(?<![\p{L}\p{N}_@])@([\p{L}\p{N}_][\p{L}\p{N}_.]*)")
EMOJI_PATTERN = regex.compile(
    r"(?:\p{Regional_Indicator}{2}|[#*0-9]\uFE0F?\u20E3|"
    r"\p{Extended_Pictographic}(?:\uFE0F|\p{Emoji_Modifier})?"
    r"(?:\u200D\p{Extended_Pictographic}(?:\uFE0F|\p{Emoji_Modifier})?)*)"
)


def _parse_json_list(text: Any, fields: Sequence[str] = JSON_LIST_FIELDS) -> List[str]:
    """Return the list-typed JSON fields of a description_json cell."""
    if text is None:
        return []
    if isinstance(text, float) and np.isnan(text):
        return []
    if isinstance(text, str):
        try:
            text = json.loads(text)
        except Exception:
            return []
    if not isinstance(text, dict):
        return []
    out: List[str] = []
    for field in fields:
        values = text.get(field) or []
        if isinstance(values, str):
            values = [values]
        for v in values:
            if isinstance(v, str) and v:
                out.append(f"{field}::{v.strip()}")
    return out


def _tokens(series: pd.Series, pattern: "regex.Pattern") -> pd.Series:
    """Multi-hot style token lists from a text column."""
    return series.fillna("").astype(str).map(
        lambda text: [m.casefold().rstrip(".") for m in pattern.findall(text)]
    )


class FeaturePipeline:
    """Fit on a labelled DataFame, transform raw inputs into a feature row.

    `compact=True` restricts features to exactly what the app can reconstruct
    from a free-text description + inferred metadata: platform & page, the JSON
    content-theme / format-access / tone multi-hots, and duration/counts.
    The token-level hashtag/mention/emoji one-hots and the people/brands/event/
    context/team/audio JSON fields are dropped, since user text rarely populates
    them — this keeps training and serving on the SAME feature space so the
    model actually discriminates on user inputs (an all-zero sparse vector
    otherwise collapses every prediction to the dominant class).
    """

    def __init__(self, compact: bool = False) -> None:
        self.compact = compact
        self.cat_encoder: Optional[OneHotEncoder] = None
        self.cat_columns: List[str] = []
        self.json_vocab: List[str] = []
        self.json_columns: List[str] = []
        self.hashtag_vocab: List[str] = []
        self.mention_vocab: List[str] = []
        self.emoji_vocab: List[str] = []
        self.base_columns: List[str] = [
            "duration_seconds",
            "n_hashtags",
            "n_mentions",
            "n_emojis",
        ]
        self.fitted = False

    # feature categories used in compact mode ----------------------------
    @property
    def _cat_fields(self) -> List[str]:
        return ["platform", "page"] if self.compact else CATEGORICAL_FIELDS

    @property
    def _json_fields(self) -> List[str]:
        return ["content_theme", "format_access", "tone"] if self.compact else JSON_LIST_FIELDS

    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> "FeaturePipeline":
        """Learn vocabularies from a training DataFrame (with raw columns)."""
        # Categorical one-hots
        cat_data = df[self._cat_fields].astype(str)
        self.cat_encoder = OneHotEncoder(
            sparse_output=False, handle_unknown="ignore", min_frequency=1
        )
        self.cat_encoder.fit(cat_data)
        self.cat_columns = [
            c.replace("cat__", "")
            for c in self.cat_encoder.get_feature_names_out()
        ]

        # JSON list multi-hot vocabulary (min 5 occurrences), limited to the
        # fields the app can populate.
        json_tokens = _parse_json_lists_series(df["description_json"], self._json_fields)
        counter = {}
        for toks in json_tokens:
            for t in toks:
                counter[t] = counter.get(t, 0) + 1
        self.json_vocab = sorted(t for t, n in counter.items() if n >= 5)
        self.json_columns = [f"json_{t.replace('::', '__')}" for t in self.json_vocab]

        # Text token vocabularies (only used in non-compact mode)
        self.hashtag_vocab = _frequent_tokens(
            _tokens(df["content"], HASHTAG_PATTERN), min_freq=5
        )
        self.mention_vocab = _frequent_tokens(
            _tokens(df["content"], MENTION_PATTERN), min_freq=5
        )
        self.emoji_vocab = _frequent_tokens(
            _tokens(df["content"], EMOJI_PATTERN), min_freq=5
        )

        self.fitted = True
        return self

    # ------------------------------------------------------------------
    def transform(self, df: pd.DataFrame):
        """Map a raw DataFrame (one or many rows) to model feature matrix.

        Returns a scipy sparse CSR matrix so downstream models (XGBoost etc.)
        train fast on the largely-sparse one-hot feature space.
        """
        import scipy.sparse as sp

        if not self.fitted:
            raise RuntimeError("FeaturePipeline must be fit() before transform().")

        parts: List["sp.csr_matrix"] = []

        # Categorical
        cat_enc = self.cat_encoder.transform(df[self._cat_fields].astype(str))
        parts.append(sp.csr_matrix(cat_enc))

        # JSON list multi-hot
        json_toks = _parse_json_lists_series(df["description_json"], self._json_fields)
        rows, cols, vals = [], [], []
        for i, toks in enumerate(json_toks):
            ts = set(toks)
            for j, vocab in enumerate(self.json_vocab):
                if vocab in ts:
                    rows.append(i)
                    cols.append(j)
                    vals.append(1.0)
        parts.append(
            sp.csr_matrix((vals, (rows, cols)), shape=(len(df), len(self.json_vocab)))
        )

        # Per-token hashtag/mention/emoji one-hots (skipped in compact mode —
        # counts below already capture that signal for the app).
        if not self.compact:
            for vocab, pattern in (
                (self.hashtag_vocab, HASHTAG_PATTERN),
                (self.mention_vocab, MENTION_PATTERN),
                (self.emoji_vocab, EMOJI_PATTERN),
            ):
                rows, cols, vals = [], [], []
                for i, toks in enumerate(_tokens(df["content"], pattern)):
                    ts = set(toks)
                    for j, v in enumerate(vocab):
                        if v in ts:
                            rows.append(i)
                            cols.append(j)
                            vals.append(1.0)
                parts.append(
                    sp.csr_matrix((vals, (rows, cols)), shape=(len(df), len(vocab)))
                )

        # Counts + duration
        if "n_hashtags" not in df.columns:
            df = df.copy()
            df["n_hashtags"] = _tokens(df["content"], HASHTAG_PATTERN).map(len)
            df["n_mentions"] = _tokens(df["content"], MENTION_PATTERN).map(len)
            df["n_emojis"] = _tokens(df["content"], EMOJI_PATTERN).map(len)
            df["duration_seconds"] = pd.to_numeric(df["duration_seconds"], errors="coerce")
        base_dense = np.zeros((len(df), len(self.base_columns)), dtype=np.float32)
        for j, col in enumerate(self.base_columns):
            base_dense[:, j] = df[col].fillna(0).to_numpy(dtype=np.float32)
        parts.append(sp.csr_matrix(base_dense))

        return sp.hstack(parts, format="csr").astype(np.float32)

    # ------------------------------------------------------------------
    def transform_row(self, row: Dict[str, Any]) -> np.ndarray:
        """Transform a single raw input dict into a feature row."""
        d = {
            "platform": str(row.get("platform", "FB")),
            "page": str(row.get("page", "All Blacks")),
            "year": str(row.get("year", "2025")),
            "category_l0": str(row.get("category_l0", "No Hashtag")),
            "category_l1": str(row.get("category_l1", "No Hashtag")),
            "category_l2": str(row.get("category_l2", "No Hashtag")),
            "content": str(row.get("title", "")),
            "description_json": _build_description_json(row, self._json_fields),
            "duration_seconds": float(row.get("duration_seconds", 20) or 20),
        }
        # counts derived from content
        content = d["content"]
        d["n_hashtags"] = len(HASHTAG_PATTERN.findall(content))
        d["n_mentions"] = len(MENTION_PATTERN.findall(content))
        d["n_emojis"] = len(EMOJI_PATTERN.findall(content))
        df = pd.DataFrame([d])
        return self.transform(df).toarray()[0]

    # ------------------------------------------------------------------
    def n_features(self) -> int:
        if not self.fitted:
            raise RuntimeError("not fitted")
        return len(self.feature_names())

    def feature_names(self) -> List[str]:
        """Return ordered list of all feature column names."""
        if not self.fitted:
            raise RuntimeError("not fitted")
        names = list(self.cat_columns) + list(self.json_columns) + list(self.base_columns)
        if not self.compact:
            # Add per-token columns (hashtag, mention, emoji) — these are
            # created in _multi_hot style in transform().
            names += [f"hashtag_{t}" for t in self.hashtag_vocab]
            names += [f"mention_{t}" for t in self.mention_vocab]
            names += [f"emoji_{t}" for t in self.emoji_vocab]
        return names


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _frequent_tokens(token_series: pd.Series, min_freq: int) -> List[str]:
    from collections import Counter

    counter: Counter = Counter()
    for toks in token_series:
        for t in toks:
            counter[t] += 1
    return sorted(t for t, n in counter.items() if n >= min_freq)


def _parse_json_lists_series(series: pd.Series, fields: Sequence[str] = JSON_LIST_FIELDS) -> Iterator[List[str]]:
    for text in series:
        yield _parse_json_list(text, fields)


def _build_description_json(row: Dict[str, Any], fields: Sequence[str] = JSON_LIST_FIELDS) -> str:
    """Build a description_json string from user-supplied multi-select fields."""
    payload: Dict[str, List[str]] = {}
    for field in fields:
        val = row.get(field)
        if not val:
            continue
        if isinstance(val, str):
            val = [val]
        items = [str(x).strip() for x in val if str(x).strip()]
        if items:
            payload[field] = items
    return json.dumps(payload)
