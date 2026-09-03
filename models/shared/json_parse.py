"""JSON description column parser.

Lifted verbatim from the original notebook cell indexed by
``1d62555f-f4c6-49c6-b2e6-9e4b70632911``. We return the parsed dicts keyed by
the original DataFrame index so the encoder module can hot-loop without
losing row alignment.
"""

from __future__ import annotations

import json

import pandas as pd

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


def _parse(text):
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


def parse_json_column(s: pd.Series) -> tuple[pd.Series, dict[int, dict]]:
    """Parse a series of JSON strings.

    Returns
    -------
    description : pd.Series
        The ``play_by_play`` text, or ``pd.NA`` when missing/empty.
    json_by_idx : dict[int, dict]
        Non-null parsed dicts keyed by the original dataframe index, used by
        :func:`models.shared.json_encode.encode_json_fields` for one-hot
        encoding.
    """
    dicts = s.map(_parse)
    description = dicts.map(
        lambda d: ((d or {}).get("play_by_play") or "").strip() or pd.NA
    ).astype("string")
    by_idx = {idx: d for idx, d in zip(s.index, dicts) if d is not None}
    return description, by_idx
