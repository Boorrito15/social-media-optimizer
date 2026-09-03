"""Multi-hot encode JSON list fields using a SAVED vocabulary.

Vocabulary comes from ``state.json_vocab`` so this function is fully
deterministic, no fit-time statistics are required (you can rebuild the
state once and call this at predict time without touching the original
dataset).
"""

from __future__ import annotations

import pandas as pd

from .json_parse import JSON_LIST_FIELDS


def encode_json_fields(
    df: pd.DataFrame,
    json_by_idx: dict[int, dict],
    vocabs: dict[str, list[str]],
) -> pd.DataFrame:
    """Return a frame of new ``json_<field>_<value>`` columns aligned to ``df.index``.

    Mirrors the original notebook behaviour: multi-hot at the token level
    (one column per ``(field, value)`` pair), so a row can have several
    ``= 1`` values per field.
    """
    new_cols: dict[str, pd.Series] = {}
    for field in JSON_LIST_FIELDS:
        vocab = vocabs.get(field, [])
        if not vocab:
            continue
        for value in vocab:
            col_name = f"json_{field}_{value}"
            data = [
                int(d is not None and value in (d.get(field, []) or []))
                for idx, d in zip(df.index, [json_by_idx.get(i) for i in df.index])
            ]
            new_cols[col_name] = pd.Series(data, index=df.index, dtype="int8")
    return pd.DataFrame(new_cols, index=df.index)
