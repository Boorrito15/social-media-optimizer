"""Shared, model-agnostic preprocessing primitives.

The submodule is broken into small, single-responsibility files so that both
the train and predict code paths use the *same* transformations. The
high-level orchestrator is :mod:`models.shared.build_features`.
"""

from .build_features import preprocess
from .build_state import build_state
from .categorical_ohe import encode_categorical
from .clustering import cluster_texts
from .drop import DROP_COLS, drop_columns
from .filter import iqr_slice, validity_mask
from .json_encode import encode_json_fields
from .json_parse import JSON_LIST_FIELDS, parse_json_column
from .post_encode import (
    EMOJI_PATTERN,
    HASHTAG_PATTERN,
    MENTION_PATTERN,
    encode_post_features,
)
from .split import split_indices

__all__ = [
    "DROP_COLS",
    "EMOJI_PATTERN",
    "HASHTAG_PATTERN",
    "JSON_LIST_FIELDS",
    "MENTION_PATTERN",
    "build_state",
    "cluster_texts",
    "drop_columns",
    "encode_categorical",
    "encode_json_fields",
    "encode_post_features",
    "iqr_slice",
    "parse_json_column",
    "preprocess",
    "split_indices",
    "validity_mask",
]
