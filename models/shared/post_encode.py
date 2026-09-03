"""Hashtag / mention / emoji multi-hot encoding.

The original notebook builds vocabs in-place from ``df[content]``. Here we
consume the *saved* vocabs (``state['hashtag_vocab']`` etc.) so this code is
deterministic and reproducible at predict-time.

The regex patterns are exposed at module level so the train-time
state-builder (``build_state``) can use the exact same patterns to count
frequencies.
"""

from __future__ import annotations

import regex
import pandas as pd

POST_TITLE_COLUMN = "content"

HASHTAG_PATTERN = regex.compile(r"(?<![\p{L}\p{N}_])#([\p{L}\p{N}_]+)")
MENTION_PATTERN = regex.compile(r"(?<![\p{L}\p{N}_@])@([\p{L}\p{N}_][\p{L}\p{N}_.]*)")
EMOJI_PATTERN = regex.compile(
    r"(?:\p{Regional_Indicator}{2}|[#*0-9]\uFE0F?\u20E3|"
    r"\p{Extended_Pictographic}(?:\uFE0F|\p{Emoji_Modifier})?"
    r"(?:\u200D\p{Extended_Pictographic}(?:\uFE0F|\p{Emoji_Modifier})?)*)"
)


def _hashtag_tokens(text: str) -> list[str]:
    return [tag.casefold() for tag in HASHTAG_PATTERN.findall(text or "")]


def _mention_tokens(text: str) -> list[str]:
    return [m.rstrip(".").casefold() for m in MENTION_PATTERN.findall(text or "")]


def _emoji_tokens(text: str) -> list[str]:
    return EMOJI_PATTERN.findall(text or "")


def _emoji_colname(value: str) -> str:
    codepoints = "_".join(f"{ord(char):x}" for char in value)
    return f"emoji_u{codepoints}"


def _multi_hot(df: pd.DataFrame, tokens: pd.Series, vocab: list[str], prefix: str, colname) -> pd.DataFrame:
    """Multi-hot encode ``tokens`` against ``vocab`` aligned with ``df.index``.

    Skips writing columns for tokens outside the saved vocab so they cannot
    leak new features at predict time.
    """
    new_cols: dict[str, pd.Series] = {}
    new_cols[f"{prefix}_count"] = tokens.map(len).astype("int16")
    vocab_set = set(vocab)
    for tok in vocab:
        new_cols[colname(tok)] = tokens.map(
            lambda ts, value=tok: int(value in ts)
        ).astype("int8")
    return pd.DataFrame(new_cols, index=df.index)


def encode_post_features(df: pd.DataFrame, state: dict) -> pd.DataFrame:
    """Add hashtag_*, mention_*, emoji_* columns + their *_count columns.

    Idempotent: re-running on the same frame drops any pre-existing
    ``<prefix>_*`` columns before re-creating them.
    """
    out = df
    for prefix in ("hashtag", "mention", "emoji"):
        old = [c for c in out.columns if c.startswith(f"{prefix}_")]
        if old:
            out = out.drop(columns=old)

    title = out[POST_TITLE_COLUMN].fillna("").astype(str)
    hashtag_tokens = title.map(_hashtag_tokens)
    mention_tokens = title.map(_mention_tokens)
    emoji_tokens   = title.map(_emoji_tokens)

    out = pd.concat(
        [
            out,
            _multi_hot(out, hashtag_tokens, state["hashtag_vocab"], "hashtag",
                       lambda t: f"hashtag_{t}"),
            _multi_hot(out, mention_tokens, state["mention_vocab"], "mention",
                       lambda m: f"mention_{m}"),
            _multi_hot(out, emoji_tokens,   state["emoji_vocab"],   "emoji",
                       _emoji_colname),
        ],
        axis=1,
    )
    return out
