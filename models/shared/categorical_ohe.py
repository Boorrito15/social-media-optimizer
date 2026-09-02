"""Load the saved platform/page OneHotEncoder."""

from __future__ import annotations

import pandas as pd


def encode_categorical(df: pd.DataFrame, ohe) -> pd.DataFrame:
    """Apply the saved OneHotEncoder to ``platform`` + ``page`` columns.

    Unknown values produce an all-zero row, which is the documented
    behaviour of ``OneHotEncoder(handle_unknown="ignore")``.
    """
    arr = ohe.transform(df[["platform", "page"]].astype(str))
    cols = ohe.get_feature_names_out(["platform", "page"])
    return pd.DataFrame(arr, columns=cols, index=df.index).astype("int8")
