"""Train/test split helper."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split


def split_indices(n_rows: int, test_size: float, seed: int):
    """Return two numpy index arrays (``idx_tr``, ``idx_te``) over ``range(n_rows)``."""
    idx_tr, idx_te = train_test_split(
        np.arange(n_rows), test_size=test_size, random_state=seed
    )
    return idx_tr, idx_te
