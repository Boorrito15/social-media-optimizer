"""Lazy-loaded sentence embedder + KMeans cluster lookup (predict-time)."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from sklearn.preprocessing import normalize


@lru_cache(maxsize=1)
def _embedder():
    """Load the embedder once per process; subsequent calls return the cached model."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("all-MiniLM-L6-v2")


def cluster_texts(texts: list[str], kmeans) -> np.ndarray:
    """Return cluster assignments for each text using the loaded embedder + saved KMeans.

    Empty / NA strings are coerced to ``""`` so the embedder can still return
    a deterministic vector. Returns an empty array if ``texts`` is empty.
    """
    if not texts:
        return np.array([], dtype=int)
    safe = [t if isinstance(t, str) else "" for t in texts]
    X = _embedder().encode(safe, show_progress_bar=False)
    return kmeans.predict(normalize(X))
