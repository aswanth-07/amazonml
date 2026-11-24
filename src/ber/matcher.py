"""Grouped cross-fitting folds.

Folds are drawn over Source 1 entities, so every candidate of an entity is
scored by a model that never saw that entity's labels (``ber.scale.fit_folds``).
The resulting out-of-fold probabilities are what the decision rule is tuned on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def group_folds(groups: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    """Fold id per row; all rows of a group share a fold. Deterministic for a seed."""
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique)
    fold_of = pd.Series(np.arange(len(shuffled)) % n_folds, index=shuffled)
    return fold_of.reindex(groups).to_numpy()
