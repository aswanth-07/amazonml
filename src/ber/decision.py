"""The leaderboard score of a set of chosen pairs.

The leaderboard averages F0.5 per Source 1 entity, singletons included, so a selection is
scored per entity, with every true match the entity has (blocking misses included) in the
denominator. The decision rules that pick the pairs are ``ber.tuning``'s; exclusive
assignment and the final selection are ``ber.scale.exclusive_adjust`` and ``decide``.
"""

from __future__ import annotations

import numpy as np

from ber.metrics import fbeta_from_counts


def macro_from_mask(
    s1_idx: np.ndarray, labels: np.ndarray, mask: np.ndarray, n_true: np.ndarray, beta: float
) -> float:
    """Leaderboard score for a selection, given each S1 entity's full true-match count.

    ``n_true`` counts true matches including those blocking missed, so the
    score is exactly what the metric would give, not a candidate-only proxy.
    """
    n_s1 = len(n_true)
    tp = np.bincount(s1_idx, weights=(mask & labels).astype(float), minlength=n_s1)
    n_pred = np.bincount(s1_idx, weights=mask.astype(float), minlength=n_s1)
    return float(fbeta_from_counts(tp, n_pred, n_true, beta).mean())
