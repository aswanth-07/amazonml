"""From pair probabilities to one match list per Source 1 entity.

The leaderboard averages F0.5 per entity, singletons included, so the decision
is per entity rather than per pair:

* ``exclusive``: Source 1 is deduplicated, so a Source 2/3 record should belong
  to at most one Source 1 entity. Only its best-scoring suitor keeps it.
* ``threshold``: keep candidates with probability >= tau.
* ``expected_f``: per entity, keep the top-j candidates that maximise a plug-in
  estimate of expected F-beta, and keep none when P(no match) is higher. Only
  candidates with probability >= ``min_prob`` are considered.

Either strategy has one scalar that is tuned on out-of-fold predictions for the
exact leaderboard metric.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from ber.blocking import POOL_IDX, S1_IDX
from ber.config import DecisionConfig
from ber.metrics import fbeta_from_counts

PROB = "prob"


def exclusive_probs(pairs: pd.DataFrame) -> np.ndarray:
    """Zero every pair whose pool record has a better-scoring S1 suitor (ties: lowest S1)."""
    order = pairs.sort_values(
        [POOL_IDX, PROB, S1_IDX], ascending=[True, False, True], kind="stable"
    )
    best = ~order[POOL_IDX].duplicated()
    keep = best.reindex(pairs.index).to_numpy()
    return np.where(keep, pairs[PROB].to_numpy(), 0.0)


def _cap_per_s1(
    pairs: pd.DataFrame, mask: np.ndarray, probs: np.ndarray, max_per_s1: int | None
) -> np.ndarray:
    if max_per_s1 is None or not mask.any():
        return mask
    frame = pd.DataFrame({S1_IDX: pairs[S1_IDX].to_numpy(), "p": np.where(mask, probs, -1.0)})
    rank = frame.groupby(S1_IDX)["p"].rank(ascending=False, method="first").to_numpy()
    return mask & (rank <= max_per_s1)


def select_threshold(
    pairs: pd.DataFrame, probs: np.ndarray, tau: float, max_per_s1: int | None = None
) -> np.ndarray:
    mask = (probs >= tau) & (probs > 0)
    return _cap_per_s1(pairs, mask, probs, max_per_s1)


def select_expected_f(
    pairs: pd.DataFrame,
    probs: np.ndarray,
    beta: float,
    min_prob: float,
    max_per_s1: int | None = None,
) -> np.ndarray:
    considered = (probs >= min_prob) & (probs > 0)
    mask = np.zeros(len(pairs), dtype=bool)
    if not considered.any():
        return mask
    idx = np.flatnonzero(considered)
    frame = pd.DataFrame({"row": idx, "g": pairs[S1_IDX].to_numpy()[idx], "p": probs[idx]})
    frame = frame.sort_values(["g", "p"], ascending=[True, False], kind="stable")
    grouped = frame.groupby("g", sort=False)["p"]
    b2 = beta * beta
    j = grouped.cumcount().to_numpy() + 1
    cum = grouped.cumsum().to_numpy()
    total = grouped.transform("sum").to_numpy()
    ef = (1 + b2) * cum / (b2 * total + j)
    if max_per_s1 is not None:
        ef = np.where(j <= max_per_s1, ef, -np.inf)
    frame["ef"] = ef
    frame["j"] = j
    frame["log_none"] = np.log1p(-np.clip(frame["p"].to_numpy(), 0.0, 1.0 - 1e-9))
    g = frame.groupby("g", sort=False)
    best_ef = g["ef"].transform("max").to_numpy()
    p_none = np.exp(g["log_none"].transform("sum").to_numpy())
    best_j = frame.assign(is_best=ef == best_ef).loc[lambda d: d["is_best"]].groupby("g")["j"].min()
    frame["best_j"] = frame["g"].map(best_j).to_numpy()
    chosen = (frame["j"].to_numpy() <= frame["best_j"].to_numpy()) & (best_ef > p_none)
    mask[frame["row"].to_numpy()[chosen]] = True
    return mask


def decide(pairs: pd.DataFrame, cfg: DecisionConfig, beta: float, param: float) -> np.ndarray:
    """Boolean mask over ``pairs`` (which need s1_idx, pool_idx, prob)."""
    probs = exclusive_probs(pairs) if cfg.exclusive else pairs[PROB].to_numpy()
    if cfg.strategy == "threshold":
        return select_threshold(pairs, probs, param, cfg.max_matches_per_s1)
    return select_expected_f(pairs, probs, beta, param, cfg.max_matches_per_s1)


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


def tune(
    pairs: pd.DataFrame, labels: np.ndarray, n_true: np.ndarray, cfg: DecisionConfig, beta: float
) -> tuple[float, pd.DataFrame]:
    """Grid-search the strategy's scalar for macro F-beta. Returns (best value, curve)."""
    grid = np.round(np.arange(cfg.grid_min, cfg.grid_max + 1e-9, cfg.grid_step), 6)
    s1_idx = pairs[S1_IDX].to_numpy()
    labels = labels.astype(bool)
    rows = []
    for value in grid:
        mask = decide(pairs, cfg, beta, float(value))
        rows.append(
            {
                "param": float(value),
                "fbeta": macro_from_mask(s1_idx, labels, mask, n_true, beta),
                "n_pred": int(mask.sum()),
            }
        )
    curve = pd.DataFrame(rows)
    best = curve.loc[curve["fbeta"].idxmax()]
    return float(best["param"]), curve


def to_id_lists(
    pairs: pd.DataFrame, mask: np.ndarray, s1_ids: Sequence[str], pool_ids: np.ndarray
) -> dict[str, list[str]]:
    """{S1 id: [matched pool ids by descending probability]} for every S1 id."""
    out: dict[str, list[str]] = {sid: [] for sid in s1_ids}
    chosen = pairs.loc[mask, [S1_IDX, POOL_IDX, PROB]].sort_values(
        [S1_IDX, PROB], ascending=[True, False]
    )
    for s1_pos, group in chosen.groupby(S1_IDX, sort=False):
        out[s1_ids[s1_pos]] = pool_ids[group[POOL_IDX].to_numpy()].tolist()
    return out
