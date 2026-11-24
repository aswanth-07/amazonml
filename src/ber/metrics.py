"""The leaderboard metric and blocking diagnostics.

Macro F-beta (beta = 0.5) over *all* Source 1 entities:

* truth empty,  prediction empty      -> 1.0 (a correctly identified singleton)
* truth empty,  prediction non-empty  -> 0.0
* truth non-empty, prediction empty   -> 0.0 (precision undefined, recall 0)
* otherwise F = (1 + b^2) * TP / (b^2 * |truth| + |prediction|)

The last line is algebraically the PDF's (1+b^2)PR / (b^2 P + R).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


def fbeta_entity(pred: Iterable[str], true: Iterable[str], beta: float = 0.5) -> float:
    pred_set, true_set = set(pred), set(true)
    if not pred_set and not true_set:
        return 1.0
    if not pred_set or not true_set:
        return 0.0
    tp = len(pred_set & true_set)
    b2 = beta * beta
    return (1 + b2) * tp / (b2 * len(true_set) + len(pred_set))


def fbeta_from_counts(
    tp: np.ndarray, n_pred: np.ndarray, n_true: np.ndarray, beta: float = 0.5
) -> np.ndarray:
    """Vectorised per-entity F-beta with the singleton rules above."""
    tp = np.asarray(tp, dtype=np.float64)
    n_pred = np.asarray(n_pred, dtype=np.float64)
    n_true = np.asarray(n_true, dtype=np.float64)
    b2 = beta * beta
    denom = b2 * n_true + n_pred
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(denom > 0, (1 + b2) * tp / denom, 0.0)
    return np.where((n_pred == 0) & (n_true == 0), 1.0, f)


def per_entity_scores(
    pred: Mapping[str, Sequence[str]], truth: Mapping[str, Sequence[str]], beta: float = 0.5
) -> pd.Series:
    """F-beta per Source 1 entity in ``truth``; entities absent from ``pred`` predicted empty."""
    return pd.Series(
        {sid: fbeta_entity(pred.get(sid, ()), true, beta) for sid, true in truth.items()},
        dtype=float,
    )


def macro_fbeta(
    pred: Mapping[str, Sequence[str]], truth: Mapping[str, Sequence[str]], beta: float = 0.5
) -> float:
    if not truth:
        raise ValueError("truth is empty")
    return float(per_entity_scores(pred, truth, beta).mean())


def breakdown(
    pred: Mapping[str, Sequence[str]],
    truth: Mapping[str, Sequence[str]],
    groups: Mapping[str, str],
    beta: float = 0.5,
) -> dict[str, dict[str, float]]:
    """Macro F-beta and count per group label (e.g. country, true cardinality)."""
    scores = per_entity_scores(pred, truth, beta)
    labels = pd.Series({sid: groups.get(sid, "?") for sid in scores.index})
    out = {}
    for label, idx in labels.groupby(labels).groups.items():
        out[str(label)] = {"fbeta": float(scores.loc[idx].mean()), "n": len(idx)}
    return out


def cardinality_labels(truth: Mapping[str, Sequence[str]]) -> dict[str, str]:
    return {sid: (str(len(m)) if len(m) < 3 else "3+") for sid, m in truth.items()}


def precision_recall(
    pred: Mapping[str, Sequence[str]], truth: Mapping[str, Sequence[str]]
) -> dict[str, float]:
    """Micro pair-level precision/recall, plus singleton accuracy. Diagnostics only."""
    tp = n_pred = n_true = 0
    singles = singles_ok = 0
    for sid, true in truth.items():
        p, t = set(pred.get(sid, ())), set(true)
        tp += len(p & t)
        n_pred += len(p)
        n_true += len(t)
        if not t:
            singles += 1
            singles_ok += int(not p)
    return {
        "pair_precision": tp / n_pred if n_pred else 1.0,
        "pair_recall": tp / n_true if n_true else 1.0,
        "singleton_accuracy": singles_ok / singles if singles else float("nan"),
        "n_pred_pairs": float(n_pred),
        "n_true_pairs": float(n_true),
    }
