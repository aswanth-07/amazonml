"""Decision rules over tens of millions of scored pairs, and their tuning.

Each rule turns (S1 position, probability) per candidate pair into a selection.
Probabilities are the exclusive-adjusted ones (a pool record keeps only its best
S1 suitor), so every rule respects exclusivity.

- ``threshold``: select p >= t.
- ``two_threshold``: an entity's best candidate is selected at p >= t_top, the
  others at p >= t_rest and only when the best was selected. The metric is per
  entity, so the first match and the additional ones earn different amounts.
- ``expected_f``: per entity, the prefix (by p) that maximises the plug-in
  expected F-beta, or nothing when P(no match) is higher; candidates below
  ``min_prob`` are ignored.

Everything is numpy over one sort per call; parameters are tuned for the exact
macro F-beta on out-of-fold probabilities and reported on held-out entities.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ber.decision import macro_from_mask

log = logging.getLogger(__name__)


def em_prior(p: np.ndarray, prior: float, iters: int = 200, tol: float = 1e-7) -> float:
    """The positive rate of a new population, from the posteriors ``p`` of a model trained
    where it was ``prior`` (Saerens, Latinne and Decaestecker 2002, an EM procedure)."""
    new = prior
    p = p.astype(np.float64)
    for _ in range(iters):
        w1, w0 = new / prior, (1 - new) / (1 - prior)
        nxt = float((w1 * p / (w1 * p + w0 * (1 - p))).mean())
        if abs(nxt - new) < tol:
            return nxt
        new = nxt
    return new


def rescale_prior(p: np.ndarray, prior: float, new: float) -> np.ndarray:
    """Posteriors re-weighted from the training positive rate to a new one (Bayes' rule)."""
    w1, w0 = new / prior, (1 - new) / (1 - prior)
    p = p.astype(np.float64)
    return (w1 * p / (w1 * p + w0 * (1 - p))).astype(np.float32)


@dataclass
class Rule:
    strategy: str = "threshold"
    params: dict[str, float] = field(default_factory=lambda: {"t": 0.5})


@dataclass
class Grouped:
    """Pairs sorted by S1 then probability (descending), with group bookkeeping."""

    order: np.ndarray  # positions into the caller's arrays
    rows: np.ndarray
    p: np.ndarray
    start: np.ndarray  # per pair: index of its group's first pair (in sorted order)
    rank: np.ndarray  # per pair: 0-based rank within its group

    @classmethod
    def build(cls, rows: np.ndarray, p: np.ndarray) -> Grouped:
        order = np.lexsort((-p, rows))
        r, ps = rows[order], p[order]
        new = np.ones(len(r), dtype=bool)
        new[1:] = r[1:] != r[:-1]
        starts = np.flatnonzero(new)
        gid = np.cumsum(new) - 1
        start = starts[gid]
        return cls(order, r, ps, start, np.arange(len(r)) - start)


def _unsort(g: Grouped, sorted_mask: np.ndarray) -> np.ndarray:
    out = np.empty(len(sorted_mask), dtype=bool)
    out[g.order] = sorted_mask
    return out


def select(g: Grouped, rule: Rule, beta: float = 0.5) -> np.ndarray:
    """Boolean selection aligned with the arrays ``g`` was built from."""
    p, prm = g.p, rule.params
    pos = p > 0
    if rule.strategy == "threshold":
        return _unsort(g, pos & (p >= prm["t"]))
    if rule.strategy == "two_threshold":
        top_ok = pos & (g.rank == 0) & (p >= prm["t_top"])
        group_ok = top_ok[g.start]
        rest = pos & (g.rank > 0) & (p >= prm["t_rest"]) & group_ok
        return _unsort(g, top_ok | rest)
    if rule.strategy == "expected_f":
        return _unsort(g, _expected_f_sorted(g, prm["min_prob"], prm.get("miss", 0.0), beta))
    raise ValueError(f"unknown decision strategy {rule.strategy!r}")


def _expected_f_sorted(g: Grouped, min_prob: float, miss: float, beta: float) -> np.ndarray:
    """Plug-in expected F-beta per prefix: (1+b2) * sum_{i<=j} p_i / (j + b2 * E|true|),
    with E|true| = sum of all p (>= min_prob) + ``miss`` (true matches blocking missed).
    Choose the best prefix unless P(no true match) = prod(1 - p) beats it."""
    b2 = beta * beta
    p = np.where(g.p >= min_prob, g.p, 0.0).astype(np.float64)
    csum = np.cumsum(p)
    before = np.concatenate([[0.0], csum])[g.start]  # sum over earlier groups
    cum = csum - before
    n_groups_end = np.append(np.flatnonzero(g.rank[1:] == 0), len(p) - 1)  # last index per group
    gid = np.cumsum(g.rank == 0) - 1
    total = cum[n_groups_end][gid]
    j = g.rank + 1.0
    ef = np.where(p > 0, (1 + b2) * cum / (j + b2 * (total + miss)), -1.0)
    log_none = np.log1p(-np.clip(p, 0.0, 1.0 - 1e-9))
    lcs = np.cumsum(log_none)
    lbefore = np.concatenate([[0.0], lcs])[g.start]
    p_none = np.exp((lcs - lbefore)[n_groups_end][gid]) * np.exp(-miss)
    # best prefix length per group: first index of the group max
    best = np.full(gid.max() + 1 if len(gid) else 0, -np.inf)
    np.maximum.at(best, gid, ef)
    is_best = ef == best[gid]
    first_best = np.full(len(best), np.iinfo(np.int64).max)
    np.minimum.at(first_best, gid[is_best], g.rank[is_best])
    return (g.rank <= first_best[gid]) & (best[gid] > p_none) & (p > 0)


@dataclass
class Scored:
    """The pairs of one set of S1 entities, ready for repeated rule evaluation."""

    g: Grouped
    labels: np.ndarray  # sorted like g
    local: np.ndarray  # sorted like g: position of the pair's S1 within ``subset``
    n_true: np.ndarray  # per S1 in ``subset``

    @classmethod
    def build(cls, rows, p, labels, n_true, subset, min_p: float = 0.005) -> Scored:
        """``subset``: sorted S1 positions. Pairs below ``min_p`` are dropped: no tuned rule
        selects them, and the entities keep their full true-match counts."""
        in_sub = np.zeros(len(n_true), dtype=bool)
        in_sub[subset] = True
        keep = in_sub[rows] & (p >= min_p)
        g = Grouped.build(rows[keep], p[keep])
        return cls(g, labels[keep][g.order], np.searchsorted(subset, g.rows), n_true[subset])

    def score(self, rule: Rule, beta: float = 0.5) -> float:
        mask = select(self.g, rule, beta)[self.g.order]
        return macro_from_mask(self.local, self.labels, mask, self.n_true, beta)


def _grid(strategy: str) -> list[dict[str, float]]:
    ts = np.round(np.arange(0.02, 0.981, 0.02), 3)
    if strategy == "threshold":
        return [{"t": float(t)} for t in np.round(np.arange(0.02, 0.981, 0.01), 3)]
    if strategy == "two_threshold":
        return [{"t_top": float(a), "t_rest": float(b)} for a, b in itertools.product(ts, ts)]
    if strategy == "expected_f":
        return [{"min_prob": float(m), "miss": float(x)}
                for m in (0.0, 0.01, 0.02, 0.05, 0.1) for x in (0.0, 0.05, 0.1, 0.2)]  # fmt: skip
    raise ValueError(strategy)


def tune(
    rows: np.ndarray,
    p: np.ndarray,
    labels: np.ndarray,
    n_true: np.ndarray,
    tune_on: np.ndarray,
    report_on: np.ndarray | None,
    beta: float = 0.5,
    strategies: tuple[str, ...] = ("threshold", "two_threshold", "expected_f"),
) -> tuple[Rule, dict[str, Any]]:
    """Best rule by macro F-beta on the sorted S1 positions ``tune_on``; every strategy's
    best is also reported on ``report_on``."""
    tune_set = Scored.build(rows, p, labels, n_true, tune_on)
    report_set = (
        Scored.build(rows, p, labels, n_true, report_on)
        if report_on is not None and len(report_on)
        else None
    )
    report: dict[str, Any] = {}
    best_rule, best_score = Rule(), -1.0
    for strategy in strategies:
        s_best, s_rule = -1.0, None
        for params in _grid(strategy):
            rule = Rule(strategy, params)
            score = tune_set.score(rule, beta)
            if score > s_best:
                s_best, s_rule = score, rule
        entry = {"params": s_rule.params, "tuned": s_best}
        if report_set is not None:
            entry["holdout"] = report_set.score(s_rule, beta)
        report[strategy] = entry
        log.info("decision %s: %s", strategy, entry)
        if s_best > best_score:
            best_rule, best_score = s_rule, s_best
    report["chosen"] = {"strategy": best_rule.strategy, "params": best_rule.params}
    return best_rule, report
