"""Decision rules over grouped pair probabilities."""

import numpy as np
import pytest

from ber.metrics import fbeta_from_counts
from ber.tuning import Grouped, Rule, Scored, select, tune

ROWS = np.array([0, 0, 0, 1, 2, 3, 3])
P = np.array([0.8, 0.9, 0.1, 0.3, 0.6, 0.45, 0.4])


def test_threshold():
    g = Grouped.build(ROWS, P)
    assert select(g, Rule("threshold", {"t": 0.5})).tolist() == [1, 1, 0, 0, 1, 0, 0]


def test_two_threshold_needs_the_best_candidate():
    g = Grouped.build(ROWS, P)
    got = select(g, Rule("two_threshold", {"t_top": 0.5, "t_rest": 0.35}))
    # entity 3's best (0.45) misses t_top, so its 0.4 is not selected either
    assert got.tolist() == [1, 1, 0, 0, 1, 0, 0]
    got = select(g, Rule("two_threshold", {"t_top": 0.42, "t_rest": 0.35}))
    assert got.tolist() == [1, 1, 0, 0, 1, 1, 1]


def test_expected_f_prefers_empty_when_no_match_is_likelier():
    g = Grouped.build(ROWS, P)
    got = select(g, Rule("expected_f", {"min_prob": 0.0, "miss": 0.0}))
    # entity 0: {0.9, 0.8}; entity 1: 0.3 -> P(none)=0.7 beats E[F]=0.35; entity 2: 0.6 kept
    assert got[:5].tolist() == [1, 1, 0, 0, 1]


def test_scored_matches_direct_macro_f():
    labels = np.array([1, 1, 0, 0, 1, 0, 1], dtype=bool)
    n_true = np.array([3.0, 0.0, 1.0, 1.0])
    subset = np.arange(4)
    s = Scored.build(ROWS, P, labels, n_true, subset, min_p=0.0)
    got = s.score(Rule("threshold", {"t": 0.5}))
    tp = np.array([2.0, 0, 1, 0])
    pred = np.array([2.0, 0, 1, 0])
    assert got == pytest.approx(fbeta_from_counts(tp, pred, n_true, 0.5).mean())


def test_tune_picks_the_best_strategy():
    rng = np.random.default_rng(0)
    rows = np.repeat(np.arange(400), 5)
    labels = rng.random(len(rows)) < 0.3
    p = np.clip(np.where(labels, 0.7, 0.2) + rng.normal(0, 0.15, len(rows)), 0.001, 0.999)
    n_true = np.bincount(rows, weights=labels.astype(float), minlength=400)
    rule, report = tune(rows, p, labels, n_true, np.arange(0, 400, 2), np.arange(1, 400, 2))
    best = max(report[s]["tuned"] for s in ("threshold", "two_threshold", "expected_f"))
    assert report[rule.strategy]["tuned"] == best
    assert all("holdout" in report[s] for s in ("threshold", "two_threshold", "expected_f"))
