import numpy as np
import pandas as pd
import pytest

from ber.config import DecisionConfig
from ber.decision import (
    PROB,
    decide,
    exclusive_probs,
    macro_from_mask,
    select_expected_f,
    select_threshold,
    to_id_lists,
    tune,
)
from ber.metrics import macro_fbeta


def _pairs(rows):
    return pd.DataFrame(rows, columns=["s1_idx", "pool_idx", PROB])


def test_exclusive_keeps_only_the_best_suitor():
    pairs = _pairs([(0, 7, 0.9), (1, 7, 0.6), (1, 8, 0.5)])
    np.testing.assert_allclose(exclusive_probs(pairs), [0.9, 0.0, 0.5])


def test_exclusive_ties_go_to_lowest_s1():
    pairs = _pairs([(3, 7, 0.8), (1, 7, 0.8)])
    np.testing.assert_allclose(exclusive_probs(pairs), [0.0, 0.8])


def test_threshold_with_cap():
    pairs = _pairs([(0, 1, 0.9), (0, 2, 0.8), (0, 3, 0.7), (1, 4, 0.2)])
    probs = pairs[PROB].to_numpy()
    assert select_threshold(pairs, probs, 0.5).tolist() == [True, True, True, False]
    assert select_threshold(pairs, probs, 0.5, max_per_s1=2).tolist() == [True, True, False, False]


def test_expected_f_prefers_nothing_for_weak_evidence():
    # E[F|take it] = 1.25*.2/(.25*.2+1) = .238 < P(no match) = .8
    pairs = _pairs([(0, 1, 0.2)])
    assert not select_expected_f(pairs, pairs[PROB].to_numpy(), beta=0.5, min_prob=0.0).any()


def test_expected_f_takes_strong_candidates():
    pairs = _pairs([(0, 1, 0.9), (0, 2, 0.85), (0, 3, 0.05), (1, 4, 0.95)])
    mask = select_expected_f(pairs, pairs[PROB].to_numpy(), beta=0.5, min_prob=0.0)
    assert mask.tolist() == [True, True, False, True]


def test_macro_from_mask_matches_reference_metric():
    pairs = _pairs([(0, 0, 0.9), (0, 1, 0.4), (1, 2, 0.7), (2, 3, 0.2)])
    labels = np.array([True, True, False, False])
    n_true = np.array(
        [3.0, 0.0, 0.0, 1.0]
    )  # S1 #0 has a match blocking missed; #3 has no candidates
    mask = np.array([True, False, True, False])
    s1_ids, pool_ids = ["A", "B", "C", "D"], np.array(["p0", "p1", "p2", "p3"])
    truth = {"A": ["p0", "p1", "missed"], "B": [], "C": [], "D": ["elsewhere"]}
    pred = to_id_lists(pairs, mask, s1_ids, pool_ids)
    assert macro_from_mask(pairs["s1_idx"].to_numpy(), labels, mask, n_true, 0.5) == pytest.approx(
        macro_fbeta(pred, truth)
    )


def test_tune_finds_the_separating_threshold():
    pairs = _pairs([(i, i, 0.8) for i in range(10)] + [(i, 100 + i, 0.3) for i in range(10, 20)])
    labels = np.array([True] * 10 + [False] * 10)
    n_true = np.array([1.0] * 10 + [0.0] * 10)
    cfg = DecisionConfig(grid_step=0.05)
    best, curve = tune(pairs, labels, n_true, cfg, 0.5)
    assert 0.3 < best <= 0.8
    assert curve["fbeta"].max() == pytest.approx(1.0)


def test_decide_dispatches_on_strategy():
    pairs = _pairs([(0, 1, 0.9), (1, 1, 0.8)])
    assert decide(
        pairs, DecisionConfig(strategy="threshold", exclusive=True), 0.5, 0.5
    ).tolist() == [True, False]
    assert decide(
        pairs, DecisionConfig(strategy="threshold", exclusive=False), 0.5, 0.5
    ).tolist() == [True, True]
    assert decide(
        pairs, DecisionConfig(strategy="expected_f", exclusive=True), 0.5, 0.0
    ).tolist() == [True, False]
