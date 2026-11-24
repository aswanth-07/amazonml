import random

import numpy as np
import pytest

from ber.metrics import (
    fbeta_entity,
    fbeta_from_counts,
    macro_fbeta,
    precision_recall,
)


def test_worked_example_from_problem_statement():
    pred = ["S2-00047", "S2-00193", "S3-00812"]
    true = ["S2-00047", "S3-00812"]
    assert fbeta_entity(pred, true) == pytest.approx(0.714, abs=5e-4)
    assert fbeta_entity(pred, true) == pytest.approx(2.5 / 3.5)


@pytest.mark.parametrize(
    ("pred", "true", "expected"),
    [
        ([], [], 1.0),  # correctly predicted singleton
        (["S2-1"], [], 0.0),  # false merge on a singleton
        ([], ["S2-1"], 0.0),  # missed every match
        (["S2-9"], ["S2-1"], 0.0),  # wrong match
        (["S2-1"], ["S2-1"], 1.0),
    ],
)
def test_singleton_and_edge_rules(pred, true, expected):
    assert fbeta_entity(pred, true) == expected


def test_precision_weighs_more_than_recall():
    # one extra wrong id hurts more than one missing right id
    over = fbeta_entity(["a", "b", "x"], ["a", "b"])
    under = fbeta_entity(["a"], ["a", "b"])
    assert under > over


def test_macro_counts_every_truth_entity_and_missing_predictions_are_empty():
    truth = {"S1-1": ["S2-1"], "S1-2": [], "S1-3": ["S3-1"]}
    pred = {"S1-1": ["S2-1"]}
    assert macro_fbeta(pred, truth) == pytest.approx((1.0 + 1.0 + 0.0) / 3)


def test_vectorised_counts_agree_with_reference():
    rng = random.Random(0)
    universe = [f"S2-{i}" for i in range(8)]
    tp, n_pred, n_true, ref = [], [], [], []
    for _ in range(500):
        p = set(rng.sample(universe, rng.randint(0, 4)))
        t = set(rng.sample(universe, rng.randint(0, 4)))
        tp.append(len(p & t))
        n_pred.append(len(p))
        n_true.append(len(t))
        ref.append(fbeta_entity(p, t))
    np.testing.assert_allclose(
        fbeta_from_counts(np.array(tp), np.array(n_pred), np.array(n_true)), ref
    )


def test_precision_recall_diagnostics():
    truth = {"S1-1": ["S2-1"], "S1-2": []}
    out = precision_recall({"S1-1": ["S2-1", "S2-2"]}, truth)
    assert out["pair_precision"] == 0.5
    assert out["pair_recall"] == 1.0
    assert out["singleton_accuracy"] == 1.0
