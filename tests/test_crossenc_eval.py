"""The cross-encoder comparison metrics."""

import numpy as np
import pytest

from ber.crossenc import evaluate


def test_perfect_and_inverted_rankings():
    y = np.array([True, False, True, False, False])
    good = evaluate(np.array([0.9, 0.1, 0.8, 0.2, 0.3]), y)
    assert good["auc"] == 1.0 and good["best_pair_f05"] == pytest.approx(1.0)
    bad = evaluate(np.array([0.1, 0.9, 0.2, 0.8, 0.7]), y)
    assert bad["auc"] == 0.0
    assert good["logloss"] < bad["logloss"]
