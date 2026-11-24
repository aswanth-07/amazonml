import numpy as np
import pytest

from ber.decision import macro_from_mask
from ber.metrics import macro_fbeta


def test_macro_from_mask_matches_reference_metric():
    s1_idx = np.array([0, 0, 1, 2])
    pool = np.array(["p0", "p1", "p2", "p3"])
    labels = np.array([True, True, False, False])
    n_true = np.array(
        [3.0, 0.0, 0.0, 1.0]
    )  # S1 #0 has a match blocking missed; #3 has no candidates
    mask = np.array([True, False, True, False])
    s1_ids = ["A", "B", "C", "D"]
    truth = {"A": ["p0", "p1", "missed"], "B": [], "C": [], "D": ["elsewhere"]}
    pred = {sid: pool[mask & (s1_idx == i)].tolist() for i, sid in enumerate(s1_ids)}
    assert macro_from_mask(s1_idx, labels, mask, n_true, 0.5) == pytest.approx(
        macro_fbeta(pred, truth)
    )
