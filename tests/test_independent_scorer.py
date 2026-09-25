"""The pipeline's metric agrees with an independent, literal implementation."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from ber.metrics import macro_fbeta

_spec = importlib.util.spec_from_file_location(
    "score_submission", Path(__file__).resolve().parents[1] / "scripts" / "score_submission.py"
)
indep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(indep)


def test_worked_example_from_the_problem_statement():
    pred = {"S2-00047", "S2-00193", "S3-00812"}
    true = {"S2-00047", "S3-00812"}
    assert indep.f05(pred, true) == pytest.approx(0.714, abs=5e-4)


def test_singletons():
    assert indep.f05(set(), set()) == 1.0
    assert indep.f05({"S2-1"}, set()) == 0.0
    assert indep.f05(set(), {"S2-1"}) == 0.0


def test_pipeline_metric_matches_independent_scorer():
    rng = np.random.default_rng(7)
    pool = [f"S2-{i}" for i in range(300)]
    truth, pred = {}, {}
    for e in range(500):
        t = set(rng.choice(pool, rng.integers(0, 5), replace=False).tolist())
        p = set(t)
        if rng.random() < 0.4:  # drop some true matches
            p = set(list(p)[: rng.integers(0, len(p) + 1)])
        if rng.random() < 0.3:  # add wrong ones
            p |= set(rng.choice(pool, rng.integers(1, 3), replace=False).tolist())
        truth[f"S1-{e}"] = t
        pred[f"S1-{e}"] = p
    ours = macro_fbeta(
        {k: sorted(v) for k, v in pred.items()}, {k: sorted(v) for k, v in truth.items()}, 0.5
    )
    assert ours == pytest.approx(indep.score(pred, truth), abs=1e-12)
