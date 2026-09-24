"""End to end on synthetic data: the path the real dataset will take."""

import json
from pathlib import Path

import pytest

from ber import schema
from ber.cli import main
from ber.io import read_id_lists
from ber.pipeline import predict_only, run_cv, run_full

pytestmark = pytest.mark.slow


def test_cross_validation_produces_a_sane_score(synthetic_dir, fast_cfg):
    result, run = run_cv(fast_cfg, synthetic_dir, tag="t")
    m = result.metrics
    assert 0.7 < m["fbeta"] <= 1.0
    assert (
        m["fbeta"] <= m["blocking"]["oracle_fbeta"] + 1e-9
    )  # a matcher cannot beat its candidate set
    assert set(m["by_country"]) == {"india", "united states"}
    assert set(m["by_true_cardinality"]) >= {"0", "1"}
    for name in ("metrics.json", "config.yaml", "decision_curve.tsv", "feature_importance.tsv"):
        assert (run.path / name).exists()


def test_full_run_writes_valid_outputs_and_generalises_to_france(
    synthetic_dir, fast_cfg, tmp_path: Path
):
    out = tmp_path / "output"
    summary = run_full(fast_cfg, synthetic_dir, out, tag="t")
    assert summary["test"]["validation_ok"], summary["test"]["validation_errors"]
    assert not summary["test"]["validation_warnings"]

    scored = summary["test"]["scored"]
    assert scored["fbeta"] > 0.7
    assert "france" in scored["by_country"]  # the unseen country is matched, not dropped
    assert scored["by_country"]["france"]["fbeta"] > 0.6

    matches = read_id_lists(out / schema.MATCHING_FILE, schema.S1_ID, schema.MATCHED_IDS)
    cands = read_id_lists(out / schema.CANDIDATE_FILE, schema.S1_ID, schema.CANDIDATE_IDS)
    assert matches.keys() == cands.keys()
    for sid, ids in matches.items():
        assert set(ids) <= set(cands[sid])

    # the saved artifact reproduces the outputs exactly
    again = tmp_path / "again"
    predict_only(Path(fast_cfg.artifacts_dir) / "model.joblib", synthetic_dir, again)
    assert (again / schema.MATCHING_FILE).read_bytes() == (out / schema.MATCHING_FILE).read_bytes()


def test_expected_f_strategy_runs(synthetic_dir, fast_cfg):
    fast_cfg.decision.strategy = "expected_f"
    result, _ = run_cv(fast_cfg, synthetic_dir, tag="ef")
    assert 0.7 < result.metrics["fbeta"] <= 1.0


def test_cli_validate_and_score(synthetic_dir, tmp_path: Path, capsys):
    truth = synthetic_dir / "test" / "test_ground_truth.tsv"
    # the truth file itself is a perfect prediction and a valid candidate set
    m = tmp_path / "m.tsv"
    c = tmp_path / "c.tsv"
    m.write_bytes(truth.read_bytes())
    c.write_text(
        truth.read_text(encoding="utf-8").replace(schema.MATCHED_IDS, schema.CANDIDATE_IDS, 1),
        encoding="utf-8",
    )
    assert (
        main(
            [
                "validate",
                "--matching",
                str(m),
                "--candidates",
                str(c),
                "--test-dir",
                str(synthetic_dir / "test"),
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert main(["score", "--pred", str(m), "--truth", str(truth)]) == 0
    assert json.loads(capsys.readouterr().out)["fbeta"] == 1.0
