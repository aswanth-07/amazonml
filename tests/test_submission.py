"""The output writer and the local validator of the submission format."""

from pathlib import Path

import pytest

from ber.submission import validate_outputs, write_outputs
from tests.conftest import write_toy_split

S1 = ["S1-00001", "S1-00002"]


def _write(tmp_path: Path, matching: str, candidates: str) -> tuple[Path, Path]:
    m, c = tmp_path / "m.tsv", tmp_path / "c.tsv"
    m.write_text(matching, encoding="utf-8", newline="\n")
    c.write_text(candidates, encoding="utf-8", newline="\n")
    return m, c


GOOD_M = "source1_entity_id\tmatched_entity_ids\nS1-00001\tS2-00001,S3-00001\nS1-00002\t\n"
GOOD_C = (
    "source1_entity_id\tcandidate_entity_ids\nS1-00001\tS2-00001,S3-00001,S3-00002\nS1-00002\t\n"
)


def test_valid_outputs_pass(tmp_path: Path):
    test_dir = write_toy_split(tmp_path)
    m, c = write_outputs(
        tmp_path / "out", {"S1-00001": ["S2-00001"]}, {"S1-00001": ["S2-00001", "S3-00001"]}, S1
    )
    report = validate_outputs(m, c, test_dir)
    assert report.ok, report.render()
    assert not report.warnings


@pytest.mark.parametrize(
    ("matching", "needle"),
    [
        ("source1_entity_id\tmatched_entity_ids\nS1-00001\tS2-00001\n", "missing"),
        (GOOD_M + "S1-00002\t\n", "duplicate source1_entity_id"),
        (
            "source1_entity_id\tmatched_entity_ids\nS1-00001\tS2-00001,S2-00001\nS1-00002\t\n",
            "duplicate ids",
        ),
        ("source1_entity_id\tmatched_entity_ids\nS1-00001\tS1-00002\nS1-00002\t\n", "Source 1 id"),
        (
            "source1_entity_id\tmatched_entity_ids\nS1-00001\tS2-99999\nS1-00002\t\n",
            "not a Source 2/3 id",
        ),
        ("source1_entity_id\tmatches\nS1-00001\t\nS1-00002\t\n", "header"),
        ('source1_entity_id\tmatched_entity_ids\nS1-00001\t"S2-00001"\nS1-00002\t\n', "quote"),
        (
            "source1_entity_id\tmatched_entity_ids\nS1-00001\tS2-00001,,S3-00001\nS1-00002\t\n",
            "empty element",
        ),
        (GOOD_M + "S1-77777\t\n", "not a Source 1 id"),
    ],
)
def test_each_rule_is_enforced(tmp_path: Path, matching: str, needle: str):
    test_dir = write_toy_split(tmp_path)
    m, c = _write(tmp_path, matching, GOOD_C)
    report = validate_outputs(m, c, test_dir)
    assert not report.ok
    assert any(needle in e for e in report.errors), report.render()


def test_match_outside_candidates_is_a_warning(tmp_path: Path):
    test_dir = write_toy_split(tmp_path)
    m, c = _write(
        tmp_path,
        GOOD_M,
        "source1_entity_id\tcandidate_entity_ids\nS1-00001\tS2-00001\nS1-00002\t\n",
    )
    report = validate_outputs(m, c, test_dir)
    assert report.ok
    assert any("never appeared as candidates" in w for w in report.warnings)
