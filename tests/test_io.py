from pathlib import Path

import pytest

from ber import schema
from ber.io import (
    DataContractError,
    load_split,
    read_id_lists,
    read_source,
    read_tsv,
    write_id_lists,
)
from tests.conftest import write_toy_split


def test_read_tsv_keeps_text_literally(tmp_path: Path):
    p = tmp_path / "x.tsv"
    p.write_text(
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        'S1-1\tNA\t12, Main St, "Opp. Park"\tUS\n'
        'S1-2\t"Joe\'s" Diner\t\tIndia\n',
        encoding="utf-8",
    )
    df = read_tsv(p)
    assert df.loc[0, "business_name"] == "NA"  # not NaN
    assert df.loc[0, "business_address"] == '12, Main St, "Opp. Park"'
    assert df.loc[1, "business_name"] == '"Joe\'s" Diner'
    assert df.loc[1, "business_address"] == ""


def test_read_tsv_handles_bom(tmp_path: Path):
    p = tmp_path / "bom.tsv"
    p.write_bytes("﻿entity_id\tbusiness_name\tbusiness_address\tcountry\nS1-1\ta\tb\tUS\n".encode())
    assert read_tsv(p).columns[0] == "entity_id"


def test_write_id_lists_format(tmp_path: Path):
    p = write_id_lists(
        tmp_path / "m.tsv", {"S1-2": ["S2-1", "S3-4"]}, ["S1-1", "S1-2"], schema.MATCHING_HEADER
    )
    raw = p.read_bytes()
    assert b"\r\n" not in raw
    assert raw.decode() == "source1_entity_id\tmatched_entity_ids\nS1-1\t\nS1-2\tS2-1,S3-4\n"
    assert read_id_lists(p, schema.S1_ID, schema.MATCHED_IDS) == {
        "S1-1": [],
        "S1-2": ["S2-1", "S3-4"],
    }


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("entity_id\tbusiness_name\tbusiness_address\tcountry\nS2-1\ta\tb\tUS\n", "prefix"),
        (
            "entity_id\tbusiness_name\tbusiness_address\tcountry\nS1-1\ta\tb\tUS\nS1-1\tc\td\tUS\n",
            "duplicate",
        ),
        ("entity_id\tbusiness_name\tcountry\nS1-1\ta\tUS\n", "missing columns"),
    ],
)
def test_read_source_rejects_contract_violations(tmp_path: Path, content: str, message: str):
    p = tmp_path / "s.tsv"
    p.write_text(content, encoding="utf-8")
    with pytest.raises(DataContractError, match=message):
        read_source(p, 1)


def test_load_split_fills_missing_truth_rows_as_singletons(tmp_path: Path):
    d = write_toy_split(tmp_path, "train")
    (d / "train_ground_truth.tsv").write_text(
        "source1_entity_id\tmatched_entity_ids\nS1-00001\tS2-00001,S3-00001\n", encoding="utf-8"
    )
    split = load_split(tmp_path, "train")
    assert split.truth == {"S1-00001": ["S2-00001", "S3-00001"], "S1-00002": []}
    assert split.notes and "singletons" in split.notes[0]
    assert len(split.pool) == 3
    assert set(split.pool["source"]) == {2, 3}


def test_load_split_rejects_truth_for_unknown_s1(tmp_path: Path):
    d = write_toy_split(tmp_path, "train")
    (d / "train_ground_truth.tsv").write_text(
        "source1_entity_id\tmatched_entity_ids\nS1-99999\t\n", encoding="utf-8"
    )
    with pytest.raises(DataContractError):
        load_split(tmp_path, "train")


def test_load_split_missing_dir_is_explained(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Plugging in the dataset"):
        load_split(tmp_path, "train")
