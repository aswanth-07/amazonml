"""The data contract of the challenge: file names, columns and id conventions."""

from __future__ import annotations

ENTITY_ID = "entity_id"
NAME = "business_name"
ADDRESS = "business_address"
COUNTRY = "country"
SOURCE_COLUMNS: tuple[str, ...] = (ENTITY_ID, NAME, ADDRESS, COUNTRY)

S1_ID = "source1_entity_id"
MATCHED_IDS = "matched_entity_ids"
CANDIDATE_IDS = "candidate_entity_ids"
MATCHING_HEADER: tuple[str, str] = (S1_ID, MATCHED_IDS)
CANDIDATE_HEADER: tuple[str, str] = (S1_ID, CANDIDATE_IDS)

SOURCE_PREFIX = {1: "S1-", 2: "S2-", 3: "S3-"}

MATCHING_FILE = "matching_results.tsv"
CANDIDATE_FILE = "candidate_pairs.tsv"


def source_file(split: str, source: int) -> str:
    return f"{split}_source{source}.tsv"


def ground_truth_file(split: str) -> str:
    return f"{split}_ground_truth.tsv"
