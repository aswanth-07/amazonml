"""Reading and writing the challenge's tab-separated files.

Every file is read with an explicit tab separator, as strings, with no NA
inference: a business called "NA" or an empty address must survive as text.
Outputs are written by hand with LF line endings and no quoting, because the
validator expects bare comma-joined id lists and Windows would otherwise emit
CRLF.
"""

from __future__ import annotations

import csv
import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ber import schema

log = logging.getLogger(__name__)

_QUOTING = {"none": csv.QUOTE_NONE, "minimal": csv.QUOTE_MINIMAL}


class DataContractError(ValueError):
    """An input file violates the challenge's data contract."""


def read_tsv(path: str | Path, quoting: str = "none") -> pd.DataFrame:
    if quoting not in _QUOTING:
        raise ValueError(f"quoting must be one of {sorted(_QUOTING)}, got {quoting!r}")
    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        quoting=_QUOTING[quoting],
        encoding="utf-8-sig",
    )
    df.columns = [c.strip() for c in df.columns]
    for col in df.columns:
        df[col] = df[col].str.strip()
    return df


def read_source(path: str | Path, source: int, quoting: str = "none") -> pd.DataFrame:
    df = read_tsv(path, quoting=quoting)
    missing = [c for c in schema.SOURCE_COLUMNS if c not in df.columns]
    if missing:
        raise DataContractError(f"{path}: missing columns {missing}; found {list(df.columns)}")
    prefix = schema.SOURCE_PREFIX[source]
    bad = ~df[schema.ENTITY_ID].str.startswith(prefix)
    if bad.any():
        sample = df.loc[bad, schema.ENTITY_ID].head(3).tolist()
        raise DataContractError(
            f"{path}: {int(bad.sum())} ids lack prefix {prefix!r}, e.g. {sample}"
        )
    dup = df[schema.ENTITY_ID].duplicated()
    if dup.any():
        sample = df.loc[dup, schema.ENTITY_ID].head(3).tolist()
        raise DataContractError(f"{path}: {int(dup.sum())} duplicate entity ids, e.g. {sample}")
    df = df.copy()
    df["source"] = source
    return df.reset_index(drop=True)


def parse_id_list(value: str) -> list[str]:
    return [token.strip() for token in value.split(",") if token.strip()]


def format_id_list(ids: Iterable[str]) -> str:
    return ",".join(ids)


def read_id_lists(path: str | Path, key_col: str, list_col: str) -> dict[str, list[str]]:
    df = read_tsv(path)
    for col in (key_col, list_col):
        if col not in df.columns:
            raise DataContractError(f"{path}: missing column {col!r}; found {list(df.columns)}")
    out: dict[str, list[str]] = {}
    for key, value in zip(df[key_col], df[list_col]):
        if key in out:
            raise DataContractError(f"{path}: duplicate row for {key}")
        out[key] = parse_id_list(value)
    return out


def write_id_lists(
    path: str | Path,
    mapping: Mapping[str, Sequence[str]],
    order: Sequence[str],
    header: tuple[str, str],
) -> Path:
    """Write one row per key in ``order``; keys absent from ``mapping`` get an empty list."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write("\t".join(header) + "\n")
        for key in order:
            fh.write(f"{key}\t{format_id_list(mapping.get(key, ()))}\n")
    return path


@dataclass
class Split:
    """One split (train or test) of the challenge: three sources and optional truth."""

    name: str
    s1: pd.DataFrame
    s2: pd.DataFrame
    s3: pd.DataFrame
    truth: dict[str, list[str]] | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def pool(self) -> pd.DataFrame:
        """Source 2 and Source 3 records, the side that gets matched to Source 1."""
        return pd.concat([self.s2, self.s3], ignore_index=True)


def load_split(data_dir: str | Path, split: str, quoting: str = "none") -> Split:
    """Load ``<data_dir>/<split>/<split>_source{1,2,3}.tsv`` and, if present, the truth file."""
    root = Path(data_dir) / split
    if not root.is_dir():
        raise FileNotFoundError(
            f"{root} does not exist. Put the challenge data at <data_dir>/{split}/ "
            f"(see README: 'Plugging in the dataset')."
        )
    frames = {s: read_source(root / schema.source_file(split, s), s, quoting) for s in (1, 2, 3)}
    truth_path = root / schema.ground_truth_file(split)
    truth = None
    notes: list[str] = []
    if truth_path.exists():
        truth = read_id_lists(truth_path, schema.S1_ID, schema.MATCHED_IDS)
        s1_ids = set(frames[1][schema.ENTITY_ID])
        unknown = [k for k in truth if k not in s1_ids]
        if unknown:
            raise DataContractError(
                f"{truth_path}: {len(unknown)} rows name unknown S1 ids, e.g. {unknown[:3]}"
            )
        missing = s1_ids.difference(truth)
        if missing:
            notes.append(f"{len(missing)} S1 entities have no truth row; treated as singletons")
            for key in missing:
                truth[key] = []
    split_obj = Split(split, frames[1], frames[2], frames[3], truth, notes)
    for note in notes:
        log.warning("%s: %s", split, note)
    return split_obj
