"""Compact, column-oriented record tables built by parallel normalisation.

At challenge scale (2.2M Source 1 and 10M Source 2/3 records per split) a
pandas frame of every normalised string would not fit in memory next to the
sparse matrices, so a table keeps only what blocking and pair features read,
as numpy arrays, and is cached to disk after the first build.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, fields
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ber import schema
from ber.normalize.address import normalize_address, normalize_country
from ber.normalize.name import LEGAL_SET, normalize_name

log = logging.getLogger(__name__)

LEGAL_BITS = {form: 1 << i for i, form in enumerate(sorted(LEGAL_SET))}


@dataclass
class Table:
    ids: np.ndarray  # entity ids (str)
    source: np.ndarray  # int8: 1, 2 or 3
    country: np.ndarray  # normalised country label (str)
    name: np.ndarray  # core name
    alias: np.ndarray  # core of the former/trade name, "" if none
    key: np.ndarray  # consonant key of the core name
    legal_bits: np.ndarray  # int64 bitmask of legal forms
    addr: np.ndarray  # core address (no landmark, no postal code)
    landmark: np.ndarray  # landmark phrase, "" if none
    numbers: np.ndarray  # address numbers, space separated
    first_num: np.ndarray  # int64 first address number, -1 if none
    postal: np.ndarray  # int64 first postal code, -1 if none
    flags: np.ndarray  # int8 bits: 1 script name, 2 domain/hashtag name, 4 name marker

    def __len__(self) -> int:
        return len(self.ids)

    def take(self, idx: np.ndarray) -> Table:
        return Table(**{f.name: getattr(self, f.name)[idx] for f in fields(self)})

    @staticmethod
    def concat(parts: list[Table]) -> Table:
        return Table(
            **{f.name: np.concatenate([getattr(p, f.name) for p in parts]) for f in fields(Table)}
        )

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({f.name: getattr(self, f.name) for f in fields(self)}, path)

    @staticmethod
    def load(path: str | Path) -> Table:
        return Table(**joblib.load(path))


def _normalize_chunk(ids, names, addrs, countries, source: int, transliterate: bool) -> dict:
    n = len(ids)
    out = {
        "name": np.empty(n, object), "alias": np.empty(n, object), "key": np.empty(n, object),
        "legal_bits": np.zeros(n, np.int64), "addr": np.empty(n, object),
        "landmark": np.empty(n, object), "numbers": np.empty(n, object),
        "first_num": np.full(n, -1, np.int64), "postal": np.full(n, -1, np.int64),
        "flags": np.zeros(n, np.int8), "country": np.empty(n, object),
    }  # fmt: skip
    for i, (name, addr, country) in enumerate(zip(names, addrs, countries)):
        c = normalize_country(country)
        nn = normalize_name(name, transliterate)
        na = normalize_address(addr, transliterate, c)
        out["country"][i] = c
        out["name"][i] = nn.core
        out["alias"][i] = nn.alias
        out["key"][i] = nn.key
        bits = 0
        for form in nn.legal.split():
            bits |= LEGAL_BITS.get(form, 0)
        out["legal_bits"][i] = bits
        out["addr"][i] = na.core
        out["landmark"][i] = na.landmark
        out["numbers"][i] = na.numbers
        if na.first and len(na.first) <= 12:
            out["first_num"][i] = int(na.first)
        if na.postal:
            out["postal"][i] = int(na.postal.split()[0])
        out["flags"][i] = (
            (1 if nn.script else 0) | (2 if nn.styled else 0) | (4 if nn.marker else 0)
        )
    out["ids"] = np.asarray(ids, dtype=object)
    out["source"] = np.full(n, source, np.int8)
    clear_caches()  # strings rarely repeat across chunks; caches only cost worker memory
    return out


def clear_caches() -> None:
    from ber.normalize import address, name, text

    for fn in (
        name.normalize_name,
        address.normalize_address,
        text.clean,
        text.skeleton,
        text.consonant_key,
    ):
        fn.cache_clear()


def release_workers() -> None:
    """Shut down joblib's reusable worker pool so its memory is returned now."""
    from joblib.externals.loky import get_reusable_executor

    get_reusable_executor().shutdown(wait=True)


def build_table(
    df: pd.DataFrame, source: int, transliterate: bool = True, n_jobs: int = 12, chunk: int = 50_000
) -> Table:
    """Normalise a source frame into a Table, in parallel over row chunks."""
    ids = df[schema.ENTITY_ID].to_numpy()
    names = df[schema.NAME].to_numpy()
    addrs = df[schema.ADDRESS].to_numpy()
    countries = df[schema.COUNTRY].to_numpy()
    bounds = [(s, min(s + chunk, len(df))) for s in range(0, len(df), chunk)]
    if len(bounds) <= 1 or n_jobs == 1:
        parts = [
            _normalize_chunk(
                ids[a:b], names[a:b], addrs[a:b], countries[a:b], source, transliterate
            )
            for a, b in bounds
        ]
    else:
        parts = Parallel(n_jobs=n_jobs, batch_size=1)(
            delayed(_normalize_chunk)(
                ids[a:b], names[a:b], addrs[a:b], countries[a:b], source, transliterate
            )
            for a, b in bounds
        )
    if not parts:
        parts = [_normalize_chunk([], [], [], [], source, transliterate)]
    return Table(**{f.name: np.concatenate([p[f.name] for p in parts]) for f in fields(Table)})
