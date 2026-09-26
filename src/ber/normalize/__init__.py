"""Record-level normalisation: raw source rows -> matching-ready columns."""

from __future__ import annotations

import pandas as pd

from ber import schema
from ber.normalize.address import normalize_address, normalize_country
from ber.normalize.name import normalize_name

# Bump whenever normalised output changes for any record: cached tables are keyed on it.
# 2: French rules (N°, elisions, articles, APP/CRS/PASS/Ste), Sep 2026.
NORMALIZE_VERSION = 2

NORMALIZED_COLUMNS = (
    "name_clean",
    "name_core",
    "name_alias",
    "name_legal",
    "name_skeleton",
    "name_acronym",
    "addr_clean",
    "addr_core",
    "addr_landmark",
    "postal",
    "addr_numbers",
    "country_norm",
    "full_text",
    "raw_text",
)


def normalize_records(df: pd.DataFrame, transliterate: bool = True) -> pd.DataFrame:
    """Return ``df`` (index reset) with the normalised columns appended."""
    out = df.reset_index(drop=True).copy()
    names = [normalize_name(v, transliterate) for v in out[schema.NAME]]
    addrs = [normalize_address(v, transliterate) for v in out[schema.ADDRESS]]
    out["name_clean"] = [n.clean for n in names]
    out["name_core"] = [n.core for n in names]
    out["name_alias"] = [n.alias for n in names]
    out["name_legal"] = [n.legal for n in names]
    out["name_skeleton"] = [n.skeleton for n in names]
    out["name_acronym"] = [n.acronym for n in names]
    out["addr_clean"] = [a.clean for a in addrs]
    out["addr_core"] = [a.core for a in addrs]
    out["addr_landmark"] = [a.landmark for a in addrs]
    out["postal"] = [a.postal for a in addrs]
    out["addr_numbers"] = [a.numbers for a in addrs]
    out["country_norm"] = [normalize_country(v) for v in out[schema.COUNTRY]]
    out["full_text"] = [f"{n.core} {a.core}".strip() for n, a in zip(names, addrs)]
    out["raw_text"] = (out[schema.NAME] + ", " + out[schema.ADDRESS]).str.strip(" ,")
    return out
