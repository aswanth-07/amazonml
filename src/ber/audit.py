"""Data audit: check the contract and the pipeline's assumptions on the real data.

Run this first when the dataset arrives. Each assumption the pipeline makes by
default is tested here and turned into a recommendation:

* exclusive assignment  - does any Source 2/3 record match more than one S1 entity?
* country partitioning  - do true matches always share a country label?
* TSV quoting           - does any field start with a quote character?
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from ber import schema
from ber.io import Split, load_split
from ber.normalize import normalize_records


def _file_integrity(path: Path) -> dict[str, Any]:
    tab_counts: Counter[int] = Counter()
    quote_leading = 0
    lines = 0
    with open(path, encoding="utf-8-sig") as fh:
        for line in fh:
            line = line.rstrip("\n").rstrip("\r")
            if not line:
                continue
            lines += 1
            tab_counts[line.count("\t")] += 1
            quote_leading += sum(1 for field in line.split("\t") if field.startswith('"'))
    return {
        "lines": lines,
        "tabs_per_line": dict(tab_counts),
        "fields_starting_with_quote": quote_leading,
    }


def _source_profile(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": len(df),
        "empty_rate": {c: float((df[c] == "").mean()) for c in schema.SOURCE_COLUMNS[1:]},
        "countries": df[schema.COUNTRY].value_counts().to_dict(),
        "name_len_median": float(df[schema.NAME].str.len().median()) if len(df) else 0.0,
        "address_len_median": float(df[schema.ADDRESS].str.len().median()) if len(df) else 0.0,
        "non_ascii_name_rate": float((~df[schema.NAME].map(str.isascii)).mean())
        if len(df)
        else 0.0,
    }


def audit_split(split: Split, data_dir: Path | None = None) -> dict[str, Any]:
    report: dict[str, Any] = {"split": split.name, "notes": list(split.notes)}
    if data_dir is not None:
        root = Path(data_dir) / split.name
        report["files"] = {p.name: _file_integrity(p) for p in sorted(root.glob("*.tsv"))}
    report["sources"] = {
        f"s{i}": _source_profile(df) for i, df in ((1, split.s1), (2, split.s2), (3, split.s3))
    }

    if split.truth is None:
        return report

    truth = split.truth
    pool = split.pool
    pool_ids = set(pool[schema.ENTITY_ID])
    all_matches = [m for ms in truth.values() for m in ms]
    counts = Counter(len(ms) for ms in truth.values())
    owner_counts = Counter(all_matches)
    shared = {m: c for m, c in owner_counts.items() if c > 1}
    unknown = [m for m in all_matches if m not in pool_ids]
    matched_pool = set(all_matches)

    gt: dict[str, Any] = {
        "s1_entities": len(truth),
        "singleton_rate": counts.get(0, 0) / max(len(truth), 1),
        "cardinality": {str(k): v for k, v in sorted(counts.items())},
        "true_pairs": len(all_matches),
        "pairs_to_s2": sum(1 for m in all_matches if m.startswith(schema.SOURCE_PREFIX[2])),
        "pairs_to_s3": sum(1 for m in all_matches if m.startswith(schema.SOURCE_PREFIX[3])),
        "unknown_matched_ids": len(unknown),
        "pool_records_matched": len(matched_pool),
        "pool_records_unmatched_rate": 1 - len(matched_pool & pool_ids) / max(len(pool_ids), 1),
        "pool_records_with_multiple_s1": len(shared),
    }

    s1n = normalize_records(split.s1).set_index(schema.ENTITY_ID)
    pooln = normalize_records(pool).set_index(schema.ENTITY_ID)
    pairs = [(s, m) for s, ms in truth.items() for m in ms if m in pooln.index]
    if pairs:
        left = s1n.loc[[s for s, _ in pairs]]
        right = pooln.loc[[m for _, m in pairs]]
        same_country = left["country_norm"].to_numpy() == right["country_norm"].to_numpy()
        postal_both = (left["postal"].to_numpy() != "") & (right["postal"].to_numpy() != "")
        postal_equal = np.array(
            [bool(set(a.split()) & set(b.split())) for a, b in zip(left["postal"], right["postal"])]
        )
        name_sim = np.array(
            [fuzz.token_set_ratio(a, b) for a, b in zip(left["name_core"], right["name_core"])]
        )
        addr_sim = np.array(
            [fuzz.token_set_ratio(a, b) for a, b in zip(left["addr_core"], right["addr_core"])]
        )
        gt["true_pairs_same_country_rate"] = float(same_country.mean())
        gt["true_pairs_postal_agree_rate_when_both_present"] = (
            float(postal_equal[postal_both].mean()) if postal_both.any() else None
        )
        gt["true_pair_name_tset_quantiles"] = _quantiles(name_sim)
        gt["true_pair_addr_tset_quantiles"] = _quantiles(addr_sim)
        cross = [(s, m) for (s, m), ok in zip(pairs, same_country) if not ok][:10]
        gt["cross_country_examples"] = cross
    report["ground_truth"] = gt
    report["recommendations"] = _recommend(report)
    return report


def _quantiles(values: np.ndarray) -> dict[str, float]:
    qs = [0.01, 0.05, 0.1, 0.25, 0.5]
    return {f"q{int(q * 100):02d}": float(np.quantile(values, q)) for q in qs}


def _recommend(report: dict[str, Any]) -> list[str]:
    out = []
    gt = report.get("ground_truth", {})
    shared = gt.get("pool_records_with_multiple_s1", 0)
    if shared == 0:
        out.append(
            "Every S2/S3 record matches at most one S1 entity: keep decision.exclusive=true."
        )
    else:
        out.append(
            f"{shared} S2/S3 records match several S1 entities: test decision.exclusive=false."
        )
    rate = gt.get("true_pairs_same_country_rate")
    if rate is not None:
        if rate >= 0.999:
            out.append(
                "True matches always share a country label: keep blocking.partition_by_country=true."
            )
        else:
            out.append(
                f"Only {rate:.1%} of true matches share a country label: set blocking.partition_by_country=false."
            )
    for name, info in report.get("files", {}).items():
        if info["fields_starting_with_quote"]:
            out.append(
                f"{name}: {info['fields_starting_with_quote']} fields start with a quote; compare data.quoting=none vs minimal."
            )
        if len(info["tabs_per_line"]) > 1:
            out.append(
                f"{name}: inconsistent tab counts per line {info['tabs_per_line']}; inspect before trusting parses."
            )
    if gt.get("unknown_matched_ids"):
        out.append(
            f"{gt['unknown_matched_ids']} ground-truth ids are absent from S2/S3; they cap achievable recall."
        )
    return out


def run_audit(
    data_dir: str | Path, splits: tuple[str, ...] = ("train", "test"), quoting: str = "none"
) -> dict[str, Any]:
    out = {}
    for name in splits:
        split = load_split(data_dir, name, quoting)
        out[name] = audit_split(split, Path(data_dir))
    return out
