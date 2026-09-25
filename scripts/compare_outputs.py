"""Compare test predictions of several versions, per country.

For each version: predicted singleton share and matches per business, by country. For
each pair of versions: how many businesses got a different match list, where, and in
which direction (more or fewer matches). Pairs the public scores with where the versions
differ, to locate what moved the score.

    python scripts/compare_outputs.py output output-v3 output-v4
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import pandas as pd

from ber.io import read_tsv


def load(out_dir: str) -> pd.Series:
    m = read_tsv(Path(out_dir) / "matching_results.tsv")
    return m.set_index("source1_entity_id")["matched_entity_ids"]


def main(dirs: list[str]) -> int:
    country = read_tsv("student_resource/dataset/test/test_source1.tsv").set_index("entity_id")[
        "country"
    ]
    preds = {d: load(d).reindex(country.index).fillna("") for d in dirs}
    rows = []
    for d, s in preds.items():
        n = s.map(lambda v: len(v.split(",")) if v else 0)
        df = pd.DataFrame({"n": n, "country": country})
        for c, g in df.groupby("country"):
            rows.append({"version": d, "country": c, "singletons_%": round(100 * (g.n == 0).mean(), 2),
                         "matches_per_business": round(g.n.mean(), 3)})  # fmt: skip
    print(pd.DataFrame(rows).pivot(index="country", columns="version").to_string(), "\n")
    for a, b in combinations(dirs, 2):
        sa = preds[a].map(lambda v: frozenset(v.split(",")) if v else frozenset())
        sb = preds[b].map(lambda v: frozenset(v.split(",")) if v else frozenset())
        diff = sa != sb
        more = (sb.map(len) > sa.map(len)) & diff
        fewer = (sb.map(len) < sa.map(len)) & diff
        print(f"{a} -> {b}: {int(diff.sum()):,} businesses differ ({100 * diff.mean():.2f}%)")
        print(pd.DataFrame({"differ_%": diff.groupby(country).mean().mul(100).round(2),
                            "more_matches": more.groupby(country).sum(),
                            "fewer_matches": fewer.groupby(country).sum()}).to_string(), "\n")  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:] or ["output", "output-v3", "output-v4"]))
