"""How many predicted matches add a country's decoy word to the business's name?

French decoys carry words that only the French pool leans on (développement,
participations, distribution, holding, international, associés; ``ber.vocab``), words no
model saw in train. A merge whose pool name adds one of them to the Source 1 name is most
likely a false merge. This counts them per country in a submission, next to the same count
for words every country's pool leans on (copy noise such as "com", "dba"), as a reference.
Only the test inputs and the submission are read.

    python scripts/check_decoy_words.py --matching output/matching_results.tsv \
        --data-dir student_resource/dataset [--out report.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ber import schema
from ber.io import read_tsv
from ber.vocab import added_words, pool_heavy, specific_words


def load_names(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(data_dir) / "test"
    cols = [schema.ENTITY_ID, schema.NAME, schema.COUNTRY]
    s1 = read_tsv(root / schema.source_file("test", 1))[cols]
    pool = pd.concat([read_tsv(root / schema.source_file("test", s))[cols] for s in (2, 3)],
                     ignore_index=True)  # fmt: skip
    return s1, pool


def check(matching: str, s1: pd.DataFrame, pool: pd.DataFrame, n_examples: int = 8) -> dict:
    heavy = pool_heavy(s1[schema.NAME].to_numpy(), s1[schema.COUNTRY].to_numpy(),
                       pool[schema.NAME].to_numpy(), pool[schema.COUNTRY].to_numpy())  # fmt: skip
    specific = specific_words(heavy)
    shared = {c: w - specific[c] for c, w in heavy.items()}
    pred = read_tsv(matching)
    pairs = pred.assign(pid=pred[schema.MATCHED_IDS].str.split(",")).explode("pid")
    pairs = pairs[pairs["pid"].fillna("") != ""]
    s1n = s1.set_index(schema.ENTITY_ID)
    pooln = pool.set_index(schema.ENTITY_ID)[schema.NAME]
    pairs = pairs.assign(
        country=s1n[schema.COUNTRY].reindex(pairs[schema.S1_ID]).to_numpy(),
        n1=s1n[schema.NAME].reindex(pairs[schema.S1_ID]).to_numpy(),
        np_=pooln.reindex(pairs["pid"]).to_numpy(),
    )
    report: dict = {"specific_words": {c: sorted(w) for c, w in specific.items()},
                    "shared_words": {c: sorted(w) for c, w in shared.items()}, "countries": {}}  # fmt: skip
    for c, grp in pairs.groupby("country"):
        spec = [added_words(a, b, specific.get(c, set())) for a, b in zip(grp["n1"], grp["np_"])]
        shar = [added_words(a, b, shared.get(c, set())) for a, b in zip(grp["n1"], grp["np_"])]
        hit = [bool(x) for x in spec]
        ex = grp[hit].head(n_examples)
        report["countries"][str(c)] = {
            "predicted_pairs": len(grp),
            "adds_specific_word": int(sum(hit)),
            "adds_specific_word_share": round(sum(hit) / max(len(grp), 1), 5),
            "businesses_with_one": int(grp.loc[hit, schema.S1_ID].nunique()),
            "adds_shared_word_share": round(sum(bool(x) for x in shar) / max(len(grp), 1), 5),
            "examples": [f"{a}  <-  {b}" for a, b in zip(ex["n1"], ex["np_"])],
        }
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matching", required=True)
    ap.add_argument("--data-dir", default="student_resource/dataset")
    ap.add_argument("--out", default="")
    ap.add_argument("--examples", type=int, default=8)
    args = ap.parse_args()
    s1, pool = load_names(args.data_dir)
    report = check(args.matching, s1, pool, args.examples)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
