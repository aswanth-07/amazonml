"""Pseudo-labelled pairs for self-training on a country the training data lacks.

From finished runs' test probabilities (``test_scores.npz``): confident matches of the
chosen country (p >= --pos) as positives, and each business's highest-scoring
candidates that were confidently rejected (p <= --neg) as hard negatives. With several
``--run`` a pair must be confident in every run (they must share the test candidate set),
so one model's confident mistakes are filtered by the other's doubt. ``--max-pos`` and
``--max-neg`` cap each class by a seeded sample, so the original labelled pairs can stay
the majority of the training mix. Only the provided test inputs and the runs' own
predictions are used: no labels, no external data.

``--drop-decoy-words`` keeps a pair out of the positives when its pool name adds a word only
this country's pool leans on (``ber.vocab``; in France développement, participations,
distribution, holding ...): such records are decoys no model saw in train, so a confident
merge of one is more likely a mistake to be learned than a match.

    python scripts/make_pseudo_labels.py --run runs/<ts>-v3 --run runs/<ts>-v4 \
        --country France --max-pos 450000 --max-neg 750000 --out artifacts/pseudo_france.npz
"""

from __future__ import annotations

import argparse

import numpy as np

from ber.io import read_tsv


def _load(run: str):
    s = np.load(f"{run}/test_scores.npz")
    return s["rows"].astype(np.int64), s["cols"].astype(np.int64), s["p"].astype(np.float32), s


def decoy_word_pairs(args, s, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """True where the pool name adds a word only this country's test pool leans on."""
    import pandas as pd

    from ber.vocab import added_words, pool_heavy, specific_words

    root = f"{args.data_dir}/test/test_source"
    use = ["entity_id", "business_name", "country"]
    s1 = read_tsv(f"{root}1.tsv")[use]
    pool = pd.concat([read_tsv(f"{root}{k}.tsv")[use] for k in (2, 3)], ignore_index=True)
    heavy = pool_heavy(s1["business_name"].to_numpy(), s1["country"].to_numpy(),
                       pool["business_name"].to_numpy(), pool["country"].to_numpy())  # fmt: skip
    words = specific_words(heavy).get(args.country, set())
    n1 = s1.set_index("entity_id")["business_name"].reindex(s["s1_ids"]).to_numpy()
    npool = pool.set_index("entity_id")["business_name"].reindex(s["pool_ids"]).to_numpy()
    bad = np.fromiter((bool(added_words(n1[r], npool[c], words)) for r, c in zip(rows, cols)),
                      bool, len(rows))  # fmt: skip
    print(
        f"{args.country}-specific pool words: {sorted(words)}; {int(bad.sum()):,} of "
        f"{len(rows):,} confident positives add one and are left out"
    )
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="repeat for agreement")
    ap.add_argument("--country", default="France")
    ap.add_argument("--data-dir", default="student_resource/dataset")
    ap.add_argument("--pos", type=float, default=0.97)
    ap.add_argument("--neg", type=float, default=0.03)
    ap.add_argument("--neg-per-entity", type=int, default=4)
    ap.add_argument("--max-pos", type=int, default=0, help="0: keep all")
    ap.add_argument("--max-neg", type=int, default=0, help="0: keep all")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="artifacts/pseudo_france.npz")
    ap.add_argument("--drop-decoy-words", action="store_true",
                    help="no positive whose pool name adds a country-specific pool-heavy word")  # fmt: skip
    args = ap.parse_args()

    rows, cols, p, s = _load(args.run[0])
    lo, hi = p.copy(), p.copy()  # the least and most confident view of each pair
    for run in args.run[1:]:
        r2, c2, p2, _ = _load(run)
        if len(r2) != len(rows) or not (np.array_equal(r2, rows) and np.array_equal(c2, cols)):
            raise SystemExit(
                f"{run}: a different test candidate set; agreement needs the same pairs"
            )
        np.minimum(lo, p2, out=lo)
        np.maximum(hi, p2, out=hi)
    country = (
        read_tsv(f"{args.data_dir}/test/test_source1.tsv")
        .set_index("entity_id")["country"]
        .reindex(s["s1_ids"])
        .to_numpy()
    )
    mine = country[rows] == args.country
    rows, cols, lo, hi = rows[mine], cols[mine], lo[mine], hi[mine]
    rng = np.random.default_rng(args.seed)

    def cap(idx: np.ndarray, k: int) -> np.ndarray:
        return idx if k <= 0 or len(idx) <= k else np.sort(rng.choice(idx, k, replace=False))

    pos = np.flatnonzero(lo >= args.pos)
    if args.drop_decoy_words:
        pos = pos[~decoy_word_pairs(args, s, rows[pos], cols[pos])]
    pos = cap(pos, args.max_pos)
    # hard negatives: per business, its highest-probability candidates every run rejected
    neg = np.flatnonzero(hi <= args.neg)
    order = neg[np.lexsort((-hi[neg], rows[neg]))]
    r = rows[order]
    rank = np.arange(len(r)) - np.searchsorted(r, r)
    neg = cap(order[rank < args.neg_per_entity], args.max_neg)
    out_r = np.concatenate([rows[pos], rows[neg]])
    out_c = np.concatenate([cols[pos], cols[neg]])
    y = np.concatenate([np.ones(len(pos), np.float32), np.zeros(len(neg), np.float32)])
    np.savez(args.out, rows=out_r, cols=out_c, labels=y)
    n_ent = len(np.unique(rows))
    print(
        f"{args.country}: {n_ent:,} businesses, {len(args.run)} run(s); {len(pos):,} "
        f"pseudo-positives, {len(neg):,} hard pseudo-negatives -> {args.out}"
    )
    print(
        "pairs left out as uncertain in some run:", int(((lo < args.pos) & (hi > args.neg)).sum())
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
