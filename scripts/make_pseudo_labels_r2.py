"""Second-round pseudo labels for a country from a run that saved decisions and scorer scores.

A predict-only run keeps its matching file and each cross-encoder's pairs and scores, but no
final probabilities. A pair becomes a pseudo match when that run matched it, every one of
its cross-encoders that scored it is confident (>= --pos), an independent earlier run did
not reject it (p >= --prior-pos), and its pool name adds no country-specific pool-heavy word
(``ber.vocab``: in France développement, participations, holding ...: decoys train never
had). A pseudo non-match is a candidate the run did not match, that every scorer and the
earlier run rejected, among each business's --neg-per-entity highest by the earlier run.
Only the provided test inputs and the runs' own outputs are used: no labels, no external data.

    python scripts/make_pseudo_labels_r2.py --out-dir output-v5 --scores-dir runs/<ts>-v5 --prior runs/<ts>-v3 \
        --country France --max-pos 450000 --max-neg 750000 --out artifacts/pseudo_france_r2.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ber.io import read_tsv
from ber.vocab import added_words, pool_heavy, specific_words


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, help="the run's output folder (matching + scores)")
    ap.add_argument("--prior", required=True, help="an earlier run directory with test_scores.npz")
    ap.add_argument("--country", default="France")
    ap.add_argument("--data-dir", default="student_resource/dataset")
    ap.add_argument("--pos", type=float, default=0.95)
    ap.add_argument("--prior-pos", type=float, default=0.3)
    ap.add_argument("--neg", type=float, default=0.05)
    ap.add_argument("--prior-neg", type=float, default=0.1)
    ap.add_argument("--neg-per-entity", type=int, default=4)
    ap.add_argument("--max-pos", type=int, default=0)
    ap.add_argument("--max-neg", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scores-dir", default="", help="test_ce* pairs/scores (default --out-dir)")
    ap.add_argument(
        "--lookalike-number-only",
        action="store_true",
        help="leave out an added pool-heavy word only when the house number also moved (the "
        "generator's planted lookalikes); same-number additions are true copies on train",
    )
    ap.add_argument(
        "--rescue-positives",
        action="store_true",
        help="matched ber.rescue pairs are positives whatever the scorers say; no rescue pair "
        "is a negative (--rescue-out writes every rescue pair)",
    )
    ap.add_argument("--rescue-out", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    scores_dir = Path(args.scores_dir) if args.scores_dir else out_dir

    prior = np.load(Path(args.prior) / "test_scores.npz")
    s1_ids, pool_ids = prior["s1_ids"], prior["pool_ids"]
    n = np.int64(len(pool_ids))
    root = f"{args.data_dir}/test/test_source"
    use = ["entity_id", "business_name", "business_address", "country"]
    s1 = read_tsv(f"{root}1.tsv")[use].set_index("entity_id").reindex(s1_ids)
    pool = pd.concat([read_tsv(f"{root}{k}.tsv")[use] for k in (2, 3)], ignore_index=True)
    pool = pool.set_index("entity_id").reindex(pool_ids)

    # the run's candidate pairs with every scorer's score (NaN where a scorer did not score)
    with np.load(scores_dir / "test_ce_pairs.npz") as z:
        key = z["rows"].astype(np.int64) * n + z["cols"]
    lo = np.load(scores_dir / "test_ce_scores.npy").astype(np.float32)
    hi = lo.copy()
    kord = np.argsort(key)
    for extra in sorted(scores_dir.glob("test_ce[0-9]*_scores.npy")):
        with np.load(scores_dir / extra.name.replace("_scores.npy", "_pairs.npz")) as z:
            k2 = z["rows"].astype(np.int64) * n + z["cols"]
        s2 = np.load(extra).astype(np.float32)
        at = kord[np.searchsorted(key, k2, sorter=kord)]  # the extra scorer's pairs: a subset
        assert np.array_equal(key[at], k2), extra
        np.minimum.at(lo, at, s2)
        np.maximum.at(hi, at, s2)
    rows, cols = key // n, key % n
    mine = s1["country"].to_numpy()[rows] == args.country
    key, rows, cols, lo, hi = key[mine], rows[mine], cols[mine], lo[mine], hi[mine]

    # the earlier run's view of the same pairs (missing: no opinion)
    pk = prior["rows"].astype(np.int64) * n + prior["cols"]
    order = np.argsort(pk)
    at = np.minimum(np.searchsorted(pk, key, sorter=order), len(pk) - 1)
    hit = pk[order[at]] == key
    pp = np.where(hit, prior["p"][order[at]], np.nan)

    m = read_tsv(out_dir / "matching_results.tsv")
    pos_s1 = np.repeat(
        m["source1_entity_id"].to_numpy(), m["matched_entity_ids"].str.count(",") + 1
    )
    pos_pool = np.concatenate(m["matched_entity_ids"].str.split(",").to_numpy())
    ok = pos_pool != ""
    s1_pos = pd.Index(s1_ids).get_indexer(pos_s1[ok])
    pool_pos = pd.Index(pool_ids).get_indexer(pos_pool[ok])
    matched = np.isin(key, s1_pos.astype(np.int64) * n + pool_pos)

    heavy = pool_heavy(s1["business_name"].to_numpy(), s1["country"].to_numpy(),
                       pool["business_name"].to_numpy(), pool["country"].to_numpy())  # fmt: skip
    words = specific_words(heavy).get(args.country, set())
    n1, n2 = s1["business_name"].to_numpy(), pool["business_name"].to_numpy()
    cand = np.flatnonzero(matched & (lo >= args.pos) & ~(pp < args.prior_pos))
    decoy = np.fromiter(
        (bool(added_words(n1[rows[i]], n2[cols[i]], words)) for i in cand), bool, len(cand)
    )
    if args.lookalike_number_only and len(cand):
        from ber.rescue import first_numbers

        a1 = first_numbers(s1["business_address"].to_numpy())[rows[cand]]
        a2 = first_numbers(pool["business_address"].to_numpy())[cols[cand]]
        decoy &= (a1 != "") & (a2 != "") & (a1 != a2)  # planted: the word and a moved number
    pos = cand[~decoy]
    neg = np.flatnonzero(
        ~matched & (hi <= args.neg) & (np.nan_to_num(pp, nan=1.0) <= args.prior_neg)
    )
    if args.rescue_positives:
        from ber.rescue import rescue_mask

        heavy = pool_heavy(
            s1["business_name"].to_numpy(), s1["country"].fillna("").str.lower().to_numpy(),
            pool["business_name"].fillna("").to_numpy(),
            pool["country"].fillna("").str.lower().to_numpy(),
        )  # fmt: skip
        res = rescue_mask(
            rows, cols, n1, s1["business_address"].to_numpy(), s1["country"].to_numpy(), n2,
            pool["business_address"].to_numpy(), pool["country"].to_numpy(), heavy,
        )  # fmt: skip
        if args.rescue_out:
            np.savez(args.rescue_out, rows=rows[res], cols=cols[res],
                     labels=np.ones(int(res.sum()), np.float32))  # fmt: skip
        pos = np.union1d(pos, np.flatnonzero(res & matched))
        neg = neg[~res[neg]]
        print(
            f"rescue: {int(res.sum()):,} pairs, {int((res & matched).sum()):,} matched -> positives"
        )
    neg = neg[np.lexsort((-pp[neg], rows[neg]))]
    r = rows[neg]
    neg = neg[(np.arange(len(r)) - np.searchsorted(r, r)) < args.neg_per_entity]
    rng = np.random.default_rng(args.seed)
    if args.max_pos and len(pos) > args.max_pos:
        pos = np.sort(rng.choice(pos, args.max_pos, replace=False))
    if args.max_neg and len(neg) > args.max_neg:
        neg = np.sort(rng.choice(neg, args.max_neg, replace=False))
    out_r = np.concatenate([rows[pos], rows[neg]])
    out_c = np.concatenate([cols[pos], cols[neg]])
    y = np.concatenate([np.ones(len(pos), np.float32), np.zeros(len(neg), np.float32)])
    np.savez(args.out, rows=out_r, cols=out_c, labels=y)
    new = int((pp[pos] < 0.97).sum()) if len(pos) else 0
    print(
        f"{args.country}: words {sorted(words)}; {int(decoy.sum()):,} confident matches left out "
        f"for a decoy word; {len(pos):,} pseudo-positives ({new:,} the earlier run was not sure "
        f"of), {len(neg):,} pseudo-negatives -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
