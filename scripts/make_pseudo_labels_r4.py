"""Round-4 pseudo labels for a country from a submission whose decisions carry the public-
confirmed decision rules, so the scorers stop re-learning the biases those rules correct.

Rounds 1-3 took positives where every French scorer was already confident, so a copy the
scorers wrongly doubted (the replace copies "X EURL & Associés") could never become a
positive, and decoys they wrongly liked (same-address word swaps "Forme Ecole SA" ->
"Forme Amicale SA") were taught as positives. Here:

- positives: the submission's matches in the country; every pair it matches that
  ``--before`` (an earlier submission) did not is always included (the rule-recovered copies);
- negatives: pairs of the run's scores in the country that the submission does not match
  and the run scored below ``--neg-max-p``, hardest first (up to ``--neg-per-entity`` per
  business); every pair ``--before`` matched that the submission dropped is always included
  (the rule-rejected decoys).

Only test inputs and our own outputs are used: no labels, no external data.

    python scripts/make_pseudo_labels_r4.py --matching output-hybrid-L/matching_results.tsv \
        --before output-v6a-fr/matching_results.tsv --scores runs/<ts>-v6a/test_scores.npz \
        --country France --max-pos 400000 --max-neg 500000 --out artifacts/pseudo_france_r4.npz
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from ber.embed import load_frames


def read_pairs(path: str, i1: pd.Index, ip: pd.Index) -> tuple[np.ndarray, np.ndarray]:
    m = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    a, b = [], []
    for x, s in zip(m.iloc[:, 0], m.iloc[:, 1]):
        for y in s.split(","):
            if y:
                a.append(x)
                b.append(y)
    return i1.get_indexer(a).astype(np.int64), ip.get_indexer(b).astype(np.int64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matching", required=True)
    ap.add_argument("--before", required=True, help="an earlier submission's matching file")
    ap.add_argument("--scores", required=True, help="a run's test_scores.npz (rows, cols, p)")
    ap.add_argument("--country", default="France")
    ap.add_argument("--data-dir", default="student_resource/dataset")
    ap.add_argument("--neg-max-p", type=float, default=0.3)
    ap.add_argument("--neg-per-entity", type=int, default=4)
    ap.add_argument("--max-pos", type=int, default=400000)
    ap.add_argument("--max-neg", type=int, default=500000)
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    s1, pool = load_frames(args.data_dir, "test")
    i1, ip = pd.Index(s1["entity_id"]), pd.Index(pool["entity_id"])
    n = len(ip)
    c1 = s1["country"].fillna("").str.strip().str.lower().to_numpy()
    country = args.country.strip().lower()
    mr, mc = read_pairs(args.matching, i1, ip)
    br, bc = read_pairs(args.before, i1, ip)
    on_m, on_b = c1[mr] == country, c1[br] == country
    mk = mr[on_m] * n + mc[on_m]
    bk = br[on_b] * n + bc[on_b]
    recovered = np.setdiff1d(mk, bk)  # matched now, not before: rule-recovered copies
    rejected = np.setdiff1d(bk, mk)  # matched before, dropped now: rule-rejected decoys
    rest = np.setdiff1d(mk, recovered)
    k = max(0, args.max_pos - len(recovered))
    pos = np.concatenate([recovered, rng.choice(rest, min(k, len(rest)), replace=False)])
    z = np.load(args.scores)
    zr, zc, zp = z["rows"].astype(np.int64), z["cols"].astype(np.int64), z["p"]
    sel = (c1[zr] == country) & (zp < args.neg_max_p)
    zr, zc, zp = zr[sel], zc[sel], zp[sel]
    zk = zr * n + zc
    keep = ~np.isin(zk, mk)
    zr, zk, zp = zr[keep], zk[keep], zp[keep]
    order = np.lexsort((-zp, zr))  # hardest first within each business
    zr, zk = zr[order], zk[order]
    start = np.r_[0, np.flatnonzero(zr[1:] != zr[:-1]) + 1]
    rank = np.arange(len(zr)) - np.repeat(start, np.diff(np.r_[start, len(zr)]))
    hard = zk[rank < args.neg_per_entity]
    hard = np.setdiff1d(hard, rejected)
    k = max(0, args.max_neg - len(rejected))
    neg = np.concatenate([rejected, rng.choice(hard, min(k, len(hard)), replace=False)])
    keys = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones(len(pos), np.float32), np.zeros(len(neg), np.float32)])
    np.savez(args.out, rows=keys // n, cols=keys % n, labels=labels)
    print(f"{country}: positives {len(pos):,} (rule-recovered {len(recovered):,}); "
          f"negatives {len(neg):,} (rule-rejected {len(rejected):,}); wrote {args.out}")  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
