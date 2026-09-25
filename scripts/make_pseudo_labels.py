"""Pseudo-labelled pairs for self-training on a country the training data lacks.

From a finished run's test probabilities (``test_scores.npz``): confident matches of the
chosen country (p >= --pos) as positives, and each business's highest-scoring
candidates that the run confidently rejected (p <= --neg) as hard negatives. Only the
provided test inputs and the run's own predictions are used: no labels, no external data.

    python scripts/make_pseudo_labels.py --run runs/<ts>-v3 --country France --out artifacts/pseudo_france.npz
"""

from __future__ import annotations

import argparse

import numpy as np

from ber.io import read_tsv


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--country", default="France")
    ap.add_argument("--data-dir", default="student_resource/dataset")
    ap.add_argument("--pos", type=float, default=0.97)
    ap.add_argument("--neg", type=float, default=0.03)
    ap.add_argument("--neg-per-entity", type=int, default=4)
    ap.add_argument("--out", default="artifacts/pseudo_france.npz")
    args = ap.parse_args()

    s = np.load(f"{args.run}/test_scores.npz")
    rows, cols, p = (
        s["rows"].astype(np.int64),
        s["cols"].astype(np.int64),
        s["p"].astype(np.float32),
    )
    country = (
        read_tsv(f"{args.data_dir}/test/test_source1.tsv")
        .set_index("entity_id")["country"]
        .reindex(s["s1_ids"])
        .to_numpy()
    )
    mine = country[rows] == args.country
    rows, cols, p = rows[mine], cols[mine], p[mine]
    pos = p >= args.pos
    # hard negatives: per business, its highest-probability candidates still confidently rejected
    neg_pool = p <= args.neg
    r, c, q = rows[neg_pool], cols[neg_pool], p[neg_pool]
    order = np.lexsort((-q, r))
    r, c = r[order], c[order]
    rank = np.arange(len(r)) - np.searchsorted(r, r)
    keep = rank < args.neg_per_entity
    out_r = np.concatenate([rows[pos], r[keep]])
    out_c = np.concatenate([cols[pos], c[keep]])
    y = np.concatenate([np.ones(pos.sum(), np.float32), np.zeros(keep.sum(), np.float32)])
    np.savez(args.out, rows=out_r, cols=out_c, labels=y)
    n_ent = len(np.unique(rows))
    print(
        f"{args.country}: {n_ent:,} businesses; {int(pos.sum()):,} pseudo-positives, "
        f"{int(keep.sum()):,} hard pseudo-negatives -> {args.out}"
    )
    print("uncertain pairs left out:", int(((p > args.neg) & (p < args.pos)).sum()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
