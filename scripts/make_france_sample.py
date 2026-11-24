"""A fixed sample of French test pairs for the cross-encoder gate's France report.

From a finished run's cross-encoder pairs (``test_ce_pairs.npz``: pairs with stage-1
p1 >= 0.02), a seeded sample of those whose Source 1 record is French. The gate reports
how many of a model's scores on them fall in (0.1, 0.9); no labels are involved.

    python scripts/make_france_sample.py --run runs/<ts>-v3 --out artifacts/france_ce_sample.npz
"""

from __future__ import annotations

import argparse

import numpy as np

from ber.io import read_tsv


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data-dir", default="student_resource/dataset")
    ap.add_argument("--country", default="France")
    ap.add_argument("--n", type=int, default=150_000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="artifacts/france_ce_sample.npz")
    args = ap.parse_args()
    with np.load(f"{args.run}/test_ce_pairs.npz") as z:
        rows, cols = z["rows"], z["cols"]
    with np.load(f"{args.run}/test_scores.npz") as z:
        s1_ids = z["s1_ids"]
    country = (
        read_tsv(f"{args.data_dir}/test/test_source1.tsv")
        .set_index("entity_id")["country"]
        .reindex(s1_ids)
        .to_numpy()
    )
    mine = np.flatnonzero(country[rows] == args.country)
    rng = np.random.default_rng(args.seed)
    pick = np.sort(rng.choice(mine, min(args.n, len(mine)), replace=False))
    np.savez(args.out, rows=rows[pick], cols=cols[pick])
    print(f"{len(pick):,} of {len(mine):,} {args.country} cross-encoder pairs -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
