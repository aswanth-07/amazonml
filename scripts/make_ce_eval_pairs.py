"""Held-out pairs for comparing cross-encoders (``scripts/ce_gate.py``).

The top-10 token-blocking candidates of randomly chosen train entities outside the
encoder share (no cross-encoder or bi-encoder trained on them), with their labels.

    python scripts/make_ce_eval_pairs.py --blocking artifacts/cache/train_blocking_KEY.joblib
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ber.embed import encoder_mask, load_frames, truth_pairs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocking", required=True)
    ap.add_argument("--data-dir", default="student_resource/dataset")
    ap.add_argument("--entities", type=int, default=60_000)
    ap.add_argument("--per-entity", type=int, default=10)
    ap.add_argument("--out", default="artifacts/ce_eval_pairs.npz")
    args = ap.parse_args()

    blk = joblib.load(args.blocking)
    rows, cols = blk["rows"].astype(np.int64), blk["cols"].astype(np.int64)
    score = blk["base"]["score"]
    del blk
    s1, pool = load_frames(args.data_dir, "train")
    held = ~encoder_mask(s1["entity_id"].to_numpy())
    rng = np.random.default_rng(11)
    pick = np.zeros(len(s1), bool)
    pick[rng.choice(np.flatnonzero(held), args.entities, replace=False)] = True
    m = pick[rows]
    rows, cols, score = rows[m], cols[m], score[m]
    rank = pd.Series(score).groupby(rows).rank(ascending=False, method="first").to_numpy()
    m = rank <= args.per_entity
    rows, cols = rows[m], cols[m]
    a, b = truth_pairs(args.data_dir, s1, pool)
    owner = np.full(len(pool), -1, np.int64)
    owner[b] = a
    labels = owner[cols] == rows
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, rows=rows, cols=cols, labels=labels)
    print(f"{len(rows):,} pairs, {labels.mean():.1%} positive -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
