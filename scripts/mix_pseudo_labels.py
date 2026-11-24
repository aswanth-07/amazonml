"""Combine pseudo-label files into one training mix (positives from some, negatives from others).

Round 2 of the France self-training takes its positives from the second round
(``make_pseudo_labels_r2.py``) and its negatives from both rounds (round 1's come from the
full candidate set, round 2's only from pairs the cross-encoders scored). A pair that is a
positive anywhere is never a negative; duplicates are dropped; each class is then sampled.

    python scripts/mix_pseudo_labels.py --pos artifacts/pseudo_france_r2.npz \
        --neg artifacts/pseudo_france.npz --neg artifacts/pseudo_france_r2.npz \
        --max-pos 450000 --max-neg 750000 --out artifacts/pseudo_france_r2mix.npz
"""

from __future__ import annotations

import argparse

import numpy as np

KEY = np.int64(20_000_000)  # > any pool position, so row * KEY + col is unique


def _part(path: str, label: float) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as z:
        keep = z["labels"] == label
        return z["rows"][keep].astype(np.int64), z["cols"][keep].astype(np.int64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", action="append", required=True, help="files whose positives to use")
    ap.add_argument("--neg", action="append", required=True, help="files whose negatives to use")
    ap.add_argument(
        "--force-pos", action="append", default=[], help="positives always kept (outside the caps)"
    )
    ap.add_argument("--exclude", action="append", default=[], help="pairs never used as negatives")
    ap.add_argument("--max-pos", type=int, default=0)
    ap.add_argument("--max-neg", type=int, default=0)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pos = np.unique(np.concatenate([r * KEY + c for r, c in (_part(p, 1.0) for p in args.pos)]))
    neg = np.unique(np.concatenate([r * KEY + c for r, c in (_part(p, 0.0) for p in args.neg)]))
    forced = np.unique(
        np.concatenate(
            [np.zeros(0, np.int64)]
            + [r * KEY + c for r, c in (_part(p, 1.0) for p in args.force_pos)]
        )
    )
    barred = [np.load(p) for p in args.exclude]
    barred = np.unique(
        np.concatenate(
            [np.zeros(0, np.int64)] + [z["rows"].astype(np.int64) * KEY + z["cols"] for z in barred]
        )
    )
    neg = neg[~np.isin(neg, pos) & ~np.isin(neg, forced) & ~np.isin(neg, barred)]
    pos = pos[~np.isin(pos, forced)]
    rng = np.random.default_rng(args.seed)
    if args.max_pos and len(pos) > args.max_pos - len(forced):
        pos = rng.choice(pos, max(args.max_pos - len(forced), 0), replace=False)
    pos = np.concatenate([pos, forced])
    if args.max_neg and len(neg) > args.max_neg:
        neg = rng.choice(neg, args.max_neg, replace=False)
    key = np.concatenate([pos, neg])
    y = np.concatenate([np.ones(len(pos), np.float32), np.zeros(len(neg), np.float32)])
    np.savez(args.out, rows=key // KEY, cols=key % KEY, labels=y)
    print(f"{len(pos):,} positives, {len(neg):,} negatives -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
