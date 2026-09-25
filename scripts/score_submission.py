"""Independent leaderboard scorer: standard library only, shares no code with ``ber``.

Implements the problem statement literally: F_0.5 per Source 1 entity, averaged
over every entity in the truth file (singletons included; an empty prediction
for a singleton scores 1.0, any prediction scores 0.0).

    python scripts/score_submission.py --pred matching.tsv --truth ground_truth.tsv
"""

from __future__ import annotations

import argparse
import csv
import sys

BETA2 = 0.25  # beta = 0.5


def read_lists(path: str) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        next(reader)  # header
        for row in reader:
            if not row:
                continue
            ids = row[1] if len(row) > 1 else ""
            out[row[0].strip()] = {x.strip() for x in ids.split(",") if x.strip()}
    return out


def f05(pred: set[str], true: set[str]) -> float:
    if not true:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(true)
    return (1 + BETA2) * p * r / (BETA2 * p + r)


def score(pred: dict[str, set[str]], truth: dict[str, set[str]]) -> float:
    """Macro F0.5 over the entities in ``truth``; an entity missing from ``pred`` predicts nothing."""
    return sum(f05(pred.get(k, set()), v) for k, v in truth.items()) / len(truth)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--truth", required=True)
    args = ap.parse_args()
    truth = read_lists(args.truth)
    pred = read_lists(args.pred)
    print(f"entities {len(truth)}  macro F0.5 {score(pred, truth):.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
