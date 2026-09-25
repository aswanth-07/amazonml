"""Stands in for ``python -m ber.crossenc score`` and ``python -m ber.listwise score`` in
tests (no torch): scores a pair by the string similarity of the two names and addresses.
As the listwise scorer it also writes the group output (``<out>_grp.npy``): the entity's
best pair score."""

import argparse
import difflib
from pathlib import Path

import numpy as np

from ber.embed import load_frames

ap = argparse.ArgumentParser()
ap.add_argument("-m")
ap.add_argument("command")
ap.add_argument("--model")
ap.add_argument("--data-dir")
ap.add_argument("--split")
ap.add_argument("--pairs")
ap.add_argument("--out")
args = ap.parse_args()
s1, pool = load_frames(args.data_dir, args.split)
a = (s1["business_name"] + " " + s1["business_address"]).str.lower().tolist()
b = (pool["business_name"] + " " + pool["business_address"]).str.lower().tolist()
with np.load(args.pairs) as z:
    rows, cols = z["rows"], z["cols"]
    assert "p1" in z.files and len(z["p1"]) == len(rows)
scores = np.asarray(
    [difflib.SequenceMatcher(None, a[r], b[c]).ratio() for r, c in zip(rows, cols)], np.float32
)
np.save(args.out, scores.astype(np.float16))
if args.m == "ber.listwise":
    best = np.zeros(int(rows.max()) + 1 if len(rows) else 0, np.float32)
    np.maximum.at(best, rows, scores)
    out = Path(args.out)
    np.save(out.with_name(f"{out.stem}_grp.npy"), best[rows].astype(np.float16))
