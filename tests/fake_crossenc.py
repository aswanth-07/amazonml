"""Stands in for ``python -m ber.crossenc score`` in tests (no torch): scores a pair by
the string similarity of the two names and addresses."""

import argparse
import difflib

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
scores = [difflib.SequenceMatcher(None, a[r], b[c]).ratio() for r, c in zip(rows, cols)]
np.save(args.out, np.asarray(scores, dtype=np.float16))
