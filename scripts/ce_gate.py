"""Promote an adapted cross-encoder only if it holds up where labels exist.

For each ``--candidate TRAINED=SOURCE=FINAL``: score the held-out US/India pairs (labels)
with the source checkpoint and the adapted one, and a sample of French test pairs (no
labels: the share of scores in (0.1, 0.9) measures how undecided a model is there). The
adapted model moves to FINAL if its held-out log-loss is within ``--tolerance`` of the
source's and its AUC is not lower by more than 5e-5; otherwise the source is copied to
FINAL, so a run waiting on FINAL gets a model either way. Runs in the torch interpreter:

    py -3.13 scripts/ce_gate.py --eval artifacts/ce_eval_pairs.npz \
        --france artifacts/france_ce_sample.npz \
        --candidate models/ce-fr.tmp=models/ce-base=models/ce-fr
"""

from __future__ import annotations

import argparse
import functools
import json
import shutil
from pathlib import Path

import numpy as np

import ber.crossenc as crossenc
from ber.crossenc import evaluate, score

# every score() call reads a split's full text; read each split once (the frames of both
# splits at once are several GB, and a second copy per call ran the machine out of memory)
crossenc.load_frames = functools.lru_cache(maxsize=2)(crossenc.load_frames)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True, help="npz: rows, cols, labels (train split)")
    ap.add_argument("--france", help="npz: rows, cols (test split), unlabelled")
    ap.add_argument("--candidate", action="append", required=True, help="TRAINED=SOURCE=FINAL")
    ap.add_argument("--data-dir", default="student_resource/dataset")
    ap.add_argument("--tolerance", type=float, default=0.05)
    ap.add_argument("--report", default="runs/ce_gate.json")
    args = ap.parse_args()

    with np.load(args.eval) as z:
        rows, cols, y = z["rows"], z["cols"], z["labels"].astype(bool)
    fr = None
    if args.france:
        with np.load(args.france) as z:
            fr = (z["rows"], z["cols"])

    def measure(model: str) -> dict:
        out = evaluate(score(model, args.data_dir, "train", rows, cols), y)
        if fr is not None:
            p = score(model, args.data_dir, "test", *fr).astype(np.float32)
            out["france_uncertain"] = round(float(((p > 0.1) & (p < 0.9)).mean()), 4)
        return out

    report = {}
    for spec in args.candidate:
        trained, source, final = spec.split("=")
        if not (Path(trained) / "crossenc.json").exists():
            print(f"{trained}: not trained; falling back to {source}")
            src, cand, ok = measure(source), None, False
        else:
            src, cand = measure(source), measure(trained)
            ok = (
                cand["logloss"] <= src["logloss"] * (1 + args.tolerance)
                and cand["auc"] >= src["auc"] - 5e-5
            )
        final_p = Path(final)
        if final_p.exists():
            shutil.rmtree(final_p)
        if ok:
            Path(trained).rename(final_p)
        else:  # copy, then rename: a waiting run must never see a half-copied model
            staging = final_p.with_name(final_p.name + ".copying")
            shutil.rmtree(staging, ignore_errors=True)
            shutil.copytree(source, staging)
            staging.rename(final_p)
        report[final] = {"source": src, "adapted": cand, "promoted": ok}
        print(final, "PROMOTED" if ok else "FELL BACK to " + source, json.dumps(report[final]))
    Path(args.report).write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
