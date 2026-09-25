"""Offline matcher experiments on a finished run's cached features.

1. Learning curve: stage-1 LightGBM trained on growing numbers of entities, all scored on
   one fixed set of held-out entities.
2. Leave one country out: train on one country, score the other, against training on the
   same country; a proxy for the unseen test country.

Reads ``train_scores.npz`` from a run directory (pair rows, labels, entity sets) and the
train feature matrix the run cached. Scores use the exact macro F0.5 with exclusive
assignment inside the evaluated entities and the best threshold for each model.

    python scripts/learning_curve.py --run runs/<ts>-v2 --features artifacts/cache/train_features_<key>_v3.npy
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np

from ber.config import load_config
from ber.decision import macro_from_mask
from ber.scale import exclusive_adjust


def pairs_of(rows: np.ndarray, entities: np.ndarray) -> np.ndarray:
    mask = np.zeros(rows.max() + 1, dtype=bool)
    mask[entities] = True
    return np.flatnonzero(mask[rows])


def best_f(rows, cols, p, labels, n_true, entities) -> tuple[float, float]:
    adj = exclusive_adjust(rows, cols, p)
    local = np.searchsorted(entities, rows)
    best = (0.0, 0.0)
    for t in np.arange(0.2, 0.96, 0.02):
        mask = (adj >= t) & (adj > 0)
        f = macro_from_mask(local, labels, mask, n_true[entities], 0.5)
        best = max(best, (f, float(t)))
    return best


def fit(x, y, params, rounds):
    n = len(y)
    rng = np.random.default_rng(0)
    va = rng.random(n) < 0.1
    dtr = lgb.Dataset(x[~va], label=y[~va], free_raw_data=True)
    dva = lgb.Dataset(x[va], label=y[va], reference=dtr, free_raw_data=True)
    return lgb.train(params, dtr, num_boost_round=rounds, valid_sets=[dva],
                     callbacks=[lgb.early_stopping(60, verbose=False)])  # fmt: skip


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--config", default="src/ber/configs/full.yaml")
    ap.add_argument("--sizes", default="125000,250000,500000,750000")
    ap.add_argument("--eval-entities", type=int, default=300_000)
    ap.add_argument("--drop", default="", help="comma-separated feature-name prefixes to leave out")
    ap.add_argument("--loco", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config, [])
    params = {**cfg.model.params, "num_threads": 20, "seed": 7, "verbose": -1}
    s = np.load(Path(args.run) / "train_scores.npz")
    rows, cols, labels, n_true = s["rows"], s["cols"], s["labels"], s["n_true"]
    eligible = s["eligible"]
    feats = np.load(args.features, mmap_mode="r")
    names = json.loads(Path(args.features).with_suffix(".json").read_text())
    drops = [d for d in args.drop.split(",") if d]
    use = [i for i, n in enumerate(names) if not any(n.startswith(d) for d in drops)]
    print(f"{len(use)} of {len(names)} features; pairs {len(rows):,}")

    rng = np.random.default_rng(3)
    pool = rng.permutation(np.flatnonzero(eligible))
    evaluate = np.sort(pool[: args.eval_entities])
    rest = pool[args.eval_entities :]
    ev = pairs_of(rows, evaluate)
    xe = np.asarray(feats[ev])[:, use]
    out = {}

    def run(tag, train_entities, eval_entities=None, eval_idx=None, x_eval=None):
        t0 = time.time()
        tr = pairs_of(rows, np.sort(train_entities))
        xt = np.asarray(feats[tr])[:, use]
        model = fit(xt, labels[tr].astype(np.int8), params, 3000)
        del xt
        ents = evaluate if eval_entities is None else eval_entities
        idx = ev if eval_idx is None else eval_idx
        xv = xe if x_eval is None else x_eval
        p = model.predict(xv, num_iteration=model.best_iteration)
        f, t = best_f(rows[idx], cols[idx], p, labels[idx], n_true, ents)
        out[tag] = {"f05": round(f, 5), "threshold": round(t, 2), "iterations": model.best_iteration,
                    "train_pairs": len(tr), "minutes": round((time.time() - t0) / 60, 1)}  # fmt: skip
        print(tag, out[tag], flush=True)

    if args.loco:
        if "country" in s.files:
            country = s["country"]
        else:  # positions follow the Source 1 file order
            from ber.io import read_tsv

            src = Path(
                cfg.data.data_dir if hasattr(cfg.data, "data_dir") else "student_resource/dataset"
            )
            country = (
                read_tsv(src / "train" / "train_source1.tsv")["country"].str.lower().to_numpy()
            )
            country = np.where(country == "us", "united states", country)
        for a, b in (("united states", "india"), ("india", "united states")):
            ea = np.sort(np.intersect1d(rest, np.flatnonzero(country == a)))[:400_000]
            eb = np.sort(np.intersect1d(evaluate, np.flatnonzero(country == b)))
            eb_idx = pairs_of(rows, eb)
            xb = np.asarray(feats[eb_idx])[:, use]
            same = np.sort(np.intersect1d(rest, np.flatnonzero(country == b)))[:400_000]
            run(f"train {a} -> score {b}", ea, eb, eb_idx, xb)
            run(f"train {b} -> score {b}", same, eb, eb_idx, xb)
    else:
        for size in (int(x) for x in args.sizes.split(",")):
            run(f"{size} entities", rest[:size])
    Path(
        args.run, f"offline_{'loco' if args.loco else 'curve'}{'_drop' if drops else ''}.json"
    ).write_text(json.dumps({"drop": drops, **out}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
