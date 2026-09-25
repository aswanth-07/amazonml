"""Where does test differ from train? Unsupervised checks that tell orphaned copies from
extra decoys from a country effect, before anyone tunes a simulation to one of them.

Test has 5.5-5.8 pool records per Source 1 business against 4.68 in train. Two stories fit
that count and call for different remedies:

- **orphans**: businesses left out of test Source 1 whose copies stay in the pool. Their
  copies have no close Source 1 record unless a namesake or neighbour exists, and they come
  in clusters (several copies of one business). Remedy: ``scale.drop_s1_share``.
- **extra decoys**: more neighbour/namesake traps per business. They sit close to a Source 1
  record, so each Source 1 record's crowd of near pool records grows. Remedy: harder
  negatives, a count-aware decision, a stricter threshold.

Checks (numpy only, on the ``ber.embed`` kNN lists of both splits):

1. Each pool record's best Source 1 suitor similarity (reverse list): train matched, train
   unmatched and test, per country, with the test histogram fitted as a mix of the two
   train components. Mass the mix cannot explain at low similarity is orphan-like.
2. Each Source 1 record's crowd: pool records above similarity thresholds (forward list).
   Extra decoys raise it; orphans leave it alone.
3. With ``--run``: the run's final probabilities on test against the train holdout, per
   country: matches per entity, predicted singletons, uncertain-band share, and pool
   records claimed above 0.5 by two suitors before exclusivity.

    python scripts/shift_diagnostics.py --data-dir DATA --emb-dir artifacts/emb [--run runs/<ts>-v3]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ber.embed import countries, emb_paths, load_frames, truth_pairs

BINS = np.linspace(0.0, 1.0, 101)
THRESH = (0.95, 0.9, 0.85, 0.8)


def best_suitor(knn: dict, n_pool: int) -> np.ndarray:
    """Each pool record's best Source 1 similarity from the reverse kNN list (NaN: none)."""
    best = np.full(n_pool, -np.inf, dtype=np.float32)
    np.maximum.at(best, knn["rev_cols"].astype(np.int64), knn["rev_sim"].astype(np.float32))
    best[~np.isfinite(best)] = np.nan
    return best


def crowd(knn: dict, n_s1: int) -> dict[float, np.ndarray]:
    """Per Source 1 record, how many pool records in its forward list reach each threshold."""
    r, s = knn["fwd_rows"].astype(np.int64), knn["fwd_sim"].astype(np.float32)
    return {t: np.bincount(r, weights=(s >= t).astype(float), minlength=n_s1) for t in THRESH}


def hist(x: np.ndarray) -> np.ndarray:
    x = x[np.isfinite(x)]
    h, _ = np.histogram(np.clip(x, 0, 1), BINS)
    return h / max(h.sum(), 1)


def mixture(test: np.ndarray, matched: np.ndarray, unmatched: np.ndarray) -> dict[str, float]:
    """Non-negative least squares of test's histogram on the two train components; the
    positive residual below the matched 1st percentile is the unexplained, orphan-like mass."""
    from scipy.optimize import nnls

    ht, hm, hu = hist(test), hist(matched), hist(unmatched)
    (a, b), _ = nnls(np.column_stack([hm, hu]), ht)
    resid = np.clip(ht - a * hm - b * hu, 0, None)
    low = float(np.nanpercentile(matched, 1)) if np.isfinite(matched).any() else 0.0
    return {"w_matched_like": round(float(a), 4), "w_decoy_like": round(float(b), 4),
            "unexplained_mass": round(float(resid.sum()), 4),
            "unexplained_below_matched_p1": round(float(resid[BINS[:-1] < low].sum()), 4),
            "matched_p1_similarity": round(low, 4)}  # fmt: skip


def split_report(data_dir: str, emb_dir: str, split: str) -> tuple[dict, dict]:
    s1, pool = load_frames(data_dir, split)
    with np.load(emb_paths(emb_dir, split)["knn"]) as z:
        knn = {k: z[k] for k in z.files}
    best = best_suitor(knn, len(pool))
    cr = crowd(knn, len(s1))
    c1, cp = countries(s1), countries(pool)
    arrays = {"best": best, "pool_country": cp, "s1_country": c1, "crowd": cr}
    if split == "train":
        _, b = truth_pairs(data_dir, s1, pool)
        matched = np.zeros(len(pool), dtype=bool)
        matched[b] = True
        arrays["matched"] = matched
    rep: dict[str, dict] = {}
    for c in sorted(set(cp.tolist()) | set(c1.tolist())):
        pm, sm = cp == c, c1 == c
        if not sm.any():
            continue
        row = {"s1": int(sm.sum()), "pool": int(pm.sum()),
               "pool_per_s1": round(float(pm.sum() / max(sm.sum(), 1)), 3),
               "best_suitor_median": round(float(np.nanmedian(best[pm])), 4) if pm.any() else None}  # fmt: skip
        for t in THRESH:
            row[f"pool_best<{t}"] = round(float(np.mean(best[pm] < t)), 4) if pm.any() else None
            row[f"s1_crowd>={t}"] = round(float(cr[t][sm].mean()), 3)
        rep[c] = row
    return rep, arrays


def run_report(run: Path, data_dir: str, split: str) -> dict:
    """Final-probability behaviour per country on a run's scores (train: holdout entities)."""
    from ber.scale import exclusive_adjust

    path = run / f"{split}_scores.npz"
    if not path.exists():
        return {}
    z = np.load(path)
    rows, cols, p = (
        z["rows"].astype(np.int64),
        z["cols"].astype(np.int64),
        z["p"].astype(np.float32),
    )
    s1, _ = load_frames(data_dir, split)
    c1 = countries(s1)
    metrics = json.loads((run / "metrics.json").read_text())
    thr = float(metrics.get("stage2", metrics.get("stage1", {})).get("threshold", 0.5))
    keep = np.ones(len(s1), dtype=bool)
    if split == "train":
        keep = z["eligible"].astype(bool) & ~z["in_sample"].astype(bool)
    adj = exclusive_adjust(rows, cols, p)
    sel = adj >= thr
    n_pred = np.bincount(rows[sel], minlength=len(s1))
    maxp = np.zeros(len(s1), np.float32)
    np.maximum.at(maxp, rows, p)
    contested = p >= 0.5
    claims = np.bincount(cols[contested], minlength=int(cols.max()) + 1)
    rep = {}
    for c in sorted(set(c1.tolist())):
        m = keep & (c1 == c)
        if not m.any():
            continue
        pm = m[rows]
        row = {"entities": int(m.sum()), "threshold": thr,
               "pred_matches_per_entity": round(float(n_pred[m].mean()), 3),
               "pred_singletons": round(float((n_pred[m] == 0).mean()), 4),
               "mean_max_p": round(float(maxp[m].mean()), 4),
               "uncertain_pairs_per_entity": round(float(((p >= 0.1) & (p <= 0.9) & pm).sum() / m.sum()), 3),
               "pool_claimed_by_2plus_above_0.5": int((claims[np.unique(cols[pm & contested])] >= 2).sum())}  # fmt: skip
        if split == "train":
            row["true_matches_per_entity"] = round(float(z["n_true"][m].mean()), 3)
            row["true_singletons"] = round(float((z["n_true"][m] == 0).mean()), 4)
        rep[c] = row
    return rep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--emb-dir", required=True)
    ap.add_argument(
        "--run", default="", help="a finished run with train_scores.npz and test_scores.npz"
    )
    ap.add_argument("--out", default="shift_diagnostics.json")
    args = ap.parse_args()

    out: dict = {}
    tr, tra = split_report(args.data_dir, args.emb_dir, "train")
    te, tea = split_report(args.data_dir, args.emb_dir, "test")
    out["per_country"] = {"train": tr, "test": te}
    matched, best_tr = tra["matched"], tra["best"]
    out["train_components"] = {
        "matched_best_suitor_quantiles": np.nanpercentile(best_tr[matched], [1, 5, 25, 50]).round(4).tolist(),
        "unmatched_best_suitor_quantiles": np.nanpercentile(best_tr[~matched], [1, 5, 25, 50]).round(4).tolist(),
        "unmatched_share": round(float((~matched).mean()), 4),
    }  # fmt: skip
    out["test_mixture"] = {"all": mixture(tea["best"], best_tr[matched], best_tr[~matched])}
    for c in sorted(set(tea["pool_country"].tolist())):
        m = tea["pool_country"] == c
        mt = tra["pool_country"] == c
        comp = (
            (best_tr[matched & mt], best_tr[~matched & mt])
            if mt.any()
            else (best_tr[matched], best_tr[~matched])
        )
        out["test_mixture"][c] = mixture(tea["best"][m], *comp)
    if args.run:
        out["run_train_holdout"] = run_report(Path(args.run), args.data_dir, "train")
        out["run_test"] = run_report(Path(args.run), args.data_dir, "test")

    Path(args.out).write_text(json.dumps(out, indent=2))
    for split in ("train", "test"):
        print(f"\n== {split}: per country")
        print(pd.DataFrame(out["per_country"][split]).T.to_string())
    print("\n== train components (best Source 1 suitor similarity of pool records)")
    print(json.dumps(out["train_components"], indent=2))
    print("\n== test pool as a mix of train's matched and unmatched records")
    print(pd.DataFrame(out["test_mixture"]).T.to_string())
    if args.run:
        for k in ("run_train_holdout", "run_test"):
            print(f"\n== {k}")
            print(pd.DataFrame(out[k]).T.to_string())
    print(f"\nwritten {args.out}")
    print(
        "\nReading it: if test's pool_best<0.85 share and unexplained_below_matched_p1 exceed "
        "train's while s1_crowd stays flat, the extra records are orphan-like (drop_s1_share "
        "is the right simulation). If s1_crowd rises and w_decoy_like grows instead, they are "
        "extra near decoys (tune on harder negatives / stricter decision). A country whose run_test "
        "uncertain_pairs_per_entity or pred_singletons stands out is where the loss is."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
