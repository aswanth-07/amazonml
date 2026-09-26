"""Which models should stage 2 combine? Every combination, measured on the holdout.

A run with ``--set scale.save_stage2=true`` writes every stage-2 column (stage-1
probability in context, the kept similarities, each cross-encoder and listwise score with
its entity-side and pool-side context, coherence) for the pairs with p1 >= 0.001. This
script retrains stage 2 offline on the run's sample with the same grouped folds for each
combination of column *families* (``ber.scale.stage2_family``), applies exclusive
assignment, tunes the decision rule on the sample's out-of-fold probabilities, and reports
the exact macro F0.5 on the holdout entities: the number the handbook compares.

    python scripts/stage2_search.py --run runs/<ts>-v4 [--mode auto|all|greedy] [--monotone]
        [--decoy-boost 2.0]

``--decoy-boost`` tunes and reports every combination at test's decoy density
(``ber.tuning.emulate_decoys``); ``holdout`` is then the emulated holdout and
``holdout_train_density`` the same rule on the holdout as it is.

"p1" and "keep" are always in; every other family is optional. ``auto`` tries all
subsets when there are at most 5 optional families, else a greedy forward selection. The
best combination is printed as the ``--set scale.stage2_exclude=...`` that reproduces it in
``ber run``. Pairs below p1 0.001 count as probability 0 (as stage 2 makes them).
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ber.matcher import group_folds
from ber.scale import exclusive_adjust, stage2_family, stage2_monotone
from ber.tuning import Scored, emulate_decoys, tune

log = logging.getLogger("stage2_search")
BASE = ("p1", "keep")


def fit_predict(x: np.ndarray, names: list[str], y, rows, train_mask, folds, n_folds, params,
                rounds, stop, monotone: bool, bags: int):  # fmt: skip
    """Out-of-fold probabilities for the training pairs, fold-average for the rest."""
    import lightgbm as lgb

    p = np.zeros(len(x), dtype=np.float64)
    tr_idx = np.flatnonzero(train_mask)
    other = np.flatnonzero(~train_mask)
    cons = stage2_monotone(names) if monotone else None  # as scale.stage2_monotone ships it
    iters = []
    for bag in range(bags):
        for k in range(n_folds):
            tr = tr_idx[folds != k]
            va = tr_idx[folds == k]
            prm = {**params, "seed": 1000 * bag + k, "verbosity": -1, "force_row_wise": True}
            if cons is not None:
                prm["monotone_constraints"] = cons
            dtr = lgb.Dataset(x[tr], label=y[tr], feature_name=names, free_raw_data=True)
            dva = lgb.Dataset(x[va], label=y[va], reference=dtr, free_raw_data=True)
            b = lgb.train(prm, dtr, num_boost_round=rounds, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(stop, verbose=False), lgb.log_evaluation(0)])  # fmt: skip
            iters.append(int(b.best_iteration))
            p[va] += b.predict(x[va], num_iteration=b.best_iteration) / bags
            if len(other):
                p[other] += b.predict(x[other], num_iteration=b.best_iteration) / (n_folds * bags)
    return p.astype(np.float32), iters


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--mode", choices=["auto", "all", "greedy"], default="auto")
    ap.add_argument(
        "--monotone", action="store_true", help="also try monotone constraints on the best"
    )
    ap.add_argument("--bags", type=int, default=1, help="seed bags for the final best combination")
    ap.add_argument("--rounds", type=int, default=3000)
    ap.add_argument("--stop", type=int, default=60)
    ap.add_argument("--lr", type=float, default=0.08)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--only", default="", help="evaluate just this comma-separated family set")
    ap.add_argument("--out", default="", help="results TSV (default: <run>/stage2_search.tsv)")
    ap.add_argument("--decoy-boost", type=float, default=1.0,
                    help="tune and report at this decoy density (test: ~2.0)")  # fmt: skip
    ap.add_argument("--max-sample", type=int, default=0,
                    help="train and tune on this many sample entities (screening); holdout unchanged")  # fmt: skip
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname).1s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")  # fmt: skip

    run = Path(args.run)
    names = json.loads((run / "stage2_inputs.json").read_text())
    mat = np.load(run / "stage2_inputs.npy", mmap_mode="r")
    meta = np.load(run / "stage2_meta.npz")
    rows, cols = meta["rows"].astype(np.int64), meta["cols"].astype(np.int64)
    labels, n_true = meta["labels"].astype(bool), meta["n_true"]
    in_sample, holdout = meta["in_sample"].astype(bool), meta["holdout"].astype(bool)
    n_folds, seed = int(meta["n_folds"]), int(meta["seed"])
    if args.max_sample and in_sample.sum() > args.max_sample:
        keep = np.random.default_rng(seed).choice(
            np.flatnonzero(in_sample), args.max_sample, replace=False
        )
        in_sample = np.zeros_like(in_sample)
        in_sample[keep] = True  # the rest of the sample is scored like any other entity
    sample_ids, holdout_ids = np.flatnonzero(in_sample), np.flatnonzero(holdout)
    train_mask = in_sample[rows]
    folds = group_folds(rows[train_mask], n_folds, seed)
    cfg_params = {"objective": "binary", "learning_rate": args.lr, "num_leaves": 127,
                  "min_data_in_leaf": 200, "feature_fraction": 0.7, "bagging_fraction": 0.7,
                  "bagging_freq": 1, "lambda_l2": 1.0, "max_bin": 127}  # fmt: skip
    if (run / "config.yaml").exists():  # the parameters the run's own stage 2 used
        from ber.config import load_config

        cfg_params.update(load_config(run / "config.yaml").model.params)
    # a leaf needs this many rows: on a small sample (with 70% bagging) the run's value can
    # forbid every split and leave a constant model; at full scale it is unchanged
    n_train = int(train_mask.sum()) * (n_folds - 1) // n_folds
    cfg_params["min_data_in_leaf"] = min(int(cfg_params.get("min_data_in_leaf", 200)),
                                         max(20, n_train // 50))  # fmt: skip
    if args.threads:
        cfg_params["num_threads"] = args.threads

    fam_of = {n: stage2_family(n) for n in names}
    optional = sorted({f for f in fam_of.values() if f not in BASE})
    log.info("%d pairs, %d columns; optional families: %s", len(rows), len(names), optional)
    results: list[dict] = []
    cache: dict[frozenset, dict] = {}

    def evaluate(fams: frozenset, monotone: bool = False, bags: int = 1) -> dict:
        key = frozenset([*fams, f"mono={monotone}", f"bags={bags}"])
        if key in cache:
            return cache[key]
        use = [n for n in names if fam_of[n] in BASE or fam_of[n] in fams]
        t0 = time.time()
        x = np.asarray(mat[:, [names.index(n) for n in use]], dtype=np.float32)
        p, iters = fit_predict(x, use, labels.astype(np.int8), rows, train_mask, folds, n_folds,
                               cfg_params, args.rounds, args.stop, monotone, bags)  # fmt: skip
        adj = exclusive_adjust(rows, cols, p)
        if args.decoy_boost > 1:
            r_e, q_e, y_e, _ = emulate_decoys(rows, cols, adj, labels, args.decoy_boost, seed)
            rule, rep = tune(r_e, q_e, y_e, n_true, sample_ids, holdout_ids)
            plain = Scored.build(rows, adj, labels, n_true, holdout_ids).score(rule)
        else:
            rule, rep = tune(rows, adj, labels, n_true, sample_ids, holdout_ids)
            plain = rep[rule.strategy]["holdout"]
        res = {
            "families": ",".join(sorted(fams)) or "(base)",
            "monotone": monotone, "bags": bags, "n_features": len(use),
            "holdout": rep[rule.strategy]["holdout"], "sample_oof": rep[rule.strategy]["tuned"],
            "holdout_train_density": plain, "decoy_boost": args.decoy_boost,
            "rule": rule.strategy,
            **{f"holdout_{s}": rep[s]["holdout"] for s in ("threshold", "two_threshold", "expected_f")},
            "mean_best_iter": float(np.mean(iters)), "minutes": round((time.time() - t0) / 60, 1),
            "exclude": ",".join(sorted(set(optional) - set(fams))),
        }  # fmt: skip
        log.info("%s", res)
        results.append(res)
        cache[key] = res
        return res

    if args.only:
        evaluate(frozenset(f for f in args.only.split(",") if f and f not in BASE))
    else:
        mode = args.mode
        if mode == "auto":
            mode = "all" if len(optional) <= 5 else "greedy"
        evaluate(frozenset())
        evaluate(frozenset(optional))
        if mode == "all":
            for r in range(1, len(optional)):
                for combo in itertools.combinations(optional, r):
                    evaluate(frozenset(combo))
        else:  # greedy forward selection
            chosen: set[str] = set()
            best = evaluate(frozenset())["holdout"]
            while True:
                trial = {
                    f: evaluate(frozenset(chosen | {f}))["holdout"]
                    for f in optional
                    if f not in chosen
                }
                if not trial:
                    break
                f, score = max(trial.items(), key=lambda kv: kv[1])
                if score <= best:
                    break
                chosen.add(f)
                best = score
        top = max(results, key=lambda r: r["holdout"])
        fams = frozenset(f for f in top["families"].split(",") if f and f != "(base)")
        if args.monotone:
            evaluate(fams, monotone=True)
        if args.bags > 1:
            evaluate(fams, monotone=False, bags=args.bags)

    table = pd.DataFrame(results).sort_values("holdout", ascending=False)
    table.to_csv(args.out or run / "stage2_search.tsv", sep="\t", index=False)
    with pd.option_context("display.max_colwidth", 60, "display.width", 200):
        print(table[["families", "monotone", "bags", "n_features", "holdout", "sample_oof", "rule",
                     "mean_best_iter"]].to_string(index=False))  # fmt: skip
    best = table.iloc[0]
    print(f"\nbest holdout macro F0.5 {best['holdout']:.5f} with {best['families']}")
    print(f"reproduce in ber run:  --set scale.stage2_exclude={best['exclude']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
