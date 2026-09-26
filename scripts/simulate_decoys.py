"""Does test's higher decoy ratio explain the public score? A simulation on train scores.

Test has 5.5-5.8 pool records per Source 1 business against 4.68 in train. If that comes
from businesses left out of Source 1 whose copies stay in the pool (orphans), removing
the same share of train businesses from the candidate set reproduces it: their copies
lose their owner and fall to the next-best suitor.

This script works on a finished run's ``train_scores.npz`` with the probabilities held
fixed (the run's pool-side features were computed with the owners present), so it is a
lower bound on the harm. It reports holdout macro F0.5:

- as is (no removal), with the run's chosen decision rule;
- after removal, same rule;
- after removal, with the rule re-tuned under the removal (on the training sample);
- after removal, with probabilities rescaled to the EM-estimated new positive rate
  (Saerens, Latinne and Decaestecker 2002) and the original rule.

    python scripts/simulate_decoys.py --run runs/<ts>-v3 [--share 0.186]

``--clone 2.0`` simulates the shift the data audit found instead (docs/data_audit.md,
section 1: test has ~2x train's decoys per business, of the ordinary kinds, not orphans):
every decoy candidate is cloned (``ber.tuning.emulate_decoys``) and the holdout is scored
with the run's rule and with a rule re-tuned at that density. ``--share 0`` skips the
orphan simulation.

    python scripts/simulate_decoys.py --run runs/<ts>-v5 --share 0 --clone 2.0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ber.scale import exclusive_adjust
from ber.tuning import Rule, Scored, em_prior, emulate_decoys, tune
from ber.tuning import rescale_prior as rescale


def holdout_f(rows, cols, p, labels, n_true, holdout, rule) -> float:
    adj = exclusive_adjust(rows, cols, p)
    s = Scored.build(rows, adj, labels, n_true, holdout)
    return s.score(rule)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--share", type=float, default=0.186, help="share of all S1 businesses removed")
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--clone", type=float, default=0.0, help="decoy density to emulate (test: 2.0)")
    args = ap.parse_args()

    run = Path(args.run)
    s = np.load(run / "train_scores.npz")
    rows, cols, p = (
        s["rows"].astype(np.int64),
        s["cols"].astype(np.int64),
        s["p"].astype(np.float32),
    )
    labels, n_true = s["labels"].astype(bool), s["n_true"]
    eligible, in_sample = s["eligible"], s["in_sample"]
    metrics = json.loads((run / "metrics.json").read_text())
    chosen = metrics.get("decision", {}).get("chosen")
    rule = (
        Rule(chosen["strategy"], chosen["params"])
        if chosen
        else Rule("threshold", {"t": metrics["stage2"]["threshold"]})
    )
    n1 = len(n_true)
    holdout = np.flatnonzero(eligible & ~in_sample)
    sample = np.flatnonzero(in_sample)

    out = {"rule": {"strategy": rule.strategy, "params": rule.params}, "share": args.share}
    out["as_is"] = holdout_f(rows, cols, p, labels, n_true, holdout, rule)
    if args.clone > 1:
        adj = exclusive_adjust(rows, cols, p)
        r_e, q_e, y_e, clones = emulate_decoys(rows, cols, adj, labels, args.clone, args.seed)
        retuned, report = tune(r_e, q_e, y_e, n_true, sample, holdout)
        out["cloned"] = {
            "boost": args.clone, "clones": clones,
            "same_rule": Scored.build(r_e, q_e, y_e, n_true, holdout).score(rule),
            "retuned_rule": report["chosen"],
            "retuned": report[retuned.strategy]["holdout"],
            "retuned_at_train_density": Scored.build(rows, adj, labels, n_true, holdout).score(retuned),
        }  # fmt: skip
        print(f"cloned decoys x{args.clone}: {clones:,} clones")
    if args.share <= 0:
        print(json.dumps(out, indent=2))
        (run / f"simulate_decoys_clone{args.clone}.json").write_text(json.dumps(out, indent=2))
        return 0

    # remove businesses from the encoder share only, so every evaluated entity stays
    rng = np.random.default_rng(args.seed)
    k = round(args.share * n1)
    pool = np.flatnonzero(~eligible)
    drop = np.zeros(n1, dtype=bool)
    drop[rng.choice(pool, min(k, len(pool)), replace=False)] = True
    keep = ~drop[rows]
    orphans = int(np.unique(cols[labels & drop[rows]]).size)
    print(f"removed {int(drop.sum()):,} of {n1:,} S1 businesses; {orphans:,} pool copies orphaned")

    r, c, q, y = rows[keep], cols[keep], p[keep], labels[keep]
    out["removed_same_rule"] = holdout_f(r, c, q, y, n_true, holdout, rule)
    adj = exclusive_adjust(r, c, q)
    retuned, report = tune(r, adj, y, n_true, sample, holdout)
    out["removed_retuned"] = {
        "rule": report["chosen"],
        "holdout": report[retuned.strategy]["holdout"],
    }
    prior_train = float(p.mean())
    prior_new = em_prior(q, prior_train)
    out["em_prior"] = {"train_mean_p": prior_train, "estimated_after_removal": prior_new,
                       "true_rate_after_removal": float(y.mean())}  # fmt: skip
    out["removed_em_adjusted"] = holdout_f(
        r, c, rescale(q, prior_train, prior_new), y, n_true, holdout, rule
    )
    print(json.dumps(out, indent=2))
    (run / f"simulate_decoys_{args.share}.json").write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
