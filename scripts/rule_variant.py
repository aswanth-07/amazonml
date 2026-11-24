"""The decision of a finished run, from its saved scores, without rerunning it.

Reads the run's ``test_scores.npz`` (every pair's final probability and p1; the cross-encoder
scores from the pair files beside it when it lacks them) and its model's decision rule, and
applies, in the pipeline's order (``ber.scale.predict_split``):

1. the candidate cascade (``--candidate-min-p1/-ce/-ce2``): every other pair is set to 0, and
   the pairs kept are ``candidate_pairs.tsv``; France's rule-recovered copies stay candidates;
2. the rule: the model's, or re-tuned on the run's ``train_scores.npz`` with each train decoy
   counted ``--density-boost`` times (test's decoy density);
3. ``--shift`` (a per-country log-odds shift) and France's decision rules (``ber.rescue``);
4. exclusive assignment and the rule; then it writes and validates both output files.

    python scripts/rule_variant.py --scores runs/<ts>-v6a/test_scores.npz \
        --model runs/<ts>-v6a/scale_model.joblib --shift france:-1.0 --out-dir output-v6a-f
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np

from ber.config import load_config
from ber.io import read_tsv
from ber.normalize.address import normalize_country
from ber.scale import (
    adjustment_masks,
    apply_adjustments,
    candidate_mask,
    decide,
    exclusive_adjust,
    id_lists,
    shift_countries,
)
from ber.submission import validate_outputs, write_outputs

CE_MIN_P1 = 0.02  # scale.ce_min_p1: the cross-encoder's pair file holds every pair at or above it


def p1_from_pairs(rows: np.ndarray, cols: np.ndarray, path: Path) -> np.ndarray:
    """Stage-1 probability per pair for a run whose test_scores.npz predates p1: the
    cross-encoder's pair file (test_ce_pairs.npz, next to it) records p1 for every pair with
    p1 >= ce_min_p1; the remaining pairs are below any candidate threshold used here (0)."""
    z = np.load(path)
    n = int(max(cols.max(), z["cols"].max())) + 1
    key = rows.astype(np.int64) * n + cols
    ce_key = z["rows"].astype(np.int64) * n + z["cols"]
    order = np.argsort(ce_key)
    ce_key, ce_p1 = ce_key[order], z["p1"][order]
    pos = np.minimum(np.searchsorted(ce_key, key), len(ce_key) - 1)
    hit = ce_key[pos] == key
    if int(hit.sum()) != len(ce_key):
        raise SystemExit(f"{path}: {len(ce_key) - int(hit.sum())} scored pairs not in the scores")
    return np.where(hit, ce_p1[pos], 0.0).astype(np.float32)


def density_rule(train_scores: Path, boost: float):
    """The two-threshold rule tuned on a run's in-sample training entities with each planted
    lookalike counted ``boost`` times (test has about twice train's); the held-out rest reports."""
    from ber.scale import exclusive_adjust
    from ber.tuning import tune_at_density

    z = np.load(train_scores)
    rows, cols, p, lab, n_true = z["rows"], z["cols"], z["p"], z["labels"], z["n_true"]
    k = (p > 1e-4) | lab
    rows, cols, p, lab = rows[k], cols[k], p[k], lab[k]
    adj = exclusive_adjust(rows, cols, p.astype(np.float32))
    ins, hold = z["in_sample"], z["eligible"] & ~z["in_sample"]
    rule, rep = tune_at_density(
        rows, cols, adj, lab, n_true, np.flatnonzero(ins), np.flatnonzero(hold), 0.5, boost=boost
    )
    d = rep["decoy_boost"]
    if "chosen_rule_holdout_emulated" in d:  # absent when the run held no entities out
        print(
            f"density rule {rule.params}: emulated holdout {d['chosen_rule_holdout_emulated']:.6f} "
            f"vs {d['train_density_rule_holdout_emulated']:.6f} for the train-density rule"
        )
    else:
        print(f"density rule {rule.params} (no held-out entities to report on)")
    return rule


def ce_from_pairs(rows: np.ndarray, cols: np.ndarray, pairs: Path, scores: Path) -> np.ndarray:
    """The small cross-encoder's score per pair, for a run whose test_scores.npz predates
    it: test_ce_scores.npy is aligned with test_ce_pairs.npz; unscored pairs are NaN."""
    z = np.load(pairs)
    sc = np.load(scores).astype(np.float32)
    n = int(max(cols.max(), z["cols"].max())) + 1
    key = rows.astype(np.int64) * n + cols
    ce_key = z["rows"].astype(np.int64) * n + z["cols"]
    order = np.argsort(ce_key)
    ce_key, sc = ce_key[order], sc[order]
    pos = np.minimum(np.searchsorted(ce_key, key), len(ce_key) - 1)
    return np.where(ce_key[pos] == key, sc[pos], np.nan).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument(
        "--shift", default="", help='"country:shift,..." (normalised labels, e.g. france)'
    )
    ap.add_argument("--candidate-min-p1", type=float, default=0.02)
    ap.add_argument(
        "--candidate-min-ce", type=float, default=0.0, help="scale.candidate_min_ce (0: off)"
    )
    ap.add_argument(
        "--candidate-min-ce2", type=float, default=0.0, help="scale.candidate_min_ce2 (0: off)"
    )
    ap.add_argument(
        "--rescue-countries", default="", help="decision.rescue_countries (e.g. france)"
    )
    ap.add_argument("--rescue-floor", type=float, default=0.8)
    ap.add_argument(
        "--replace-words", default="", help='decision.replace_words (e.g. "france:fils|groupe")'
    )
    ap.add_argument(
        "--swap-decoy-shift", default="", help='decision.swap_decoy_shift (e.g. "france:-10")'
    )
    ap.add_argument("--swap-anywhere", action="store_true", help="decision.swap_anywhere")
    ap.add_argument(
        "--namesake-shift", default="", help='decision.namesake_shift (e.g. "france:-2.0")'
    )
    ap.add_argument(
        "--density-boost",
        type=float,
        default=0.0,
        help="tune the two-threshold rule on the run's train_scores.npz (beside --scores) with "
        "each lookalike counted this often (test has ~2x train's; ber.tuning.tune_at_density)",
    )
    ap.add_argument("--data-dir", default="student_resource/dataset")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    z = np.load(args.scores)
    rows, cols, p = z["rows"], z["cols"], z["p"]
    if "p1" in z.files:
        p1 = z["p1"]
    else:
        if args.candidate_min_p1 < CE_MIN_P1:
            raise SystemExit(f"--candidate-min-p1 below {CE_MIN_P1} needs p1 in --scores")
        pairs = Path(args.scores).with_name("test_ce_pairs.npz")
        p1 = p1_from_pairs(rows, cols, pairs)
        print(f"p1 from {pairs}: {int((p1 >= args.candidate_min_p1).sum()):,} candidate pairs")
    s1_ids, pool_ids = z["s1_ids"], z["pool_ids"]
    model = joblib.load(args.model)
    s1 = read_tsv(Path(args.data_dir) / "test" / "test_source1.tsv").set_index("entity_id")
    country = np.array([normalize_country(c) for c in s1["country"].reindex(s1_ids)])
    ce = None
    if args.candidate_min_ce > 0:
        if "ce" in z.files:
            ce = z["ce"]
        else:
            base = Path(args.scores).with_name
            ce = ce_from_pairs(rows, cols, base("test_ce_pairs.npz"), base("test_ce_scores.npy"))
    ce2 = None
    if args.candidate_min_ce2 > 0:  # band pairs only: elsewhere the column repeats ce
        if "ce2" in z.files:
            ce2 = z["ce2"]
        else:
            base = Path(args.scores).with_name
            ce2 = ce_from_pairs(rows, cols, base("test_ce2_pairs.npz"), base("test_ce2_scores.npy"))
    cfg = load_config(
        None,
        [
            f"scale.candidate_min_p1={args.candidate_min_p1}",
            f"scale.candidate_min_ce={args.candidate_min_ce}",
            f"scale.candidate_min_ce2={args.candidate_min_ce2}",
            f"decision.rescue_countries={args.rescue_countries}",
            f"decision.rescue_floor={args.rescue_floor}",
            f"decision.replace_words={args.replace_words}",
            f"decision.swap_decoy_shift={args.swap_decoy_shift}",
            f"decision.swap_anywhere={str(args.swap_anywhere).lower()}",
            f"decision.namesake_shift={args.namesake_shift}",
        ],
    )
    # the pipeline's order (ber.scale.predict_split): adjustment masks, the candidate set
    # (a rescue pair stage 1 keeps stays one), the country shift, then the adjustments
    sd = SimpleNamespace(
        rows=rows,
        cols=cols,
        s1=SimpleNamespace(ids=np.asarray(s1_ids)),
        pool=SimpleNamespace(ids=np.asarray(pool_ids)),
    )
    rescue, delta = adjustment_masks(sd, args.data_dir, "test", cfg)
    keep = candidate_mask(p1, ce, cfg, ce2)
    if keep is None:
        keep = np.ones(len(rows), dtype=bool)
    if rescue is not None:
        keep |= rescue & (p1 >= args.candidate_min_p1)
    probs = np.where(keep, p, 0.0).astype(np.float32)
    rule = model.rule()
    if args.density_boost > 0:
        rule = density_rule(Path(args.scores).with_name("train_scores.npz"), args.density_boost)
    probs = shift_countries(probs, rows, country, args.shift)
    if rescue is not None or delta is not None:
        before = probs >= 0.5
        probs = apply_adjustments(probs, rescue, delta, cfg, rule)
        print(f"adjustments: {int((before != (probs >= 0.5)).sum()):,} pairs crossed 0.5")
    adj = exclusive_adjust(rows, cols, probs)
    mask = decide(rows, adj, rule, 0.5)
    matches = id_lists(rows[mask], cols[mask], adj[mask], s1_ids, pool_ids)
    candidates = id_lists(rows[keep], cols[keep], p1[keep], s1_ids, pool_ids)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    m_path, c_path = write_outputs(out, matches, candidates, list(s1_ids))
    report = validate_outputs(m_path, c_path, Path(args.data_dir) / "test")
    print(report.render())
    by_c = {c: int(mask[country[rows] == c].sum()) for c in np.unique(country)}
    print(f"rule {rule}; shift {args.shift or 'none'}; matched pairs by country {by_c}")
    print(f"candidates per S1 {keep.sum() / len(s1_ids):.3f}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
