"""End-to-end orchestration: prepare -> cross-validate -> fit -> predict -> write."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ber import __version__, schema
from ber.blocking import POOL_IDX, S1_IDX, CandidateGenerator, CandidateSet
from ber.config import Config, dump_config, from_dict
from ber.decision import PROB, decide, to_id_lists, tune
from ber.features import build_features
from ber.io import Split, load_split
from ber.matcher import Matcher
from ber.metrics import (
    blocking_report,
    breakdown,
    cardinality_labels,
    macro_fbeta,
    precision_recall,
)
from ber.normalize import normalize_records
from ber.submission import validate_outputs, write_outputs
from ber.utils import git_revision, seed_everything, timed

log = logging.getLogger(__name__)


@dataclass
class Prepared:
    split: Split
    s1n: pd.DataFrame
    pooln: pd.DataFrame
    cands: CandidateSet
    features: pd.DataFrame
    labels: np.ndarray | None = None
    n_true: np.ndarray | None = None

    @property
    def s1_ids(self) -> np.ndarray:
        return self.s1n[schema.ENTITY_ID].to_numpy()

    @property
    def pool_ids(self) -> np.ndarray:
        return self.pooln[schema.ENTITY_ID].to_numpy()

    def pair_frame(self, probs: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(
            {
                S1_IDX: self.cands.pairs[S1_IDX].to_numpy(),
                POOL_IDX: self.cands.pairs[POOL_IDX].to_numpy(),
                PROB: probs,
            }
        )

    def candidate_lists(self) -> dict[str, list[str]]:
        return self.cands.id_lists(self.s1_ids, self.pool_ids)


def _pair_labels(
    s1_ids: np.ndarray,
    pool_ids: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    truth: dict[str, list[str]],
) -> np.ndarray:
    true_keys = {f"{sid}|{m}" for sid, matches in truth.items() for m in matches}
    keys = pd.Series(s1_ids[rows]).str.cat(pd.Series(pool_ids[cols]), sep="|")
    return keys.isin(true_keys).to_numpy()


def prepare(split: Split, cfg: Config) -> Prepared:
    with timed(f"{split.name}: normalise"):
        s1n = normalize_records(split.s1, cfg.normalize.transliterate)
        pooln = normalize_records(split.pool, cfg.normalize.transliterate)
    with timed(f"{split.name}: blocking"):
        cands = CandidateGenerator(cfg.blocking).generate(s1n, pooln)
    with timed(f"{split.name}: features"):
        feats = build_features(cands, s1n, pooln)
    prepared = Prepared(split, s1n, pooln, cands, feats)
    if split.truth is not None:
        prepared.labels = _pair_labels(
            prepared.s1_ids,
            prepared.pool_ids,
            cands.pairs[S1_IDX].to_numpy(),
            cands.pairs[POOL_IDX].to_numpy(),
            split.truth,
        )
        prepared.n_true = np.array(
            [len(split.truth.get(sid, ())) for sid in prepared.s1_ids], dtype=np.float64
        )
        log.info(
            "%s: %d candidate pairs, %d positive (%.2f%%)",
            split.name,
            len(feats),
            int(prepared.labels.sum()),
            100 * prepared.labels.mean() if len(feats) else 0.0,
        )
    return prepared


@dataclass
class CVResult:
    param: float
    curve: pd.DataFrame
    oof: pd.DataFrame
    predictions: dict[str, list[str]]
    metrics: dict[str, Any]
    matcher: Matcher


def evaluate(prepared: Prepared, predictions: dict[str, list[str]], beta: float) -> dict[str, Any]:
    """Every number worth looking at for a scored split."""
    truth = prepared.split.truth
    assert truth is not None
    candidates = prepared.candidate_lists()
    countries = dict(zip(prepared.s1_ids, prepared.s1n["country_norm"]))
    out: dict[str, Any] = {
        "fbeta": macro_fbeta(predictions, truth, beta),
        **precision_recall(predictions, truth),
        "by_country": breakdown(predictions, truth, countries, beta),
        "by_true_cardinality": breakdown(predictions, truth, cardinality_labels(truth), beta),
        "blocking": {
            **blocking_report(candidates, truth, len(prepared.pooln), beta),
            **prepared.cands.stats,
        },
    }
    precap = prepared.cands.precap
    if precap is not None and len(precap):
        labels = _pair_labels(
            prepared.s1_ids,
            prepared.pool_ids,
            precap[S1_IDX].to_numpy(),
            precap[POOL_IDX].to_numpy(),
            truth,
        )
        n_true = sum(len(v) for v in truth.values()) or 1
        out["blocking"]["precap_pair_recall"] = float(labels.sum() / n_true)
        out["blocking"]["per_blocker_recall"] = {
            col.removeprefix("hit_"): float((labels & (precap[col].to_numpy() > 0)).sum() / n_true)
            for col in prepared.cands.hit_columns
        }
    return out


def cross_validate(prepared: Prepared, cfg: Config) -> CVResult:
    if prepared.labels is None or prepared.n_true is None:
        raise ValueError("cross-validation needs a split with ground truth")
    seed_everything(cfg.seed)
    matcher = Matcher(cfg.model, cfg.seed)
    with timed("cross-fit"):
        oof_probs = matcher.cross_fit(
            prepared.features, prepared.labels.astype(int), prepared.cands.pairs[S1_IDX].to_numpy()
        )
    pairs = prepared.pair_frame(oof_probs)
    if cfg.decision.threshold is None:
        param, curve = tune(pairs, prepared.labels, prepared.n_true, cfg.decision, cfg.beta)
    else:
        param, curve = float(cfg.decision.threshold), pd.DataFrame()
    mask = decide(pairs, cfg.decision, cfg.beta, param)
    predictions = to_id_lists(pairs, mask, prepared.s1_ids, prepared.pool_ids)
    metrics = evaluate(prepared, predictions, cfg.beta)
    metrics["decision"] = {
        "strategy": cfg.decision.strategy,
        "param": param,
        "exclusive": cfg.decision.exclusive,
    }
    metrics["best_iterations"] = matcher.best_iterations
    log.info("out-of-fold macro F%.1f = %.4f (param %.2f)", cfg.beta, metrics["fbeta"], param)
    oof = pairs.assign(label=prepared.labels.astype(np.int8), selected=mask.astype(np.int8))
    return CVResult(param, curve, oof, predictions, metrics, matcher)


class RunDir:
    """``runs/<timestamp>-<tag>/``: config, metrics, curves, errors. One per invocation."""

    def __init__(self, runs_dir: str | Path, tag: str) -> None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.path = Path(runs_dir) / f"{stamp}-{tag}"
        self.path.mkdir(parents=True, exist_ok=True)

    def json(self, name: str, obj: Any) -> None:
        with open(self.path / name, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(obj, fh, indent=2, default=_json_default)

    def frame(self, name: str, df: pd.DataFrame) -> None:
        df.to_csv(self.path / name, sep="\t", index=False, lineterminator="\n")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def error_tables(
    prepared: Prepared, predictions: dict[str, list[str]], limit: int = 2000
) -> dict[str, pd.DataFrame]:
    """Side-by-side false positives, false negatives and blocking misses for inspection."""
    truth = prepared.split.truth or {}
    candidates = {k: set(v) for k, v in prepared.candidate_lists().items()}
    s1 = prepared.split.s1.set_index(schema.ENTITY_ID)
    pool = prepared.split.pool.set_index(schema.ENTITY_ID)
    fp, fn = [], []
    for sid, true in truth.items():
        pred, t = set(predictions.get(sid, ())), set(true)
        fp += [(sid, x, "") for x in sorted(pred - t)]
        fn += [
            (
                sid,
                x,
                "missed_by_blocking"
                if x not in candidates.get(sid, set())
                else "rejected_by_matcher",
            )
            for x in sorted(t - pred)
        ]

    def table(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
        rows = rows[:limit]
        known = [(a, b, c) for a, b, c in rows if b in pool.index]
        return pd.DataFrame(
            {
                "s1_id": [a for a, _, _ in known],
                "s1_name": [s1.at[a, schema.NAME] for a, _, _ in known],
                "s1_address": [s1.at[a, schema.ADDRESS] for a, _, _ in known],
                "s1_country": [s1.at[a, schema.COUNTRY] for a, _, _ in known],
                "other_id": [b for _, b, _ in known],
                "other_name": [pool.at[b, schema.NAME] for _, b, _ in known],
                "other_address": [pool.at[b, schema.ADDRESS] for _, b, _ in known],
                "other_country": [pool.at[b, schema.COUNTRY] for _, b, _ in known],
                "reason": [c for _, _, c in known],
            }
        )

    return {"errors_false_positive.tsv": table(fp), "errors_false_negative.tsv": table(fn)}


def run_cv(cfg: Config, data_dir: str | Path, tag: str = "cv") -> tuple[CVResult, RunDir]:
    run = RunDir(cfg.runs_dir, tag)
    dump_config(cfg, run.path / "config.yaml")
    train = load_split(data_dir, "train", cfg.data.quoting)
    prepared = prepare(train, cfg)
    result = cross_validate(prepared, cfg)
    _record_cv(run, prepared, result)
    return result, run


def _record_cv(run: RunDir, prepared: Prepared, result: CVResult) -> None:
    run.json("metrics.json", {"version": __version__, "git": git_revision(), "cv": result.metrics})
    if len(result.curve):
        run.frame("decision_curve.tsv", result.curve)
    run.frame("feature_importance.tsv", result.matcher.feature_importance())
    for name, df in error_tables(prepared, result.predictions).items():
        run.frame(name, df)
    log.info("run recorded at %s", run.path)


@dataclass
class Artifact:
    matcher: Matcher
    param: float
    config: dict[str, Any]
    version: str = __version__

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: str | Path) -> Artifact:
        return joblib.load(path)


def predict_split(
    prepared: Prepared, artifact: Artifact
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    cfg = from_dict(artifact.config)
    probs = artifact.matcher.predict_proba(prepared.features)
    pairs = prepared.pair_frame(probs)
    mask = decide(pairs, cfg.decision, cfg.beta, artifact.param)
    return to_id_lists(pairs, mask, prepared.s1_ids, prepared.pool_ids), prepared.candidate_lists()


def run_full(
    cfg: Config, data_dir: str | Path, out_dir: str | Path, tag: str = "run"
) -> dict[str, Any]:
    """Train on train, tune on out-of-fold predictions, predict test, write and validate outputs."""
    run = RunDir(cfg.runs_dir, tag)
    dump_config(cfg, run.path / "config.yaml")
    train = load_split(data_dir, "train", cfg.data.quoting)
    test = load_split(data_dir, "test", cfg.data.quoting)

    ptrain = prepare(train, cfg)
    result = cross_validate(ptrain, cfg)
    with timed("final fit"):
        result.matcher.fit(ptrain.features, ptrain.labels.astype(int))
    _record_cv(run, ptrain, result)
    artifact = Artifact(result.matcher, result.param, cfg.to_dict())
    artifact.save(Path(cfg.artifacts_dir) / "model.joblib")
    artifact.save(run.path / "model.joblib")

    ptest = prepare(test, cfg)
    matches, candidates = predict_split(ptest, artifact)
    matching_path, candidate_path = write_outputs(out_dir, matches, candidates, test.s1_ids)
    report = validate_outputs(matching_path, candidate_path, Path(data_dir) / "test")
    print(report.render())

    summary: dict[str, Any] = {
        "version": __version__,
        "git": git_revision(),
        "cv": result.metrics,
        "test": {
            "n_s1": len(test.s1),
            "n_pool": len(ptest.pooln),
            "predicted_pairs": sum(len(v) for v in matches.values()),
            "predicted_singletons": sum(1 for v in matches.values() if not v),
            "candidate_stats": ptest.cands.stats,
            "validation_ok": report.ok,
            "validation_errors": report.errors,
            "validation_warnings": report.warnings,
        },
    }
    if test.truth is not None:  # synthetic data ships test truth; the real test set does not
        summary["test"]["scored"] = evaluate(ptest, matches, cfg.beta)
        log.info("test macro F%.1f = %.4f", cfg.beta, summary["test"]["scored"]["fbeta"])
    run.json("metrics.json", summary)
    log.info("outputs in %s; run recorded at %s", out_dir, run.path)
    return summary


def predict_only(artifact_path: str | Path, data_dir: str | Path, out_dir: str | Path) -> Path:
    artifact = Artifact.load(artifact_path)
    cfg = from_dict(artifact.config)
    test = load_split(data_dir, "test", cfg.data.quoting)
    ptest = prepare(test, cfg)
    matches, candidates = predict_split(ptest, artifact)
    matching_path, _ = write_outputs(out_dir, matches, candidates, test.s1_ids)
    return matching_path
