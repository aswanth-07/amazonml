"""Command line: ``ber <command>`` (or ``python -m ber``, or ``python src/main.py``)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from ber import __version__, schema

log = logging.getLogger("ber")


def _config(args: argparse.Namespace):
    from ber.config import load_config

    cfg = load_config(args.config, args.set)
    if getattr(args, "data_dir", None):
        cfg.data.data_dir = args.data_dir
    return cfg


def cmd_synth(args: argparse.Namespace) -> int:
    from ber.synth import SynthConfig, write_synthetic

    cfg = SynthConfig(
        train_entities=args.train_entities, test_entities=args.test_entities, seed=args.seed
    )
    out = write_synthetic(args.out, cfg)
    print(f"synthetic dataset written to {out}")
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    from ber.audit import run_audit

    report = run_audit(args.data_dir, tuple(args.split), args.quoting)
    text = json.dumps(report, indent=2, default=str)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(text, encoding="utf-8")
    print(text)
    for split, rep in report.items():
        for rec in rep.get("recommendations", []):
            print(f"[{split}] {rec}")
    return 0


def cmd_cv(args: argparse.Namespace) -> int:
    from ber.pipeline import run_cv

    cfg = _config(args)
    result, run = run_cv(cfg, cfg.data.data_dir, args.tag)
    m = result.metrics
    print(f"OOF macro F{cfg.beta}: {m['fbeta']:.4f}  (param {result.param:.2f})")
    print(
        f"blocking oracle F{cfg.beta}: {m['blocking']['oracle_fbeta']:.4f}  pair recall {m['blocking']['pair_recall']:.4f}"
    )
    print(f"by country: { {k: round(v['fbeta'], 4) for k, v in m['by_country'].items()} }")
    print(f"run: {run.path}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from ber.pipeline import run_full

    cfg = _config(args)
    out_dir = args.out_dir or cfg.output_dir
    summary = run_full(cfg, cfg.data.data_dir, out_dir, args.tag)
    print(f"OOF macro F{cfg.beta}: {summary['cv']['fbeta']:.4f}")
    if "scored" in summary["test"]:
        print(f"test macro F{cfg.beta} (synthetic truth): {summary['test']['scored']['fbeta']:.4f}")
    return 0 if summary["test"]["validation_ok"] else 1


def cmd_predict(args: argparse.Namespace) -> int:
    from ber.pipeline import predict_only

    path = predict_only(args.model, args.data_dir, args.out_dir)
    print(f"wrote {path}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    from ber.submission import run_official_validator, validate_outputs

    matching = Path(args.matching)
    candidates = Path(args.candidates)
    report = validate_outputs(matching, candidates, args.test_dir)
    print("local validator:", report.render())
    code = 0 if report.ok else 1
    official = run_official_validator(matching, candidates, Path(args.test_dir))
    if official is None:
        print("official validator: not found (drop student_resource/ into the repo root to enable)")
    else:
        print(f"official validator exit code: {official}")
        code = code or official
    return code


def cmd_score(args: argparse.Namespace) -> int:
    from ber.io import read_id_lists
    from ber.metrics import blocking_report, macro_fbeta, precision_recall

    pred = read_id_lists(args.pred, schema.S1_ID, schema.MATCHED_IDS)
    truth = read_id_lists(args.truth, schema.S1_ID, schema.MATCHED_IDS)
    out = {"fbeta": macro_fbeta(pred, truth, args.beta), **precision_recall(pred, truth)}
    if args.candidates:
        cands = read_id_lists(args.candidates, schema.S1_ID, schema.CANDIDATE_IDS)
        out["blocking"] = blocking_report(cands, truth, n_pool=0, beta=args.beta)
    print(json.dumps(out, indent=2))
    return 0


def cmd_package(args: argparse.Namespace) -> int:
    from ber.submission import package_submission

    test_dir = Path(args.data_dir) / "test" if args.data_dir else None
    path = package_submission(
        args.team, args.repo_root, args.out_dir, args.doc, args.dest, test_dir
    )
    print(f"wrote {path}")
    return 0


def cmd_fetch_model(args: argparse.Namespace) -> int:
    from ber.pretrained import fetch

    dest = Path(args.dest or f"models/{args.name.split('/')[-1]}")
    print(f"downloaded {args.name} to {fetch(args.name, dest)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ber", description="Business entity resolution pipeline")
    p.add_argument("--version", action="version", version=__version__)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    def with_config(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--config", help="YAML config (default: built-in defaults)")
        sp.add_argument(
            "--set",
            action="append",
            default=[],
            metavar="KEY=VALUE",
            help="dotted override, repeatable",
        )
        sp.add_argument("--data-dir", help="dataset root containing train/ and test/")
        sp.add_argument("--tag", default=sp.prog.split()[-1], help="suffix for the runs/ directory")

    sp = sub.add_parser("synth", help="write a synthetic dataset in the challenge format")
    sp.add_argument("--out", default="data/synthetic")
    sp.add_argument("--train-entities", type=int, default=3000)
    sp.add_argument("--test-entities", type=int, default=2000)
    sp.add_argument("--seed", type=int, default=7)
    sp.set_defaults(func=cmd_synth)

    sp = sub.add_parser("audit", help="check the data contract and the pipeline's assumptions")
    sp.add_argument("--data-dir", default="dataset")
    sp.add_argument("--split", nargs="+", default=["train", "test"], choices=["train", "test"])
    sp.add_argument("--quoting", default="none", choices=["none", "minimal"])
    sp.add_argument("--json", help="also write the report here")
    sp.set_defaults(func=cmd_audit)

    sp = sub.add_parser("cv", help="cross-validated score on train (blocking + matcher + decision)")
    with_config(sp)
    sp.set_defaults(func=cmd_cv)

    sp = sub.add_parser(
        "run", help="full pipeline: train, tune, predict test, write + validate outputs"
    )
    with_config(sp)
    sp.add_argument("--out-dir", help="where to write the two TSVs (default: config output_dir)")
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("predict", help="predict test with a saved model artifact")
    sp.add_argument("--model", default="artifacts/model.joblib")
    sp.add_argument("--data-dir", default="dataset")
    sp.add_argument("--out-dir", default="output")
    sp.set_defaults(func=cmd_predict)

    sp = sub.add_parser(
        "validate", help="validate output files (local rules + official script if present)"
    )
    sp.add_argument("--matching", default=f"output/{schema.MATCHING_FILE}")
    sp.add_argument("--candidates", default=f"output/{schema.CANDIDATE_FILE}")
    sp.add_argument("--test-dir", default="dataset/test")
    sp.set_defaults(func=cmd_validate)

    sp = sub.add_parser("score", help="score a matching file against a ground-truth file")
    sp.add_argument("--pred", required=True)
    sp.add_argument("--truth", required=True)
    sp.add_argument("--candidates")
    sp.add_argument("--beta", type=float, default=0.5)
    sp.set_defaults(func=cmd_score)

    sp = sub.add_parser("package", help="build <team>_submission.zip")
    sp.add_argument("--team", required=True)
    sp.add_argument("--out-dir", default="output")
    sp.add_argument("--doc", default="docs/Documentation_template.md")
    sp.add_argument("--dest", default=".")
    sp.add_argument("--repo-root", default=".")
    sp.add_argument(
        "--data-dir", default="dataset", help="validate against <data-dir>/test first ('' to skip)"
    )
    sp.set_defaults(func=cmd_package)

    sp = sub.add_parser(
        "fetch-model", help="download a licence-checked pretrained encoder (setup only)"
    )
    sp.add_argument("name")
    sp.add_argument("--dest")
    sp.set_defaults(func=cmd_fetch_model)
    return p


def main(argv: list[str] | None = None) -> int:
    from ber.utils import setup_logging

    args = build_parser().parse_args(argv)
    setup_logging(args.verbose)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
