"""Command line: ``python -m ber.cli <command>`` or ``python src/main.py <command>``.

``run`` trains and predicts one pipeline version; ``fetch-model`` downloads a pretrained
encoder once (``scripts/reproduce.sh`` step 1).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ber import __version__


def _config(args: argparse.Namespace):
    from ber.config import load_config

    cfg = load_config(args.config, args.set)
    if getattr(args, "data_dir", None):
        cfg.data.data_dir = args.data_dir
    return cfg


def _print_summary(summary: dict, beta: float) -> None:
    b = summary["train_blocking"]
    print(
        f"blocking: pair recall {b['pair_recall']:.4f}, oracle F{beta} {b['oracle_fbeta']:.4f}, {b['candidates_per_s1']:.1f} candidates/S1"
    )
    for stage in ("stage1", "stage2"):
        if stage in summary:
            s = summary[stage]
            hold = s.get("holdout", float("nan"))
            print(
                f"{stage}: sample OOF F{beta} {s['sample_oof']:.5f}, holdout F{beta} {hold:.5f} (threshold {s['threshold']:.2f})"
            )
    full = summary["train_full"]
    countries = ", ".join(f"{k}: {v['fbeta']:.4f}" for k, v in full["by_country"].items())
    print(f"train, all entities: F{beta} {full['fbeta']:.5f}; by country {{{countries}}}")


def cmd_run(args: argparse.Namespace) -> int:
    from ber.scale import run

    cfg = _config(args)
    out_dir = args.out_dir or cfg.output_dir
    summary = run(cfg, cfg.data.data_dir, out_dir, args.tag)
    _print_summary(summary, cfg.beta)
    if "scored" in summary["test"]:
        print(f"test macro F{cfg.beta} (synthetic truth): {summary['test']['scored']['fbeta']:.4f}")
    return 0 if summary["test"]["validation_ok"] else 1


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

    sp = sub.add_parser(
        "run", help="full pipeline: train, tune, predict test, write + validate outputs"
    )
    with_config(sp)
    sp.add_argument("--out-dir", help="where to write the two TSVs (default: config output_dir)")
    sp.set_defaults(func=cmd_run)

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
