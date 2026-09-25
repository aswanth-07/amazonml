"""Fill the table, blocking and feature caches of one split ahead of a run.

A run finds them and skips straight to scoring, so a split can be prepared in
parallel with training on another. Usage::

    python scripts/prepare_split.py --config src/ber/configs/full.yaml \
        --data-dir student_resource/dataset --split test
"""

from __future__ import annotations

import argparse
import logging

from ber.config import load_config
from ber.scale import block_split, feature_matrix, load_tables, split_key


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--no-features", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname).1s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")  # fmt: skip
    cfg = load_config(args.config, args.set)
    s1, pool = load_tables(args.data_dir, args.split, cfg)
    sd = block_split(s1, pool, cfg, args.split, split_key(args.data_dir, args.split, cfg))
    if not args.no_features:
        feature_matrix(sd, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
