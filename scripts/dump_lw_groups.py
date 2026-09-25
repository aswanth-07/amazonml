"""Write ``lw_groups_train.npz`` (the training groups of ``ber.listwise``) from a finished run,
without rerunning it.

The run's stage-1 models score every train pair again from the cached blocking and
feature matrix (so this needs the machine that holds ``artifacts/cache``), the entities are
split exactly as ``ber.scale.run`` splits them, and the listwise share is carved out of the
eligible entities outside the stage-1/2 sample. A later run with the same config plus
``--set scale.lw_share=<share>`` leaves the same share out of its holdout, so the holdout
stays honest once the listwise scores are stage-2 features.

    python scripts/dump_lw_groups.py --run runs/<ts>-v3 --data-dir DATA --share 0.5
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib

from ber import scale
from ber.config import load_config


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run", required=True, help="finished run directory (config.yaml, scale_model.joblib)"
    )
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--share", type=float, default=0.5, help="scale.lw_share")
    ap.add_argument("--min-p1", type=float, default=0.002, help="scale.lw_dump_min_p1")
    ap.add_argument("--out", default="", help="default: <run>/lw_groups_train.npz")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname).1s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")  # fmt: skip

    run = Path(args.run)
    cfg = load_config(run / "config.yaml",
                      [f"scale.lw_share={args.share}", f"scale.lw_dump_min_p1={args.min_p1}"])  # fmt: skip
    s1, pool = scale.load_tables(args.data_dir, "train", cfg)
    truth = scale.load_split_truth(args.data_dir)
    eligible, drop = scale.train_entities(s1.ids, cfg)
    sd = scale.block_split(
        s1, pool, cfg, "train", scale.split_key(args.data_dir, "train", cfg), drop
    )
    labels, n_true = scale.pair_labels(sd, truth)
    _, _, lw, holdout = scale.split_entities(s1.ids, eligible, cfg)
    feats, names = scale.feature_matrix(sd, cfg)
    model = joblib.load(run / "scale_model.joblib")
    p1, _ = scale.score_all(sd, cfg, model.stage1, model.feature_names, feats, names)
    out = Path(args.out) if args.out else run / "lw_groups_train.npz"
    scale.dump_listwise_groups(out, sd, p1, labels, n_true, lw, holdout, cfg)
    print(f"{out}: listwise share {int(lw.sum()):,} entities, holdout {len(holdout):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
