from __future__ import annotations

from pathlib import Path

import pytest

from ber.config import Config, load_config
from ber.synth import SynthConfig, write_synthetic


@pytest.fixture(scope="session")
def synthetic_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A small synthetic dataset in the challenge layout (train: US+India, test: +France)."""
    root = tmp_path_factory.mktemp("synthetic")
    return write_synthetic(root, SynthConfig(train_entities=500, test_entities=350, seed=11))


@pytest.fixture
def fast_cfg(tmp_path: Path) -> Config:
    return load_config(
        None,
        [
            "model.n_folds=3",
            "model.num_boost_round=300",
            "model.early_stopping_rounds=30",
            f"runs_dir={(tmp_path / 'runs').as_posix()}",
            f"artifacts_dir={(tmp_path / 'artifacts').as_posix()}",
            "decision.grid_step=0.05",
        ],
    )


def write_toy_split(root: Path, split: str = "test") -> Path:
    """Three tiny sources for validator and IO tests."""
    d = root / split
    d.mkdir(parents=True, exist_ok=True)
    header = "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
    (d / f"{split}_source1.tsv").write_text(
        header + "S1-00001\tAcme Corp\t1 Main St, Austin, TX 78701\tUS\n"
        "S1-00002\tSri Laxmi Traders\tNo. 5, MG Road, Bengaluru 560001\tIndia\n",
        encoding="utf-8",
        newline="\n",
    )
    (d / f"{split}_source2.tsv").write_text(
        header + "S2-00001\tACME Corporation\t1 Main Street, Austin\tUS\n",
        encoding="utf-8",
        newline="\n",
    )
    (d / f"{split}_source3.tsv").write_text(
        header
        + "S3-00001\tAcme Corp.\t1 Main St\tUS\nS3-00002\tBoulangerie Dupont\t3 rue de Paris, 75001 Paris\tFrance\n",
        encoding="utf-8",
        newline="\n",
    )
    return d
