from pathlib import Path

import pytest

from ber.config import DEFAULT_CONFIG_PATH, Config, ConfigError, load_config


def test_default_yaml_mirrors_dataclass_defaults():
    assert load_config(DEFAULT_CONFIG_PATH) == Config()


def test_dotted_overrides():
    cfg = load_config(
        None, ["model.n_folds=3", "blocking.tfidf.0.top_k=40", "decision.threshold=0.6"]
    )
    assert cfg.model.n_folds == 3
    assert cfg.blocking.tfidf[0].top_k == 40
    assert cfg.decision.threshold == 0.6
    assert isinstance(cfg.blocking.tfidf[0].ngram_range, tuple)


def test_yaml_file_merges_over_defaults(tmp_path):
    p = tmp_path / "exp.yaml"
    p.write_text("model:\n  params:\n    num_leaves: 15\n", encoding="utf-8")
    cfg = load_config(p)
    assert cfg.model.params["num_leaves"] == 15
    assert cfg.model.params["objective"] == "binary"  # untouched keys survive


@pytest.mark.parametrize(
    "override",
    [
        "model.n_foldz=3",
        "decision.strategy=magic",
        "model.kind=svm",
        "model.n_folds=1",
        "blocking.max_candidates=0",
    ],
)
def test_invalid_config_is_rejected(override):
    with pytest.raises(ConfigError):
        load_config(None, [override])


@pytest.mark.parametrize(
    "path",
    sorted((Path(__file__).resolve().parents[1] / "configs").glob("*.yaml")),
    ids=lambda p: p.name,
)
def test_shipped_experiment_configs_load(path):
    load_config(path)


def test_round_trip_through_dict():
    cfg = load_config(None, ["seed=7"])
    from ber.config import from_dict

    assert from_dict(cfg.to_dict()) == cfg
