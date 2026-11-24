"""Typed configuration, loaded from YAML with strict key checking.

The dataclass defaults are the complete configuration; ``configs/default.yaml``
restates them for readers and a test keeps the two identical. Overrides use
dotted paths, e.g. ``--set model.n_folds=3 --set scale.k_emb=20``.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).parent / "configs" / "default.yaml"


class ConfigError(ValueError):
    pass


@dataclass
class DataConfig:
    data_dir: str = "dataset"
    quoting: str = "none"  # "none" treats quotes as text; "minimal" honours CSV quoting


@dataclass
class NormalizeConfig:
    transliterate: bool = True  # map non-Latin scripts to ASCII with anyascii


@dataclass
class BlockingConfig:
    # a Source 1 record is searched against pool records of its own country (and those with
    # no country); the search itself is configured under ``scale``
    partition_by_country: bool = True


def _default_lgb_params() -> dict[str, Any]:
    return {
        "objective": "binary",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "verbosity": -1,
    }


@dataclass
class ModelConfig:
    """LightGBM for stage 1 and stage 2, cross-fitted over entity folds."""

    n_folds: int = 5
    num_boost_round: int = 2000
    early_stopping_rounds: int = 100
    params: dict[str, Any] = field(default_factory=_default_lgb_params)


@dataclass
class DecisionConfig:
    """Each S2/S3 record goes to at most one S1 entity (exclusive assignment); a rule from
    ``ber.tuning`` tuned for macro F-beta on out-of-fold predictions picks the matches."""

    # the stage-1/2 threshold grid (per-stage holdout reports)
    grid_min: float = 0.02
    grid_max: float = 0.98
    grid_step: float = 0.01
    # tune the rule at test's decoy density: every train decoy (a candidate record with no
    # true pair) counts this many times (ber.tuning.emulate_decoys). Test has ~2x train's
    # decoys per business; 1.0 tunes at train density
    decoy_boost: float = 1.0
    # "country:shift,...": add a log-odds shift to the final probabilities of one country's
    # pairs before deciding (e.g. "france:-0.5" is stricter on France); for a country absent
    # from train, whose probabilities no holdout can calibrate
    country_logit_shift: str = ""
    # countries (comma-separated, lower-case labels) whose same-address copies that only add
    # their country's pool-heavy words get the final probability floored at rescue_floor
    # (ber.rescue: 97-99% true on train; France-adapted scorers were taught otherwise)
    rescue_countries: str = ""
    rescue_floor: float = 0.8
    # "country:word|word,...": same-address copies that swap one Source 1 name word for
    # these words get the rescue floor too (ber.rescue.replace_mask; the generator's replace
    # copies, whose words France's scorers were taught as decoy vocabulary)
    replace_words: str = ""
    # "country:shift": log-odds shift of same-address in-place swaps of one name word for a
    # common Source 1 word (ber.rescue.swap_decoy_mask; train truth 1-10%); swaps to the
    # country's replace_words are never shifted
    swap_decoy_shift: str = ""
    # also swaps whose new word ends the name (not only in-place ones)
    swap_anywhere: bool = False
    # "country:shift,...": log-odds shift of pairs whose pool record has no address and
    # the exact name of a Source 1 name >= 2 businesses of that country share (ber.rescue)
    namesake_shift: str = ""


@dataclass
class ScaleConfig:
    """The pipeline ``python -m ber.cli run`` runs: inverted-index blocking, the embedding
    view, stage 1 on a sample of entities, cross-encoders, stage 2 (stacking)."""

    k: int = 20  # kept per Source 1 entity by combined cosine after re-scoring
    k_name: int = 8  # ... plus its best by name cosine
    k_addr: int = 8  # ... plus its best by address cosine
    k_reverse: int = 1  # ... plus each pool record's best Source 1 suitors
    generate: int = 100  # wide first pass per Source 1 query, combined view (rare tokens only)
    generate_name: int = 50  # wide first pass, name view
    generate_addr: int = 50  # wide first pass, address view
    generate_reverse: int = 5  # wide first pass per pool query (reverse direction); 0 disables
    name_weight: float = (
        0.6  # weight of the name view in the blocking vector (address gets sqrt(1-w^2))
    )
    df_cap: int = 5000  # query tokens more frequent than this in the pool are not searched on
    query_budget: int = (
        6000  # prefix filter: search on a query's rarest tokens up to this summed df; 0 = off
    )
    n_threads: int = 16
    n_jobs: int = 12
    chunk_rows: int = 500
    block_chunk: int = (
        200_000  # S1 queries searched, re-scored and selected per pass (bounds memory)
    )
    train_sample: int = 300_000  # Source 1 entities whose candidates train the matcher
    feature_chunk: int = 2_000_000
    cache_dir: str = "artifacts/cache"
    emb_dir: str = ""  # embeddings + kNN lists from ber.embed; empty: no embedding view
    k_emb: int = 10  # candidates added per Source 1 entity from its embedding kNN list
    k_emb_reverse: int = 1  # ... plus each pool record's best Source 1 suitors by embedding
    ce_model: str = ""  # cross-encoder from ber.crossenc; "" disables the stage-2 ce feature
    ce_python: str = "py -3.13"  # interpreter with torch + transformers that runs it
    ce_min_p1: float = 0.02  # pairs scored by the cross-encoder: stage-1 probability >= this
    ce_wait: int = 0  # seconds to wait for a cross-encoder that is still training
    ce_pool_context: bool = False  # cross-encoder rank/gap/margin among a pool record's suitors
    ce_train_all: bool = False  # train: cross-encode every entity's pairs, not only eligible ones
    # stage 1 as the last blocking stage: only pairs with p1 >= this reach the matching model
    # and candidate_pairs.tsv (0: every blocked pair is a candidate)
    candidate_min_p1: float = 0.0
    # then the small cross-encoder (the first in ce_model) drops pairs it scores below this
    candidate_min_ce: float = 0.0
    # then the second (large) cross-encoder drops the band pairs it scores below this
    candidate_min_ce2: float = 0.0
    stage2_min_p1: float = 0.0  # stage 2 trains only on pairs with p1 >= this (the candidates)
    stage2_extra: float = 0.0  # share of holdout entities stage 2 also learns from
    stage2_bags: int = 1  # stage-2 models per fold, each with its own subsampling seed
    ce_band: str = (
        ""  # "lo,hi": models after the first score only pairs whose first-model score is inside
    )
    stage1_model: str = ""  # a previous run's scale_model.joblib whose stage 1 is reused
    feature_donor_run: str = ""  # a previous run directory whose cached features are reused
    drop_s1_share: float = 0.0  # train only: share of S1 businesses removed (orphan simulation)
    drop_seed: int = 7
    fingerprint: bool = False  # stage-2 source-fingerprint columns (scale.fingerprint_columns)


@dataclass
class Config:
    seed: int = 42
    beta: float = 0.5
    output_dir: str = "output"
    runs_dir: str = "runs"
    artifacts_dir: str = "artifacts"
    data: DataConfig = field(default_factory=DataConfig)
    normalize: NormalizeConfig = field(default_factory=NormalizeConfig)
    blocking: BlockingConfig = field(default_factory=BlockingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    scale: ScaleConfig = field(default_factory=ScaleConfig)

    def to_dict(self) -> dict[str, Any]:
        return _plain(asdict(self))

    def validate(self) -> Config:
        if self.model.n_folds < 2:
            raise ConfigError("model.n_folds must be >= 2")
        return self


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _build(cls: type, data: Any, path: str) -> Any:
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a mapping, got {type(data).__name__}")
    hints = get_type_hints(cls)
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ConfigError(f"{path}: unknown keys {unknown}; allowed {sorted(known)}")
    kwargs = {name: _coerce(hints[name], value, f"{path}.{name}") for name, value in data.items()}
    return cls(**kwargs)


def _coerce(tp: Any, value: Any, path: str) -> Any:
    origin, args = get_origin(tp), get_args(tp)
    if is_dataclass(tp):
        return _build(tp, value, path)
    if origin is list and args and is_dataclass(args[0]):
        if not isinstance(value, list):
            raise ConfigError(f"{path}: expected a list")
        return [_build(args[0], item, f"{path}[{i}]") for i, item in enumerate(value)]
    if origin is tuple:
        return tuple(value)
    return value


def _apply_override(tree: dict[str, Any], dotted: str) -> None:
    if "=" not in dotted:
        raise ConfigError(f"override {dotted!r} must look like key.path=value")
    key, raw = dotted.split("=", 1)
    parts = key.strip().split(".")
    node: Any = tree
    for part in parts[:-1]:
        node = node[int(part)] if isinstance(node, list) else node.setdefault(part, {})
    last = parts[-1]
    value = yaml.safe_load(raw)
    if isinstance(node, list):
        node[int(last)] = value
    else:
        node[last] = value


def from_dict(data: dict[str, Any]) -> Config:
    return _build(Config, data, "config").validate()


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    """Defaults, then the YAML file (if any), then dotted overrides."""
    tree = Config().to_dict()
    if path is not None:
        with open(path, encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
        tree = _merge(tree, loaded)
    for item in overrides or []:
        _apply_override(tree, item)
    return from_dict(tree)


def _merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def dump_config(config: Config, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        yaml.safe_dump(config.to_dict(), fh, sort_keys=False)
