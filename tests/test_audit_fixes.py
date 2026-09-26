"""The data-audit fixes (docs/data_audit.md): accent-folded model text, French address rules,
tuning at test decoy density, country-specific decoy words, run_all's pinned settings."""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ber.config import load_config

ROOT = Path(__file__).resolve().parents[1]


# ----------------------------------------------------------------------------- model text
@pytest.mark.parametrize(
    ("raw", "folded"),
    [
        ("Çlub Spôrtive de Frânce", "Club Sportive de France"),
        ("Àmicale des Pàrents, Fïls & Prîmaire", "Amicale des Parents, Fils & Primaire"),
        ("Sámson's Grádíng | 12 Rue de l’Église", "Samson's Grading | 12 Rue de l’Eglise"),
        ("Café Ecole", "Cafe Ecole"),  # a separate combining accent on a Latin letter
        ("plain ascii | 5 Main St", "plain ascii | 5 Main St"),
        ("N° 39, 2º étage", "N° 39, 2º etage"),  # degree and ordinal signs have no accent
        ("Straße Œuvre ø", "Straße Œuvre ø"),  # letters without a decomposition stay
    ],
)
def test_fold_latin_drops_only_latin_accents(raw, folded):
    from ber.normalize.text import fold_latin

    assert fold_latin(raw) == folded


@pytest.mark.parametrize(
    "text",
    ["महाराष्ट्र इंडस्ट्रीज़ प्राइवेट लिमिटेड", "சுப்ரீம் மோடர்ன்", "ଡାଇନାମିକ୍ ହୋଟେଲ୍", "పర్‌ఫెక్ట్ యునైటెడ్"],
)
def test_fold_latin_leaves_other_scripts_untouched(text):
    from ber.normalize.text import fold_latin

    assert fold_latin(text) == text
    assert fold_latin(f"Éclair {text}") == f"Eclair {text}"


def test_models_record_and_follow_their_text(tmp_path, monkeypatch):
    from ber import crossenc, embed, listwise
    from ber.modeltext import default_fold, model_fold

    monkeypatch.delenv("BER_TEXT_FOLD", raising=False)
    assert default_fold()
    monkeypatch.setenv("BER_TEXT_FOLD", "0")
    assert not default_fold()
    assert not model_fold(tmp_path)  # nothing recorded: a model from before folding
    (tmp_path / "crossenc.json").write_text(json.dumps({"text_fold": True}))
    assert model_fold(tmp_path)
    (tmp_path / "crossenc.json").write_text(json.dumps({"kind": "listwise"}))
    (tmp_path / "listwise.json").write_text(json.dumps({"text_fold": True}))
    assert model_fold(tmp_path)

    df = pd.DataFrame({"business_name": ["Çlub Spôrtive"], "business_address": ["12 Rue Lèze"]})
    assert embed.record_texts(df) == ["query: Çlub Spôrtive | 12 Rue Lèze"]
    assert embed.record_texts(df, True) == ["query: Club Sportive | 12 Rue Leze"]
    assert crossenc._text(df, True) == listwise.texts(df, True) == ["Club Sportive | 12 Rue Leze"]
    assert crossenc._text(df) == listwise.texts(df) == ["Çlub Spôrtive | 12 Rue Lèze"]


def test_chunked_loader_folds_like_the_frames(synthetic_dir):
    from ber.embed import load_frames
    from ber.listwise import texts, texts_at

    s1, pool = load_frames(synthetic_dir, "train")
    need1, needp = np.arange(0, len(s1), 7), np.arange(0, len(pool), 11)
    a, b, _, _ = texts_at(synthetic_dir, "train", need1, needp, fold=True)
    assert a == [texts(s1, True)[i] for i in need1] and b == [texts(pool, True)[i] for i in needp]


# ----------------------------------------------------------------------------- French addresses
@pytest.mark.parametrize(
    ("s1", "pool"),
    [
        ("3 Rue de la Paix, Nantes, Pays de la Loire", "3 R PAIX, NANTES, Loire-Atlantique"),
        ("4 Cours de l'Intendance, Bordeaux", "4 CRS INTENDANCE, BORDEAUX"),
        ("71 Rue de l’Alma, Roubaix", "71 RUE ALMA, ROUBAIX"),
        ("8 Passage du Nord, Lille", "8 PASS DU NORD, LILLE"),
        ("Appartement 12, 5 Rue Kléber", "APP 12, 5 RUE KLEBER"),
        ("39 Rue Violette, Lille", "N° 39 Rue Violette, Lille"),
        ("12 Rue Sainte-Catherine, Bordeaux", "12 RUE STE CATHERINE, BORDEAUX"),
    ],
)
def test_french_pool_variants_normalise_like_source1(s1, pool):
    from ber.normalize.address import normalize_address

    a, b = normalize_address(s1, True, "france"), normalize_address(pool, True, "france")
    assert a.clean == b.clean and a.first == b.first


def test_french_rules_stay_out_of_us_and_indian_addresses():
    from ber.normalize.address import normalize_address

    us = normalize_address("3 Rue de la Paix, New Orleans, LA", True, "united states").clean
    assert us.split()[:5] == ["3", "rue", "de", "la", "paix"]  # articles kept outside France
    assert "ste" in normalize_address("12 Main St, Suite 5, NY", True, "united states").clean
    assert "no" not in normalize_address("No 12, MG Road, Pune", True, "india").clean.split()
    # the régions still map with elided names
    for name in ("Côte-d'Or", "Côte-d’Or", "Val-d'Oise"):
        assert normalize_address(name, True, "france").clean in (
            "bourgognefranchecomte",
            "iledefrance",
        )


def test_table_cache_key_follows_the_normaliser(tmp_path, monkeypatch):
    import ber.scale as scale

    f = tmp_path / "x.tsv"
    f.write_text("a")
    cfg = load_config(None, [])
    before = scale._cache_key(f, cfg)
    monkeypatch.setattr(scale, "NORMALIZE_VERSION", scale.NORMALIZE_VERSION + 1)
    assert scale._cache_key(f, cfg) != before


# ----------------------------------------------------------------------------- decoy density
def _toy_pairs():
    # entity 0: true copy (col 0) + decoy (col 1); entity 1: true copy (col 2) + decoys 3, 4
    rows = np.array([0, 0, 1, 1, 1, 1])
    cols = np.array([0, 1, 2, 3, 4, 1])
    p = np.array([0.95, 0.6, 0.9, 0.55, 0.001, 0.0], dtype=np.float32)
    labels = np.array([True, False, True, False, False, False])
    return rows, cols, p, labels


def test_emulate_decoys_clones_only_live_decoys():
    from ber.tuning import emulate_decoys

    rows, cols, p, labels = _toy_pairs()
    r, q, y, n = emulate_decoys(rows, cols, p, labels, 2.0)
    assert n == 2  # cols 1 and 3; col 4 is under min_p, the zero pair of col 1 is not live
    assert r[6:].tolist() == [0, 1] and q[6:].tolist() == pytest.approx([0.6, 0.55])
    assert not y[6:].any() and y[:6].tolist() == labels.tolist()
    assert emulate_decoys(rows, cols, p, labels, 1.0)[3] == 0
    r3, *_ = emulate_decoys(rows, cols, p, labels, 3.0)
    assert len(r3) == 6 + 4
    many = np.arange(2000)
    _, _, _, k = emulate_decoys(many, many, np.full(2000, 0.5), np.zeros(2000, bool), 1.5, seed=3)
    assert 900 < k < 1100  # the fractional part clones about half the decoys


def test_tuning_at_density_is_stricter_and_reports_both():
    from ber.tuning import tune_at_density

    rng = np.random.default_rng(0)
    n = 3000
    rows = np.repeat(np.arange(n), 2)
    cols = np.arange(2 * n)
    true = rng.random(n) < 0.9
    labels = np.zeros(2 * n, bool)
    labels[0::2] = true
    p = np.where(labels, rng.beta(8, 2, 2 * n), rng.beta(2, 5, 2 * n)).astype(np.float32)
    n_true = true.astype(float)
    tune_on, report_on = np.arange(0, n, 2), np.arange(1, n, 2)
    _, plain = tune_at_density(rows, cols, p, labels, n_true, tune_on, report_on)
    _, rep = tune_at_density(rows, cols, p, labels, n_true, tune_on, report_on, boost=2.0)
    info = rep["decoy_boost"]
    assert info["clones"] > 0 and info["train_density_rule"] == plain["chosen"]
    # on the emulated holdout the re-tuned rule beats the train-density one
    assert info["chosen_rule_holdout_emulated"] >= info["train_density_rule_holdout_emulated"]
    assert info["train_density_rule_holdout"] >= info["chosen_rule_holdout_train_density"]


@pytest.mark.slow
def test_scale_run_tunes_at_test_density(synthetic_dir, tmp_path: Path):
    from ber.scale import run

    cfg = load_config(None, [
        f"scale.cache_dir={(tmp_path / 'cache').as_posix()}",
        f"runs_dir={(tmp_path / 'runs').as_posix()}",
        f"artifacts_dir={(tmp_path / 'artifacts').as_posix()}",
        "scale.train_sample=250", "scale.n_jobs=1", "scale.n_threads=2", "model.n_folds=3",
        "model.num_boost_round=100", "model.early_stopping_rounds=20", "decision.decoy_boost=2.0",
    ])  # fmt: skip
    summary = run(cfg, synthetic_dir, tmp_path / "out")
    info = summary["decision"]["decoy_boost"]
    assert info["boost"] == 2.0 and info["clones"] > 0
    assert 0 < info["chosen_rule_holdout_emulated"] <= 1
    assert summary["test"]["validation_ok"], summary["test"]["errors"]


# ----------------------------------------------------------------------------- decoy words
def _names():
    s1 = ["Dupont SARL", "Martin SAS", "Leroy", "Bernard Club", "Petit"] * 60
    pool = (["Dupont Participations", "Martin Holding", "Leroy com", "Bernard Club"] * 60
            + ["Petit com"] * 60 + ["Dupnot Sarl"] * 20)  # fmt: skip
    us_s1 = ["Acme LLC", "Bolt Inc", "Core", "Dune"] * 60
    us_pool = ["Acme Midtown", "Bolt com", "Core", "Dune com"] * 60
    return s1, pool, us_s1, us_pool


def test_country_specific_words_leave_shared_copy_noise_out():
    from ber.vocab import added_words, name_tokens, pool_heavy, specific_words

    s1, pool, us_s1, us_pool = _names()
    heavy = pool_heavy(s1 + us_s1, ["France"] * len(s1) + ["US"] * len(us_s1),
                       pool + us_pool, ["France"] * len(pool) + ["US"] * len(us_pool))  # fmt: skip
    assert {"participations", "holding", "com"} <= heavy["France"]
    assert "dupnot" not in heavy["France"]  # a typo under the count floor
    spec = specific_words(heavy)
    assert spec["France"] == {"participations", "holding"} and spec["US"] == {"midtown"}
    assert added_words("Dupont SARL", "Dupont Participations", spec["France"]) == ["participations"]
    assert added_words("Dupont Participations", "Dupont Participations", spec["France"]) == []
    assert name_tokens("Çlub Spôrtive SARL") == {"club", "sportive"}  # legal form left out


def test_check_decoy_words_counts_merges(tmp_path):
    spec = importlib.util.spec_from_file_location("cdw", ROOT / "scripts" / "check_decoy_words.py")
    cdw = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cdw)
    s1_names, pool_names, _, _ = _names()
    s1 = pd.DataFrame({"entity_id": [f"S1-{i}" for i in range(len(s1_names))],
                       "business_name": s1_names, "country": "France"})  # fmt: skip
    pool = pd.DataFrame({"entity_id": [f"S2-{i}" for i in range(len(pool_names))],
                         "business_name": pool_names, "country": "France"})  # fmt: skip
    # S1-0 (Dupont SARL) merged with S2-0 (Dupont Participations); S1-3 with S2-3 (same name)
    m = tmp_path / "m.tsv"
    m.write_text("source1_entity_id\tmatched_entity_ids\nS1-0\tS2-0\nS1-3\tS2-3\nS1-1\t\n")
    rep = cdw.check(str(m), s1, pool)
    fr = rep["countries"]["France"]
    assert fr["predicted_pairs"] == 2 and fr["adds_specific_word"] == 1
    assert fr["businesses_with_one"] == 1 and "Participations" in fr["examples"][0]


# ----------------------------------------------------------------------------- run_all settings
def _run_all():
    spec = importlib.util.spec_from_file_location("run_all", ROOT / "scripts" / "run_all.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _args(**kw):
    return argparse.Namespace(**{"start": "", "drop": None, "decoy_boost": None, **kw})


def test_run_all_defaults_follow_the_audit_and_resumes_keep_their_settings():
    ra = _run_all()
    fresh, state = _args(), {}
    ra.pin_settings(fresh, state)
    assert (fresh.drop, fresh.decoy_boost) == (0.0, 2.0)
    # a run started before the settings existed keeps the orphan simulation at train density
    old = {"run_A": "runs/x-A"}
    args = _args()
    ra.pin_settings(args, old)
    assert (args.drop, args.decoy_boost) == (0.186, 1.0)
    with pytest.raises(SystemExit):
        ra.pin_settings(
            _args(decoy_boost=2.0), {"run_A": "x", "settings": {"drop": 0.0, "decoy_boost": 1.0}}
        )
    # --from runA redoes the runs, so it may change them
    redo = _args(start="runA", decoy_boost=2.0)
    ra.pin_settings(redo, {"run_A": "x", "settings": {"drop": 0.186, "decoy_boost": 1.0}})
    assert (redo.drop, redo.decoy_boost) == (0.0, 2.0)
