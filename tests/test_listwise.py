"""The listwise group reranker: grouping, the entity split that keeps the holdout honest,
the stage-2 plumbing, and (with torch) a tiny model trained and scored end to end."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ber.config import load_config
from ber.listwise import Group, build_groups, encode_group, training_groups
from ber.modeltext import record_texts
from ber.scale import scorer_columns, split_entities


def _pairs(rng, n_ent=40, max_c=40):
    counts = rng.integers(0, max_c, n_ent)
    rows = np.repeat(np.arange(n_ent), counts)
    return rows, rng.random(len(rows)).astype(np.float32)


def test_every_pair_is_scored_once_next_to_its_entitys_best():
    rng = np.random.default_rng(0)
    rows, p1 = _pairs(rng)
    k = 8
    groups = build_groups(rows, p1, k, np.random.default_rng(1))
    primary = np.concatenate([g.idx[g.primary] for g in groups])
    assert np.array_equal(np.sort(primary), np.arange(len(rows)))  # each pair primary exactly once
    for g in groups:
        assert len(g.idx) <= k and (rows[g.idx] == g.s1).all()
        top = np.sort(np.flatnonzero(rows == g.s1)[np.argsort(-p1[rows == g.s1])][: k // 2])
        if not g.primary.all():  # overflow group: the anchors are the entity's likeliest pairs
            assert np.array_equal(np.sort(g.idx[~g.primary]), top)


def test_training_groups_draw_a_floor_and_stay_in_the_share():
    rng = np.random.default_rng(3)
    rows, p1 = _pairs(rng)
    ents = np.zeros(40, dtype=bool)
    ents[::2] = True
    groups = training_groups(rows, p1, ents, np.random.default_rng(4), k=8)
    assert groups and all(ents[g.s1] for g in groups)
    assert all(p1[g.idx].min() >= 0.002 for g in groups)
    assert all(len(g.idx) <= 8 for g in groups)


def test_encoded_group_fits_and_marks_every_candidate():
    g = Group(0, np.array([5, 2, 9]), np.array([True, True, False]))
    q = np.arange(100, 160, dtype=np.int32)
    cands = [np.arange(200, 260, dtype=np.int32)] * 3
    e = encode_group(g, q, cands, cls=1, sep=2, max_len=64, len_s1=10, len_cand=40)
    assert len(e.ids) <= 64 and e.ids[0] == 1
    assert (e.seg == 1).sum() == 10
    assert all((e.seg == j + 2).sum() > 0 for j in range(3))
    assert np.array_equal(e.idx, g.idx) and np.array_equal(e.primary, g.primary)


def test_listwise_columns_add_pool_side_context_and_rest_mass():
    rows = np.array([0, 0, 1, 1])
    cols = np.array([7, 8, 7, 9])
    ce = np.array([0.9, 0.2, 0.6, np.nan], np.float32)
    f = scorer_columns(rows, cols, {"lw": ce, "ce": ce}, {"lw"})
    assert "ce_rank_pool" not in f and "ce_rest_s1" not in f  # main's columns for "ce"
    # pool record 7: suitor 0 (0.9) beats suitor 1 (0.6)
    assert f["lw_rank_pool"][[0, 2]].tolist() == [0, 1]
    assert f["lw_gap_pool"][2] == pytest.approx(-0.3)
    assert f["lw_margin_pool"][0] == pytest.approx(0.3)
    assert f["lw_rest_s1"][0] == pytest.approx(0.2) and f["lw_rest_s1"][2] == pytest.approx(0.0)
    assert np.isnan(f["lw_rank_pool"][3]) and np.isnan(f["lw_rest_s1"][3])


def test_split_entities_holds_out_every_eligible_entity_outside_the_sample():
    ids = np.array([f"S1-{i:05d}" for i in range(5000)])
    eligible = np.ones(5000, dtype=bool)
    eligible[::7] = False
    cfg = load_config(None, ["scale.train_sample=1000"])
    sample, in_s, hold = split_entities(ids, eligible, cfg)
    assert len(sample) == 1000 and eligible[sample].all() and in_s[sample].all()
    assert np.array_equal(hold, np.flatnonzero(eligible & ~in_s))
    assert np.array_equal(split_entities(ids, eligible, cfg)[0], sample)  # deterministic


def texts(df, fold: bool = False) -> list[str]:
    return record_texts(df["business_name"].tolist(), df["business_address"].tolist(), fold)


# ----------------------------------------------------------------------------- torch
def _tiny_encoder(path: Path, texts: list[str]) -> Path:
    """A 2-layer BERT with a word-level vocabulary of the synthetic texts (no download)."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import BertConfig, BertModel, PreTrainedTokenizerFast

    tk = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tk.train_from_iterator(texts, trainers.WordLevelTrainer(
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]"], vocab_size=4000))  # fmt: skip
    fast = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="[UNK]", pad_token="[PAD]",
                                   cls_token="[CLS]", sep_token="[SEP]")  # fmt: skip
    fast.save_pretrained(path)
    cfg = BertConfig(vocab_size=tk.get_vocab_size(), hidden_size=32, num_hidden_layers=2,
                     num_attention_heads=2, intermediate_size=64, max_position_embeddings=256)  # fmt: skip
    BertModel(cfg).save_pretrained(path)
    return path


def _toy_groups(data_dir: Path, path: Path) -> Path:
    """True matches plus random same-country decoys per train entity, a noisy p1, labels."""
    from ber.embed import countries, load_frames, truth_pairs

    s1, pool = load_frames(data_dir, "train")
    a, b = truth_pairs(data_dir, s1, pool)
    rng = np.random.default_rng(0)
    c1, cp = countries(s1), countries(pool)
    neg_r = np.repeat(np.arange(len(s1)), 3)
    neg_c = np.array([rng.choice(np.flatnonzero(cp == c1[r])) for r in neg_r])
    owner = np.full(len(pool), -1)
    owner[b] = a
    rows, cols = np.concatenate([a, neg_r]), np.concatenate([b, neg_c])
    labels = owner[cols] == rows
    p1 = np.clip(np.where(labels, 0.7, 0.1) + rng.normal(0, 0.2, len(rows)), 0.003, 0.999)
    n_true = np.bincount(a, minlength=len(s1)).astype(float)
    train = np.zeros(len(s1), dtype=bool)
    train[: len(s1) // 2] = True
    np.savez(path, rows=rows, cols=cols, p1=p1.astype(np.float32), labels=labels, n_true=n_true,
             train=train, valid=~train)  # fmt: skip
    return path


def test_tiny_listwise_model_trains_and_scores_every_pair(synthetic_dir, tmp_path: Path):
    from ber import listwise
    from ber.embed import load_frames
    from ber.scale import scorer_kind

    s1, pool = load_frames(synthetic_dir, "train")
    init = _tiny_encoder(tmp_path / "tiny", texts(s1) + texts(pool))
    groups = _toy_groups(synthetic_dir, tmp_path / "groups.npz")
    out = listwise.train(synthetic_dir, init, groups, tmp_path / "lw", k=8, max_len=128, batch=16,
                         lr=1e-3, epochs=2, n_valid=60, eval_every=20, workers=0)  # fmt: skip
    assert scorer_kind(out) == "listwise"
    with np.load(groups) as z:
        rows, cols, p1 = z["rows"], z["cols"], z["p1"]
    pp, gp = listwise.score(out, synthetic_dir, "train", rows, cols, p1, batch=32, workers=0)
    assert pp.shape == rows.shape == gp.shape and not np.isnan(pp).any()
    assert ((pp >= 0) & (pp <= 1)).all()
    ones = pd.Series(gp.astype(np.float32)).groupby(rows).nunique()
    assert (ones == 1).all()  # one group probability per entity
    one = rows == rows[0]  # a single entity: one group
    pp1, gp1 = listwise.score(out, synthetic_dir, "train", rows[one], cols[one], p1[one], workers=0)
    assert not np.isnan(pp1).any() and not np.isnan(gp1).any()
    empty = listwise.score(out, synthetic_dir, "train", rows[:0], cols[:0], p1[:0], workers=0)
    assert len(empty[0]) == len(empty[1]) == 0


# ----------------------------------------------------------------------------- pipeline
@pytest.mark.slow
def test_scale_run_with_listwise_scorer(synthetic_dir, tmp_path: Path):
    from ber.scale import run

    fake = (Path(__file__).parent / "fake_crossenc.py").as_posix()
    lw_model = tmp_path / "lw-fake"
    lw_model.mkdir()
    (lw_model / "crossenc.json").write_text('{"kind": "listwise"}')
    base = [
        f"scale.cache_dir={(tmp_path / 'cache').as_posix()}",
        f"artifacts_dir={(tmp_path / 'artifacts').as_posix()}",
        f"scale.ce_model=unused_pairwise,{lw_model.as_posix()}",
        f'scale.ce_python=\'"{Path(sys.executable).as_posix()}" "{fake}"\'',
        "scale.ce_min_p1=0.001", "scale.ce_pool_context=true", "scale.fingerprint=true",
        "scale.train_sample=250", "scale.n_jobs=1", "scale.n_threads=2",
        "model.n_folds=3", "model.num_boost_round=200", "model.early_stopping_rounds=20",
    ]  # fmt: skip
    summary = run(load_config(None, [*base, f"runs_dir={(tmp_path / 'r1').as_posix()}"]),
                  synthetic_dir, tmp_path / "out")  # fmt: skip
    assert summary["test"]["validation_ok"], summary["test"]["errors"]
    run_dir = next((tmp_path / "r1").iterdir())
    scores = np.load(run_dir / "train_scores.npz")
    assert not (scores["in_sample"] & ~scores["eligible"]).any()
    names = pd.read_csv(run_dir / "feature_importance_stage2.tsv", sep="\t")["feature"].tolist()
    assert {"ce", "ce_rank_pool", "lw", "lw_rest_s1", "lw_grp", "lw_rank_pool"} <= set(names)
    assert {"fp_shared", "fp_missing", "fp_n_sib", "fp_inv"} <= set(names)
    assert (run_dir / "train_lw_scores.npy").exists()
    assert (run_dir / "train_lw_scores_grp.npy").exists()

    # the scores of the same models on the same pairs come from the cache, not the scorers
    cached = sorted((tmp_path / "cache" / "scores").glob("*.npy"))
    # train: pairwise, listwise, listwise group
    assert len([p for p in cached if p.name.startswith("train_")]) == 3
    stamps = {p: p.stat().st_mtime_ns for p in cached}
    again = run(load_config(None, [*base, f"runs_dir={(tmp_path / 'r2').as_posix()}"]),
                synthetic_dir, tmp_path / "out2")  # fmt: skip
    assert again["test"]["validation_ok"]
    assert {p: p.stat().st_mtime_ns for p in cached} == stamps  # reused, not rewritten


def test_listwise_scorer_reads_whole_groups_even_with_a_band(synthetic_dir, tmp_path: Path):
    """main's ``ce_band`` limits later pairwise models to the first one's unsure pairs; a
    listwise model reads whole groups, so it scores every gated pair and returns its group
    output."""
    from types import SimpleNamespace

    from ber.embed import load_frames
    from ber.scale import cross_scores

    fake = (Path(__file__).parent / "fake_crossenc.py").as_posix()
    lw_model = tmp_path / "lw-fake"
    lw_model.mkdir()
    (lw_model / "crossenc.json").write_text('{"kind": "listwise"}')
    cfg = load_config(None, [
        f"scale.ce_model=unused_a,{lw_model.as_posix()},unused_b",
        f'scale.ce_python=\'"{Path(sys.executable).as_posix()}" "{fake}"\'',
        "scale.ce_min_p1=0.5", "scale.ce_band=0.3,0.8",
        f"scale.cache_dir={(tmp_path / 'cache').as_posix()}",
    ])  # fmt: skip
    s1, pool = load_frames(synthetic_dir, "train")
    rng = np.random.default_rng(0)
    rows = rng.integers(0, len(s1), 400)
    cols = rng.integers(0, len(pool), 400)
    p1 = rng.random(400)
    sd = SimpleNamespace(name="train", rows=rows, cols=cols)
    out = cross_scores(sd, cfg, p1, synthetic_dir, tmp_path)
    assert set(out) == {"ce", "lw", "lw_grp", "ce2"}
    gated = p1 >= 0.5
    assert not np.isnan(out["lw"][gated]).any() and np.isnan(out["lw"][~gated]).all()
    assert not np.isnan(out["lw_grp"][gated]).any()
    unsure = gated & (out["ce"] > 0.3) & (out["ce"] < 0.8)
    with np.load(tmp_path / "train_ce2_pairs.npz") as z:
        assert len(z["rows"]) == unsure.sum()  # the band still applies to pairwise models


def test_chunked_text_loader_matches_the_full_frames(synthetic_dir):
    from ber.embed import load_frames
    from ber.listwise import texts_at

    s1, pool = load_frames(synthetic_dir, "train")
    rng = np.random.default_rng(2)
    need1 = np.sort(rng.choice(len(s1), 50, replace=False))
    needp = np.sort(rng.choice(len(pool), 80, replace=False))
    a, b, n1, npool = texts_at(synthetic_dir, "train", need1, needp)
    assert (n1, npool) == (len(s1), len(pool))
    assert a == [texts(s1)[i] for i in need1] and b == [texts(pool)[i] for i in needp]


def test_groups_from_raw_data_hold_every_true_copy_and_same_country_negatives(
    synthetic_dir, tmp_path
):
    from ber.embed import countries, encoder_mask, load_frames, truth_pairs

    out = tmp_path / "g.npz"
    subprocess.run(
        [sys.executable, str(Path(__file__).parents[1] / "scripts" / "make_lw_groups.py"),
         "--data-dir", str(synthetic_dir), "--out", str(out), "--train", "300", "--valid", "60"],
        capture_output=True, text=True, check=True,
    )  # fmt: skip
    z = np.load(out)
    s1, pool = load_frames(synthetic_dir, "train")
    a, b = truth_pairs(synthetic_dir, s1, pool)
    enc = encoder_mask(s1["entity_id"].to_numpy())
    train, valid = z["train"], z["valid"]
    assert (
        train.sum() == min(300, enc.sum()) and not (train & ~enc).any() and not (valid & enc).any()
    )
    rows, cols, labels = z["rows"], z["cols"], z["labels"]
    picked = train | valid
    truth = set(zip(a[picked[a]].tolist(), b[picked[a]].tolist()))
    assert set(zip(rows[labels].tolist(), cols[labels].tolist())) == truth  # every true copy
    owner = dict(zip(b.tolist(), a.tolist()))
    assert all(owner.get(c, -1) != r for r, c in zip(rows[~labels], cols[~labels]))
    c1, cp = countries(s1), countries(pool)
    assert (c1[rows] == cp[cols]).all()
    assert (~labels).sum() > 0 and (z["p1"] >= 0.05).all()
    assert np.array_equal(z["n_true"], np.bincount(a, minlength=len(s1)))


# ----------------------------------------------------------------------------- C7: pool side
def _pool_groups(data_dir, out, n_train=400):
    subprocess.run(
        [sys.executable, str(Path(__file__).parents[1] / "scripts" / "make_lw_groups.py"),
         "--data-dir", str(data_dir), "--out", str(out), "--side", "pool",
         "--train", str(n_train), "--valid", "80", "--unowned", "0.3"],
        capture_output=True, text=True, check=True,
    )  # fmt: skip
    return out


def test_pool_side_groups_hold_the_owner_and_only_encoder_share_suitors(synthetic_dir, tmp_path):
    from ber.embed import encoder_mask, load_frames, truth_pairs

    z = np.load(_pool_groups(synthetic_dir, tmp_path / "p.npz"))
    s1, pool = load_frames(synthetic_dir, "train")
    a, b = truth_pairs(synthetic_dir, s1, pool)
    enc = encoder_mask(s1["entity_id"].to_numpy())
    owner = np.full(len(pool), -1)
    owner[b] = a
    rows, cols, labels = z["rows"], z["cols"], z["labels"]
    assert str(z["side"]) == "pool" and len(z["train"]) == len(pool)
    assert enc[rows].all()  # no holdout entity's label is learned
    keys = np.flatnonzero(z["train"] | z["valid"])
    owned = keys[owner[keys] >= 0]
    assert set(zip(rows[labels].tolist(), cols[labels].tolist())) == set(
        zip(owner[owned].tolist(), owned.tolist())
    )
    assert (owner[z["train"]] < 0).mean() > 0.2  # records nobody owns are in
    assert (owner[cols[~labels]] != rows[~labels]).all()


def test_tiny_pool_side_model_trains_with_orphans_and_scores_per_record(synthetic_dir, tmp_path):
    from ber import listwise
    from ber.embed import load_frames
    from ber.scale import scorer_kind

    groups = _pool_groups(synthetic_dir, tmp_path / "p.npz")
    s1, pool = load_frames(synthetic_dir, "train")
    init = _tiny_encoder(tmp_path / "tiny", texts(s1) + texts(pool))
    out = listwise.train(synthetic_dir, init, groups, tmp_path / "lwp", k=8, max_len=128, batch=16,
                         lr=1e-3, epochs=2, n_valid=40, eval_every=20, workers=0)  # fmt: skip
    meta = json.loads((out / "listwise.json").read_text())
    assert meta["side"] == "pool" and meta["drop_owner"] == 0.2
    assert scorer_kind(out) == "listwise_pool"
    with np.load(groups) as z:
        rows, cols, p1 = z["rows"], z["cols"], z["p1"]
    pp, gp = listwise.score(out, synthetic_dir, "train", rows, cols, p1, batch=32, workers=0)
    assert not np.isnan(pp).any() and not np.isnan(gp).any()
    per_record = pd.Series(gp.astype(np.float32)).groupby(cols).nunique()
    assert (per_record == 1).all()  # one "someone owns it" probability per pool record


def test_orphan_drop_removes_the_owner_from_a_share_of_groups():
    rng = np.random.default_rng(0)
    keys = np.repeat(np.arange(300), 3)
    labels = np.tile([1.0, 0.0, 0.0], 300).astype(np.float32)
    p1 = rng.random(len(keys)).astype(np.float32) * 0.5 + 0.1
    ents = np.ones(300, dtype=bool)
    groups = training_groups(keys, p1, ents, np.random.default_rng(1), 8, labels=labels,
                             drop_owner=0.25)  # fmt: skip
    without = np.mean([labels[g.idx].sum() == 0 for g in groups])
    assert 0.15 < without < 0.35 and all(len(g.idx) for g in groups)


def test_pool_side_scorer_gets_its_own_prefix(synthetic_dir, tmp_path):
    from types import SimpleNamespace

    from ber.embed import load_frames
    from ber.scale import cross_scores

    fake = (Path(__file__).parent / "fake_crossenc.py").as_posix()
    lwp = tmp_path / "lwp-fake"
    lwp.mkdir()
    (lwp / "crossenc.json").write_text('{"kind": "listwise", "side": "pool"}')
    cfg = load_config(None, [
        f"scale.ce_model=unused_a,{lwp.as_posix()}",
        f'scale.ce_python=\'"{Path(sys.executable).as_posix()}" "{fake}"\'',
        "scale.ce_min_p1=0.5", f"scale.cache_dir={(tmp_path / 'cache').as_posix()}",
    ])  # fmt: skip
    s1, pool = load_frames(synthetic_dir, "train")
    rng = np.random.default_rng(0)
    sd = SimpleNamespace(name="train", rows=rng.integers(0, len(s1), 200),
                         cols=rng.integers(0, len(pool), 200))  # fmt: skip
    out = cross_scores(sd, cfg, rng.random(200), synthetic_dir, tmp_path)
    assert set(out) == {"ce", "lwp", "lwp_grp"}


def test_fingerprint_tells_the_sources_variant_from_a_neighbour():
    """S1 says 6515; the business's S3 copies carry their source's variant 6513; a
    neighbour decoy says 6541 and misses the variant."""
    from ber.scale import fingerprint_columns

    s1_addr = np.array(["6515 pilotbird ln houston tx"])
    pool_addr = np.array(["6513 pilotbird ln houston tx", "6513 pilotbird ln houston tx",
                          "6541 pilotbird ln houston tx", "6515 pilotbird ln houston tx"])  # fmt: skip
    pool_source = np.array([3, 3, 3, 2])
    rows = np.zeros(4, dtype=np.int64)
    cols = np.arange(4)
    p1 = np.array([0.95, 0.6, 0.5, 0.97], np.float32)
    f = fingerprint_columns(rows, cols, p1, s1_addr, pool_addr, pool_source)
    assert f["fp_shared"][1] == 1 and f["fp_missing"][1] == 0  # the typo'd sibling
    assert f["fp_shared"][2] == 0 and f["fp_missing"][2] == 1  # the neighbour
    assert f["fp_n_sib"].tolist() == [0, 1, 1, 0]  # S2 has no confident S2 sibling
    assert np.isnan(f["fp_shared"][[0, 3]]).all() and f["fp_inv"][3] == 0
    # source balance: each has one confident candidate in the other source; rank by p1
    # within its own source (the S3 candidates 0.95, 0.6, 0.5; the lone S2 one)
    assert f["fp_other"].tolist() == [1, 1, 1, 1]
    assert f["fp_rank_src"].tolist() == [0, 1, 2, 0]
