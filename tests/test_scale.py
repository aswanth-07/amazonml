"""The large-scale path: noise seen in the real data, retrieval, context features, end to end."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from ber.blocking.knn import partitioned_topk, sparse_topk
from ber.config import load_config
from ber.normalize.address import normalize_address
from ber.normalize.name import normalize_name
from ber.normalize.text import consonant_key, deleet
from ber.pairfeat import group_stats


@pytest.mark.parametrize(
    ("romanised", "english"),
    [
        ("painiar", "pioneer"),
        ("sophtveyr", "software"),
        ("imdstrij", "industries"),
        ("inphotek", "infotech"),
        ("daynamik", "dynamic"),
        ("impeks", "impex"),
        ("marketimg", "marketing"),
        ("praivet", "private"),
        ("limtid", "limited"),
    ],
)
def test_consonant_key_joins_romanised_and_english(romanised, english):
    assert consonant_key(romanised) == consonant_key(english)


def test_deleet_only_touches_words():
    assert deleet("g1obal") == "global"
    assert deleet("6lobal") == "global"
    assert deleet("4536") == "4536"
    assert deleet("4b") == "4b"


@pytest.mark.parametrize(
    ("raw", "core", "alias"),
    [
        (
            "Deltatavo formerly known as Niex Holding Company Center",
            "deltatavo",
            "niex holding ctr",
        ),
        ("Ectosol fka Perinton Interfaith Center", "ectosol", "perinton interfaith ctr"),
        ("Xylodelta formerly: Sapphire LLC", "xylodelta", "sapphire"),
        ("Calohaloio t/a Gurgaon Sales Pvt Ltd", "calohaloio", "gurgaon sales"),
    ],
)
def test_former_and_trade_names_split(raw, core, alias):
    n = normalize_name(raw)
    assert (n.core, n.alias) == (core, alias)
    assert n.marker


@pytest.mark.parametrize(
    ("raw", "core"),
    [
        ("JWALA INFRAVERTURES LIMITED #30473", "jwala infravertures"),
        ("regional prime horizon center (ID: 96415)", "regional prime horizon ctr"),
        ("Smt Pushp Management (Company)", "pushp management"),
        ("-- Holloway Peak Inc Seafood", "holloway peak seafood"),
        ("LLC Sapphire", "sapphire"),
        ("XOE Private Buildtech Limited", "xoe buildtech"),
        ("G1obal Dte L.L.C. L.L.C.", "global dte"),
        ("Keystone Dividend [Company]", "keystone dividend"),
        ("M/s Surat Travels", "surat travels"),
    ],
)
def test_name_noise_is_removed(raw, core):
    assert normalize_name(raw).core == core


def test_domain_and_hashtag_names():
    for raw, squash in (
        ("lynx.com", "lynx"),
        ("Pushpmanagement.Com", "pushpmanagement"),
        ("#báskervillemeadows", "baskervillemeadows"),
    ):
        n = normalize_name(raw)
        assert n.styled and n.squash == squash


def test_indic_script_name_keys_match_english():
    native = normalize_name("ਪਾਇਨੀਅਰ ਇੰਡਸਟ੍ਰੀਜ਼ ਪ੍ਰਾਈਵੇਟ ਲਿਮਟਿਡ")
    english = normalize_name("Pioneer Industries Private Limited")
    assert native.script and not english.script
    assert native.key == english.key
    assert native.legal == english.legal == "ltd pvt"


def test_indian_address_states_and_numbers():
    a = normalize_address("No.199, Valluvarkottam High Rd, Chennai, தமிழ்நாடு", country="india")
    b = normalize_address("CHENNAI, Tamil Nadu, 199, VALLUVARKOTTAM HIGH ROAD", country="india")
    c = normalize_address("Valluvarkottam High Rd, Chennai, TN, H.no 0199, null", country="india")
    assert set(a.core.split()) == set(b.core.split()) == set(c.core.split())
    assert a.first == b.first == c.first == "199"


def test_us_address_reordering_and_zero_padding():
    a = normalize_address("57 Windsor Drive, Belleville, Illinois", country="united states")
    b = normalize_address("BELLEVILLE, IL, 057 WINDSOR DR", country="united states")
    assert set(a.core.split()) == set(b.core.split())
    assert a.first == b.first == "57"


def test_state_maps_do_not_leak_across_countries():
    # "GA" is Georgia in the US and Goa in India; neither map applies to France
    assert (
        "ga"
        in normalize_address("1 Main St, Atlanta, Georgia", country="united states").core.split()
    )
    assert "ga" in normalize_address("Panaji, Goa", country="india").core.split()
    assert "georgia" in normalize_address("Rue de Georgia, Lyon", country="france").core.split()


def test_sparse_topk_matches_brute_force():
    rng = np.random.default_rng(1)
    q = sp.random(60, 40, density=0.2, format="csr", random_state=1, dtype=np.float32)
    d = sp.random(300, 40, density=0.2, format="csr", random_state=2, dtype=np.float32)
    rows, cols, scores = sparse_topk(q, d, k=7, n_threads=4, chunk_rows=13)
    dense = (q @ d.T).toarray()
    for i in range(60):
        got = cols[rows == i]
        positive = np.flatnonzero(dense[i] > 0)
        expect = positive[np.argsort(-dense[i][positive], kind="stable")][:7]
        assert np.allclose(np.sort(dense[i][got]), np.sort(dense[i][expect]))
    assert np.allclose(scores, dense[rows, cols])
    del rng


def test_partitioned_topk_keeps_countries_apart():
    q = sp.csr_matrix(np.ones((2, 3), np.float32))
    d = sp.csr_matrix(np.ones((3, 3), np.float32))
    rows, cols, _ = partitioned_topk(
        q, d, np.array(["us", "fr"]), np.array(["us", "fr", ""]), k=5, n_threads=2
    )
    pairs = set(zip(rows.tolist(), cols.tolist()))
    assert pairs == {(0, 0), (0, 2), (1, 1), (1, 2)}


def test_prefix_filter_keeps_rarest_tokens_within_budget():
    from ber.vectorize import prefix_filter

    m = sp.csr_matrix(np.array([[1, 1, 1, 1], [0, 1, 0, 1], [0, 0, 0, 1]], dtype=np.float32))
    df = np.array([5, 50, 500, 9000])
    out = prefix_filter(m, df, budget=100, cap=5000).toarray()
    assert out[0].tolist() == [1, 1, 0, 0]  # 5 + 50 <= 100; 500 would exceed the budget
    assert out[1].tolist() == [0, 1, 0, 0]  # the 9000 token is over the cap
    assert out[2].tolist() == [0, 0, 0, 0]  # only an over-cap token: nothing to search on
    tight = prefix_filter(m, df, budget=1, cap=5000).toarray()
    assert tight[0].tolist() == [1, 0, 0, 0]  # the rarest token is always kept


def test_country_index_matches_brute_force():
    from ber.blocking.multiview import CountryIndex

    rng = np.random.default_rng(3)
    qn = sp.random(40, 30, density=0.3, format="csr", random_state=3, dtype=np.float32)
    qa = sp.random(40, 30, density=0.3, format="csr", random_state=4, dtype=np.float32)
    dn = sp.random(200, 30, density=0.3, format="csr", random_state=5, dtype=np.float32)
    da = sp.random(200, 30, density=0.3, format="csr", random_state=6, dtype=np.float32)
    q_country = rng.choice(["us", "in"], 40)
    d_country = rng.choice(["us", "in", ""], 200)
    index = CountryIndex(dn, da, d_country, partition=True)
    rows, cols = index.search(
        qn,
        qa,
        q_country,
        m_comb=4,
        m_name=3,
        m_addr=2,
        wn2=0.36,
        wa2=0.64,
        n_threads=2,
        chunk_rows=7,
    )
    got = set(zip(rows.tolist(), cols.tolist()))
    sn, sa = (qn @ dn.T).toarray(), (qa @ da.T).toarray()
    for i in range(40):
        allowed = (d_country == q_country[i]) | (d_country == "")
        expect = set()
        for s, m in ((sn[i], 3), (sa[i], 2), (0.36 * sn[i] + 0.64 * sa[i], 4)):
            cand = np.flatnonzero(allowed & (s > 0))
            expect |= set(cand[np.argsort(-s[cand], kind="stable")][:m].tolist())
        mine = {c for r, c in got if r == i}
        assert mine == expect  # random float scores: no ties, so the lists are unique


def test_group_stats():
    keys = np.array([0, 0, 0, 1, 2, 2])
    vals = np.array([0.2, 0.9, 0.5, 0.3, 0.1, 0.4], np.float32)
    rank, best, second, size = group_stats(keys, vals)
    assert rank.tolist() == [2, 0, 1, 0, 1, 0]
    assert best.tolist() == pytest.approx([0.9, 0.9, 0.9, 0.3, 0.4, 0.4])
    assert second.tolist() == pytest.approx([0.5, 0.5, 0.5, -1.0, 0.1, 0.1])
    assert size.tolist() == [3, 3, 3, 1, 2, 2]


@pytest.mark.slow
def test_scale_run_on_synthetic(synthetic_dir, tmp_path: Path):
    from ber.scale import run

    cfg = load_config(
        None,
        [
            f"scale.cache_dir={(tmp_path / 'cache').as_posix()}",
            f"runs_dir={(tmp_path / 'runs').as_posix()}",
            f"artifacts_dir={(tmp_path / 'artifacts').as_posix()}",
            "scale.train_sample=250",
            "scale.n_jobs=1",
            "scale.n_threads=2",
            "model.n_folds=3",
            "model.num_boost_round=200",
            "model.early_stopping_rounds=20",
        ],
    )
    summary = run(cfg, synthetic_dir, tmp_path / "out")
    assert summary["test"]["validation_ok"], summary["test"]["errors"]
    assert summary["train_blocking"]["pair_recall"] > 0.95
    assert summary["test"]["scored"]["fbeta"] > 0.8
    assert (tmp_path / "artifacts" / "scale_model.joblib").exists()

    # a rerun reads the blocking and feature caches and reproduces the outputs exactly
    cache = tmp_path / "cache"
    for split in ("train", "test"):
        assert len(list(cache.glob(f"{split}_blocking_*.joblib"))) == 1
        assert len(list(cache.glob(f"{split}_features_*.json"))) == 1
    again = run(cfg, synthetic_dir, tmp_path / "out2")
    assert again["train_blocking"] == summary["train_blocking"]
    for name in ("matching_results.tsv", "candidate_pairs.tsv"):
        a = (tmp_path / "out" / name).read_bytes()
        assert a == (tmp_path / "out2" / name).read_bytes()


def _fake_embeddings(data_dir: Path, emb_dir: Path, k: int = 10, dim: int = 64) -> None:
    """Embeddings and kNN files in ber.embed's format, from hashed name trigrams (no torch)."""
    import zlib

    from ber.embed import countries, emb_paths, load_frames

    def vec(texts):
        out = np.zeros((len(texts), dim), np.float32)
        for i, t in enumerate(texts):
            t = f"  {t.lower()}  "
            for j in range(len(t) - 2):
                out[i, zlib.crc32(t[j : j + 3].encode()) % dim] += 1
        out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)
        return out.astype(np.float16)

    emb_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        p = emb_paths(emb_dir, split)
        s1, pool = load_frames(data_dir, split)
        q, d = vec(s1["business_name"].tolist()), vec(pool["business_name"].tolist())
        np.save(p["s1"], q)
        np.save(p["pool"], d)
        np.save(p["s1_ids"], s1["entity_id"].to_numpy().astype("U"))
        np.save(p["pool_ids"], pool["entity_id"].to_numpy().astype("U"))
        qc, dc = countries(s1), countries(pool)
        sims = q.astype(np.float32) @ d.astype(np.float32).T
        sims[qc[:, None] != dc[None, :]] = -2
        fwd = np.argsort(-sims, axis=1, kind="stable")[:, :k]
        rev = np.argmax(sims, axis=0)
        np.savez(
            p["knn"],
            fwd_rows=np.repeat(np.arange(len(q)), k).astype(np.int32),
            fwd_cols=fwd.ravel().astype(np.int32),
            fwd_sim=np.take_along_axis(sims, fwd, 1).ravel().astype(np.float16),
            rev_rows=rev.astype(np.int32),
            rev_cols=np.arange(len(d)).astype(np.int32),
            rev_sim=sims[rev, np.arange(len(d))].astype(np.float16),
        )


@pytest.mark.slow
def test_scale_run_with_embedding_view(synthetic_dir, tmp_path: Path):
    from ber.embed import encoder_mask
    from ber.scale import run

    _fake_embeddings(synthetic_dir, tmp_path / "emb")

    def cfg_for(cache: str, emb: bool):
        return load_config(
            None,
            [
                f"scale.cache_dir={(tmp_path / cache).as_posix()}",
                f"scale.emb_dir={(tmp_path / 'emb').as_posix() if emb else ''}",
                f"runs_dir={(tmp_path / 'runs' / cache).as_posix()}",
                f"artifacts_dir={(tmp_path / 'artifacts').as_posix()}",
                "scale.train_sample=150",
                "scale.n_jobs=1",
                "scale.n_threads=2",
                "model.n_folds=3",
                "model.num_boost_round=200",
                "model.early_stopping_rounds=20",
            ],
        )

    summary = run(cfg_for("fresh", emb=True), synthetic_dir, tmp_path / "out")
    # the same run on top of a token-only run's caches copies shared pairs' features
    run(cfg_for("warm", emb=False), synthetic_dir, tmp_path / "out_plain", predict_test=False)
    run(cfg_for("warm", emb=True), synthetic_dir, tmp_path / "out_warm")
    for name in ("matching_results.tsv", "candidate_pairs.tsv"):
        a = (tmp_path / "out" / name).read_bytes()
        assert a == (tmp_path / "out_warm" / name).read_bytes()
    assert summary["test"]["validation_ok"], summary["test"]["errors"]
    assert summary["test"]["scored"]["fbeta"] > 0.8
    tb = summary["train_blocking"]
    assert "cos_emb_recall@1" in tb
    s1_ids = pd.read_csv(synthetic_dir / "train" / "train_source1.tsv", sep="\t")["entity_id"]
    assert tb["eligible_entities"] == int((~encoder_mask(s1_ids.to_numpy())).sum())
    run_dir = next((tmp_path / "runs" / "fresh").iterdir())
    names = pd.read_csv(run_dir / "feature_importance.tsv", sep="\t")["feature"].tolist()
    assert {"cos_emb", "cos_emb_rank_s1", "cos_emb_margin_pool"} <= set(names)
    names2 = pd.read_csv(run_dir / "feature_importance_stage2.tsv", sep="\t")["feature"].tolist()
    assert {"coh_emb", "partner_p1", "cos_emb"} <= set(names2)
    scores = np.load(run_dir / "train_scores.npz")
    assert len(scores["p"]) == len(scores["rows"]) == len(scores["labels"])
    assert (run_dir / "test_scores.npz").exists()


@pytest.mark.slow
def test_scale_run_with_cross_encoder(synthetic_dir, tmp_path: Path):
    import sys

    from ber.scale import run

    fake = (Path(__file__).parent / "fake_crossenc.py").as_posix()
    cfg = load_config(
        None,
        [
            f"scale.cache_dir={(tmp_path / 'cache').as_posix()}",
            f"runs_dir={(tmp_path / 'runs').as_posix()}",
            f"artifacts_dir={(tmp_path / 'artifacts').as_posix()}",
            "scale.ce_model=unused",
            f'scale.ce_python=\'"{Path(sys.executable).as_posix()}" "{fake}"\'',
            "scale.ce_min_p1=0.001",
            "scale.train_sample=250",
            "scale.n_jobs=1",
            "scale.n_threads=2",
            "model.n_folds=3",
            "model.num_boost_round=200",
            "model.early_stopping_rounds=20",
        ],
    )
    summary = run(cfg, synthetic_dir, tmp_path / "out")
    assert summary["test"]["validation_ok"], summary["test"]["errors"]
    assert summary["cross_encoder_pairs"] > 0
    run_dir = next((tmp_path / "runs").iterdir())
    names = pd.read_csv(run_dir / "feature_importance_stage2.tsv", sep="\t")["feature"].tolist()
    assert {"ce", "ce_rank_s1", "ce_gap_s1", "ce_margin_s1"} <= set(names)


def test_detail_features_tell_neighbours_from_typos():
    from ber.pairfeat import detail_features
    from ber.table import build_table

    def table(rows, source):
        df = pd.DataFrame(
            rows, columns=["entity_id", "business_name", "business_address", "country"]
        )
        return build_table(df, source, True, n_jobs=1)

    s1 = table([("S1-1", "Howard", "450 Mass Ave, Washington, DC", "US"),
                ("S1-2", "WPQ Realtors", "12 Main St, Pune", "India")], 1)  # fmt: skip
    pool = table([("S2-1", "Howard", "471 Mass Ave, Washington, DC", "US"),
                  ("S2-2", "Howard", "45 Mass Ave, Washington, DC", "US"),
                  ("S2-3", "RPQ Realtors", "12 Main St, Pune", "India")], 2)  # fmt: skip
    f = detail_features(s1, pool, np.array([0, 0, 1]), np.array([0, 1, 2]))
    assert f["first_num_logdiff"][0] == pytest.approx(np.log1p(21))
    assert f["first_num_prefix"][1] == 1.0  # 45 is a truncation of 450
    assert np.isnan(f["first_num_prefix"][[0, 2]]).all()  # same length: not applicable
    assert f["first_num_edit"][:3].tolist() == [2.0, 1.0, 0.0]
    assert f["name_short_tok_diff"].tolist() == [0.0, 0.0, 1.0]
    assert f["name_extra_tok_s1"][2] == 1 and f["name_extra_tok_pool"][2] == 1


def test_coherence_partner_is_the_best_other_candidate():
    from ber.scale import coherence_columns

    emb = np.eye(4, dtype=np.float16)
    emb[3] = emb[0]  # pool record 3 is a copy of record 0
    rows = np.array([0, 0, 0, 1])
    cols = np.array([0, 1, 3, 2])
    p1 = np.array([0.9, 0.2, 0.5, 0.7], np.float32)
    f = coherence_columns(rows, cols, p1, emb)
    # partner of the best (col 0) is the second best (col 3); the others pair with col 0
    assert f["coh_emb"][:3].tolist() == [1.0, 0.0, 1.0]
    assert f["partner_p1"][:3].tolist() == pytest.approx([0.5, 0.9, 0.9])
    assert np.isnan(f["coh_emb"][3]) and np.isnan(f["partner_p1"][3])  # entity with one pair


def test_ce_columns_rank_only_scored_pairs():
    from ber.scale import ce_columns

    rows = np.array([0, 0, 0, 1])
    ce = np.array([0.2, np.nan, 0.9, 0.5], np.float32)
    f = ce_columns(rows, ce)
    assert f["ce_rank_s1"][[0, 2, 3]].tolist() == [1, 0, 0]
    assert np.isnan(f["ce_rank_s1"][1]) and np.isnan(f["ce_gap_s1"][1])
    assert f["ce_margin_s1"][2] == pytest.approx(0.7)
