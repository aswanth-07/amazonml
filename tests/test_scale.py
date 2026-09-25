"""The large-scale path: noise seen in the real data, retrieval, context features, end to end."""

from pathlib import Path

import numpy as np
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
    rows, cols = index.search(qn, qa, q_country, m_comb=4, m_name=3, m_addr=2, wn2=0.36, wa2=0.64, n_threads=2, chunk_rows=7)
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
