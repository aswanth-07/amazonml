import numpy as np
import pytest
import scipy.sparse as sp

from ber.blocking import CandidateGenerator
from ber.blocking.blockers import key_candidates
from ber.blocking.search import country_partitions, pair_cosine, topk_cosine
from ber.config import BlockingConfig
from ber.io import load_split
from ber.normalize import normalize_records


def _normalised(rng: np.random.Generator, n: int, d: int, sparse: bool):
    x = rng.random((n, d)).astype(np.float32)
    x[x < 0.7] = 0
    x[:, 0] += 0.01  # no all-zero rows
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return sp.csr_matrix(x) if sparse else x


@pytest.mark.parametrize("sparse", [True, False])
@pytest.mark.parametrize("budget_mb", [512, 0])
def test_topk_matches_brute_force(sparse: bool, budget_mb: int):
    rng = np.random.default_rng(0)
    q, d = _normalised(rng, 37, 20, sparse), _normalised(rng, 53, 20, sparse)
    rows, cols, sims = topk_cosine(q, d, k=5, min_sim=0.0, budget_mb=budget_mb)
    dense = (q @ d.T).toarray() if sparse else q @ d.T
    for i in range(37):
        expected = set(np.argsort(-dense[i])[:5])
        assert set(cols[rows == i]) == expected
    np.testing.assert_allclose(sims, dense[rows, cols], rtol=1e-5)
    np.testing.assert_allclose(pair_cosine(q, d, rows, cols), sims, rtol=1e-5)


def test_topk_min_sim_and_small_pool():
    q = sp.csr_matrix(np.array([[1.0, 0.0]], dtype=np.float32))
    d = sp.csr_matrix(np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32))
    rows, cols, _ = topk_cosine(q, d, k=10, min_sim=0.5)
    assert rows.tolist() == [0] and cols.tolist() == [1]


def test_country_partitions():
    s1 = np.array(["us", "india", ""])
    pool = np.array(["us", "india", "", "france"])
    parts = {tuple(a): tuple(b) for a, b in country_partitions(s1, pool, enabled=True)}
    assert parts[(0,)] == (0, 2)  # own country + unknown
    assert parts[(1,)] == (1, 2)
    assert parts[(2,)] == (0, 1, 2, 3)  # unknown country sees everything
    (only,) = list(country_partitions(s1, pool, enabled=False))
    assert len(only[0]) == 3 and len(only[1]) == 4


def test_key_candidates_skip_oversized_blocks():
    s1 = [{"a"}, {"b"}, set()]
    pool = [{"a"}, {"a"}, {"a"}, {"b"}]
    rows, cols = key_candidates(s1, pool, max_block_size=2)
    assert list(zip(rows.tolist(), cols.tolist())) == [(1, 3)]


def test_generator_on_synthetic(synthetic_dir):
    split = load_split(synthetic_dir, "test")
    s1n, pooln = normalize_records(split.s1), normalize_records(split.pool)
    cfg = BlockingConfig(max_candidates=20)
    cands = CandidateGenerator(cfg).generate(s1n, pooln)
    pairs = cands.pairs
    assert pairs.groupby("s1_idx").size().max() <= 20
    assert not pairs.duplicated(["s1_idx", "pool_idx"]).any()
    a = s1n["country_norm"].to_numpy()[pairs["s1_idx"]]
    b = pooln["country_norm"].to_numpy()[pairs["pool_idx"]]
    assert ((a == b) | (a == "") | (b == "")).all()
    for col in cands.score_columns:
        assert pairs[col].between(-1e-4, 1 + 1e-4).all()
    lists = cands.id_lists(s1n["entity_id"].to_numpy(), pooln["entity_id"].to_numpy())
    assert set(lists) == set(split.s1_ids)
    # France is blocked like any other country
    fr = s1n.index[s1n["country_norm"] == "france"]
    assert len(fr) and pairs["s1_idx"].isin(fr).any()
