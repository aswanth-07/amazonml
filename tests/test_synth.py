from ber import schema
from ber.io import load_split


def test_layout_and_contract(synthetic_dir):
    for split in ("train", "test"):
        s = load_split(synthetic_dir, split)
        assert s.truth is not None
        assert set(s.truth) == set(s.s1[schema.ENTITY_ID])
        pool_ids = set(s.pool[schema.ENTITY_ID])
        assert all(m in pool_ids for ms in s.truth.values() for m in ms)
        owners = [m for ms in s.truth.values() for m in ms]
        assert len(owners) == len(set(owners))  # Source 1 is deduplicated


def test_france_only_in_test(synthetic_dir):
    train = load_split(synthetic_dir, "train")
    test = load_split(synthetic_dir, "test")
    assert "France" not in set(train.s1[schema.COUNTRY]) | set(train.pool[schema.COUNTRY])
    assert "France" in set(test.s1[schema.COUNTRY])


def test_has_singletons_and_multi_matches(synthetic_dir):
    truth = load_split(synthetic_dir, "train").truth
    sizes = [len(v) for v in truth.values()]
    assert 0 in sizes
    assert max(sizes) >= 2
