"""Pair features for the matcher.

Every feature is a comparison between two records or a record's standing among
its competitors; none encodes a country or source identity directly, so the
model transfers to a country it never saw in training (France in the test set).
Missing information stays NaN rather than being guessed as a mismatch.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein
from rapidfuzz.process import cpdist

from ber.blocking import POOL_IDX, S1_IDX, CandidateSet

Scorer = Callable[..., float]

# Columns whose rank/gap among an entity's competitors is informative.
GROUP_COLUMNS = ("cap_score", "name_tset", "addr_tset", "full_tset")


def _pairwise(a: Sequence[str], b: Sequence[str], scorer: Scorer, scale: float) -> np.ndarray:
    if len(a) == 0:
        return np.empty(0, dtype=np.float32)
    sims = cpdist(list(a), list(b), scorer=scorer, workers=-1, dtype=np.float32) / scale
    empty = np.fromiter((not x or not y for x, y in zip(a, b)), dtype=bool, count=len(a))
    sims[empty] = np.nan
    return sims


def _sets(texts: Sequence[str]) -> list[frozenset[str]]:
    return [frozenset(t.split()) for t in texts]


def _idf(*collections: list[frozenset[str]]) -> dict[str, float]:
    df: Counter[str] = Counter()
    n = 0
    for sets in collections:
        for s in sets:
            df.update(s)
            n += 1
    return {tok: math.log((n + 1) / (count + 1)) + 1.0 for tok, count in df.items()}


def _set_features(
    left: list[frozenset[str]],
    right: list[frozenset[str]],
    rows: np.ndarray,
    cols: np.ndarray,
    idf: dict[str, float] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Jaccard and IDF-weighted Jaccard; NaN when either side is empty."""
    n = len(rows)
    jac = np.full(n, np.nan, dtype=np.float32)
    wjac = np.full(n, np.nan, dtype=np.float32)
    for i, (r, c) in enumerate(zip(rows, cols)):
        a, b = left[r], right[c]
        if not a or not b:
            continue
        inter = a & b
        union = a | b
        jac[i] = len(inter) / len(union)
        if idf is not None:
            wjac[i] = sum(idf.get(t, 1.0) for t in inter) / sum(idf.get(t, 1.0) for t in union)
    return jac, wjac


def _overlap_flags(
    left: Sequence[str], right: Sequence[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(match, conflict, missing) for space-separated code lists such as postal codes."""
    n = len(left)
    match = np.zeros(n, dtype=np.float32)
    conflict = np.zeros(n, dtype=np.float32)
    missing = np.zeros(n, dtype=np.float32)
    for i, (a, b) in enumerate(zip(left, right)):
        if not a or not b:
            missing[i] = 1.0
        elif set(a.split()) & set(b.split()):
            match[i] = 1.0
        else:
            conflict[i] = 1.0
    return match, conflict, missing


def build_features(cands: CandidateSet, s1n: pd.DataFrame, pooln: pd.DataFrame) -> pd.DataFrame:
    """Feature frame aligned row-for-row with ``cands.pairs``."""
    pairs = cands.pairs
    rows = pairs[S1_IDX].to_numpy()
    cols = pairs[POOL_IDX].to_numpy()

    def left(col: str) -> list[str]:
        return s1n[col].to_numpy()[rows].tolist()

    def right(col: str) -> list[str]:
        return pooln[col].to_numpy()[cols].tolist()

    f: dict[str, np.ndarray] = {}

    # --- names
    a, b = left("name_core"), right("name_core")
    f["name_ratio"] = _pairwise(a, b, fuzz.ratio, 100)
    f["name_partial"] = _pairwise(a, b, fuzz.partial_ratio, 100)
    f["name_tsort"] = _pairwise(a, b, fuzz.token_sort_ratio, 100)
    f["name_tset"] = _pairwise(a, b, fuzz.token_set_ratio, 100)
    f["name_jw"] = _pairwise(a, b, JaroWinkler.normalized_similarity, 1)
    f["name_lev"] = _pairwise(a, b, Levenshtein.normalized_similarity, 1)
    f["name_clean_tset"] = _pairwise(
        left("name_clean"), right("name_clean"), fuzz.token_set_ratio, 100
    )
    f["name_skel_ratio"] = _pairwise(
        left("name_skeleton"), right("name_skeleton"), fuzz.token_sort_ratio, 100
    )
    f["name_len_ratio"] = np.array(
        [min(len(x), len(y)) / max(len(x), len(y)) if x and y else np.nan for x, y in zip(a, b)],
        dtype=np.float32,
    )
    f["name_first_eq"] = np.array(
        [float(x.split()[0] == y.split()[0]) if x and y else np.nan for x, y in zip(a, b)],
        dtype=np.float32,
    )
    acr_a, acr_b = left("name_acronym"), right("name_acronym")
    f["name_acronym_match"] = np.array(
        [
            float(
                (len(xa) >= 2 and xa == y.replace(" ", ""))
                or (len(yb) >= 2 and yb == x.replace(" ", ""))
            )
            for x, y, xa, yb in zip(a, b, acr_a, acr_b)
        ],
        dtype=np.float32,
    )
    al_a, al_b = left("name_alias"), right("name_alias")
    alias_best = np.full(len(a), np.nan, dtype=np.float32)
    has_alias = np.fromiter((bool(x or y) for x, y in zip(al_a, al_b)), dtype=bool, count=len(a))
    if has_alias.any():
        idx = np.flatnonzero(has_alias)
        sub = lambda xs: [xs[i] for i in idx]  # noqa: E731
        cands_sims = [
            _pairwise(sub(a), sub(al_b), fuzz.token_set_ratio, 100),
            _pairwise(sub(al_a), sub(b), fuzz.token_set_ratio, 100),
            _pairwise(sub(al_a), sub(al_b), fuzz.token_set_ratio, 100),
        ]
        alias_best[idx] = np.nanmax(np.vstack(cands_sims), axis=0)
    f["name_alias_best"] = alias_best
    f["name_best"] = np.fmax(f["name_tset"], alias_best)

    name_sets_s1, name_sets_pool = _sets(s1n["name_core"]), _sets(pooln["name_core"])
    idf_name = _idf(name_sets_s1, name_sets_pool)
    f["name_jaccard"], f["name_idf"] = _set_features(
        name_sets_s1, name_sets_pool, rows, cols, idf_name
    )

    legal_match, legal_conflict, legal_missing = _overlap_flags(
        left("name_legal"), right("name_legal")
    )
    f["legal_match"], f["legal_conflict"], f["legal_missing"] = (
        legal_match,
        legal_conflict,
        legal_missing,
    )
    name_digits_a = [" ".join(t for t in x.split() if t.isdigit()) for x in a]
    name_digits_b = [" ".join(t for t in y.split() if t.isdigit()) for y in b]
    _, f["name_digit_conflict"], _ = _overlap_flags(name_digits_a, name_digits_b)

    # --- addresses
    a, b = left("addr_core"), right("addr_core")
    f["addr_ratio"] = _pairwise(a, b, fuzz.ratio, 100)
    f["addr_partial"] = _pairwise(a, b, fuzz.partial_ratio, 100)
    f["addr_tsort"] = _pairwise(a, b, fuzz.token_sort_ratio, 100)
    f["addr_tset"] = _pairwise(a, b, fuzz.token_set_ratio, 100)
    f["addr_jw"] = _pairwise(a, b, JaroWinkler.normalized_similarity, 1)
    f["addr_clean_tset"] = _pairwise(
        left("addr_clean"), right("addr_clean"), fuzz.token_set_ratio, 100
    )
    f["addr_missing"] = np.fromiter(
        (float(not x or not y) for x, y in zip(a, b)), dtype=np.float32, count=len(a)
    )
    addr_sets_s1, addr_sets_pool = _sets(s1n["addr_core"]), _sets(pooln["addr_core"])
    idf_addr = _idf(addr_sets_s1, addr_sets_pool)
    f["addr_jaccard"], f["addr_idf"] = _set_features(
        addr_sets_s1, addr_sets_pool, rows, cols, idf_addr
    )

    f["postal_match"], f["postal_conflict"], f["postal_missing"] = _overlap_flags(
        left("postal"), right("postal")
    )
    num_a, num_b = left("addr_numbers"), right("addr_numbers")
    _, f["num_conflict"], _ = _overlap_flags(num_a, num_b)
    num_sets_a = _sets(s1n["addr_numbers"])
    num_sets_b = _sets(pooln["addr_numbers"])
    f["num_jaccard"], _ = _set_features(num_sets_a, num_sets_b, rows, cols, None)
    f["num_first_eq"] = np.array(
        [float(x.split()[0] == y.split()[0]) if x and y else np.nan for x, y in zip(num_a, num_b)],
        dtype=np.float32,
    )
    lm_a, lm_b = left("addr_landmark"), right("addr_landmark")
    f["landmark_any"] = np.fromiter(
        (float(bool(x or y)) for x, y in zip(lm_a, lm_b)), dtype=np.float32, count=len(lm_a)
    )
    f["landmark_tset"] = _pairwise(lm_a, lm_b, fuzz.token_set_ratio, 100)

    # --- whole record
    f["full_tset"] = _pairwise(left("full_text"), right("full_text"), fuzz.token_set_ratio, 100)
    ca, cb = left("country_norm"), right("country_norm")
    f["same_country"] = np.array(
        [float(x == y) if x and y else np.nan for x, y in zip(ca, cb)], dtype=np.float32
    )
    f["cand_is_s3"] = (pooln["source"].to_numpy()[cols] == 3).astype(np.float32)

    # --- blocking evidence
    for col in [*cands.score_columns, *cands.hit_columns, "n_blockers", "cap_score", "cap_rank"]:
        f[col] = pairs[col].to_numpy(dtype=np.float32)

    feats = pd.DataFrame(f)
    feats[S1_IDX] = rows
    feats[POOL_IDX] = cols
    _add_group_features(feats)
    return feats.drop(columns=[S1_IDX, POOL_IDX])


def _second_largest(values: pd.Series, keys: pd.Series) -> pd.Series:
    """Per-row: the second-largest value in the row's group (-1 when the group has one row)."""
    frame = pd.DataFrame({"k": keys.to_numpy(), "v": values.to_numpy()})
    frame = frame.sort_values(["k", "v"], ascending=[True, False], kind="stable")
    frame["r"] = frame.groupby("k", sort=False).cumcount()
    second = frame.loc[frame["r"] == 1].set_index("k")["v"]
    return pd.Series(keys.map(second).fillna(-1.0).to_numpy(), index=values.index)


def _add_group_features(feats: pd.DataFrame) -> None:
    """How a pair ranks among the S1 entity's candidates and among the pool record's suitors."""
    for key, tag in ((S1_IDX, "s1"), (POOL_IDX, "pool")):
        keys = feats[key]
        feats[f"n_cand_{tag}"] = keys.map(keys.value_counts()).astype(np.float32)
        for col in GROUP_COLUMNS:
            values = feats[col].fillna(-1.0)
            g = values.groupby(keys, sort=False)
            feats[f"{col}_rank_{tag}"] = g.rank(ascending=False, method="min").astype(np.float32)
            feats[f"{col}_gap_{tag}"] = (values - g.transform("max")).astype(np.float32)
            feats[f"{col}_margin_{tag}"] = (values - _second_largest(values, keys)).astype(
                np.float32
            )


def feature_columns(feats: pd.DataFrame) -> list[str]:
    return list(feats.columns)
