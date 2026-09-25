"""Vectorised pair features for tens of millions of candidate pairs.

Everything here is array-at-a-time: string similarities run in RapidFuzz's
multi-threaded C++ (``cpdist``), token overlaps are sparse row dot products,
and group context (how a pair ranks among an entity's candidates and among a
pool record's suitors) is computed with sorts over the whole candidate set.
No feature names a country or a source, so the model transfers to France.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cpdist

from ber.table import Table
from ber.vectorize import row_dot, row_nnz

GROUP_VALUES = ("score", "cos_name", "cos_addr", "cos_emb")  # those present in the base


def group_stats(
    keys: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per row: (0-based rank within its key group by value desc, group max, group second max, group size)."""
    n = len(keys)
    if n == 0:
        z = np.empty(0, np.float32)
        return z, z, z, z
    order = np.lexsort((-values, keys))
    ks = keys[order]
    vs = values[order]
    new = np.empty(n, dtype=bool)
    new[0] = True
    np.not_equal(ks[1:], ks[:-1], out=new[1:])
    starts = np.flatnonzero(new)
    gid = np.cumsum(new) - 1
    sizes = np.diff(np.append(starts, n))
    best = vs[starts]
    second = np.where(sizes > 1, vs[np.minimum(starts + 1, n - 1)], -1.0)
    out_rank = np.empty(n, np.float32)
    out_rank[order] = np.arange(n) - starts[gid]
    out_best = np.empty(n, np.float32)
    out_best[order] = best[gid]
    out_second = np.empty(n, np.float32)
    out_second[order] = second[gid]
    out_size = np.empty(n, np.float32)
    out_size[order] = sizes[gid]
    return out_rank, out_best, out_second, out_size


def group_features(
    rows: np.ndarray, cols: np.ndarray, base: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Rank, gap to best and margin over runner-up, on both sides of each pair."""
    out: dict[str, np.ndarray] = {}
    for side, keys in (("s1", rows), ("pool", cols)):
        for name in GROUP_VALUES:
            if name not in base:
                continue
            v = base[name]
            rank, best, second, size = group_stats(keys, v)
            out[f"{name}_rank_{side}"] = rank
            out[f"{name}_gap_{side}"] = v - best
            out[f"{name}_margin_{side}"] = np.where(rank == 0, v - second, v - best)
            if name == "score":
                out[f"n_cand_{side}"] = size
    return out


def _lens(arr: np.ndarray) -> np.ndarray:
    return np.fromiter((len(s) for s in arr), dtype=np.int32, count=len(arr))


def _sim(a: np.ndarray, b: np.ndarray, scorer, scale: float, empty: np.ndarray) -> np.ndarray:
    out = cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32) / scale
    out[empty] = np.nan
    return out


def string_features(
    s1: Table, pool: Table, rows: np.ndarray, cols: np.ndarray
) -> dict[str, np.ndarray]:
    f: dict[str, np.ndarray] = {}
    an, bn = s1.name[rows], pool.name[cols]
    la, lb = _lens(an), _lens(bn)
    e = (la == 0) | (lb == 0)
    f["name_ratio"] = _sim(an, bn, fuzz.ratio, 100, e)
    f["name_tset"] = _sim(an, bn, fuzz.token_set_ratio, 100, e)
    f["name_tsort"] = _sim(an, bn, fuzz.token_sort_ratio, 100, e)
    f["name_partial"] = _sim(an, bn, fuzz.partial_ratio, 100, e)
    f["name_jw"] = _sim(an, bn, JaroWinkler.normalized_similarity, 1, e)
    f["name_len_s1"] = la.astype(np.float32)
    f["name_len_pool"] = lb.astype(np.float32)

    ka, kb = s1.key[rows], pool.key[cols]
    ek = (_lens(ka) == 0) | (_lens(kb) == 0)
    f["key_ratio"] = _sim(ka, kb, fuzz.ratio, 100, ek)
    f["key_tset"] = _sim(ka, kb, fuzz.token_set_ratio, 100, ek)

    sa = np.array([s.replace(" ", "") for s in an], dtype=object)
    sb = np.array([s.replace(" ", "") for s in bn], dtype=object)
    f["squash_ratio"] = _sim(sa, sb, fuzz.ratio, 100, e)
    f["squash_partial"] = _sim(sa, sb, fuzz.partial_ratio, 100, e)

    alias = pool.alias[cols]
    ea = (_lens(alias) == 0) | (la == 0)
    f["alias_tset"] = _sim(an, alias, fuzz.token_set_ratio, 100, ea)
    f["name_best"] = np.fmax(f["name_tset"], f["alias_tset"])

    ba, bb = s1.legal_bits[rows], pool.legal_bits[cols]
    both = (ba != 0) & (bb != 0)
    f["legal_match"] = (both & ((ba & bb) != 0)).astype(np.float32)
    f["legal_conflict"] = (both & ((ba & bb) == 0)).astype(np.float32)
    f["legal_missing"] = (~both).astype(np.float32)

    aa, ab = s1.addr[rows], pool.addr[cols]
    lpa, lpb = _lens(aa), _lens(ab)
    ea2 = (lpa == 0) | (lpb == 0)
    f["addr_ratio"] = _sim(aa, ab, fuzz.ratio, 100, ea2)
    f["addr_tset"] = _sim(aa, ab, fuzz.token_set_ratio, 100, ea2)
    f["addr_tsort"] = _sim(aa, ab, fuzz.token_sort_ratio, 100, ea2)
    f["addr_partial"] = _sim(aa, ab, fuzz.partial_ratio, 100, ea2)
    f["addr_len_s1"] = lpa.astype(np.float32)
    f["addr_len_pool"] = lpb.astype(np.float32)

    ma, mb = s1.landmark[rows], pool.landmark[cols]
    em = (_lens(ma) == 0) | (_lens(mb) == 0)
    f["landmark_tset"] = _sim(ma, mb, fuzz.token_set_ratio, 100, em)

    fa, fb = s1.first_num[rows], pool.first_num[cols]
    has = (fa >= 0) & (fb >= 0)
    f["first_num_eq"] = np.where(has, (fa == fb).astype(np.float32), np.nan).astype(np.float32)
    pa, pb = s1.postal[rows], pool.postal[cols]
    hasp = (pa >= 0) & (pb >= 0)
    f["postal_eq"] = np.where(hasp, (pa == pb).astype(np.float32), np.nan).astype(np.float32)

    fl_a, fl_b = s1.flags[rows], pool.flags[cols]
    f["s1_script"] = (fl_a & 1).astype(np.float32)
    f["pool_script"] = (fl_b & 1).astype(np.float32)
    f["pool_styled"] = ((fl_b & 2) > 0).astype(np.float32)
    f["pool_marker"] = ((fl_b & 4) > 0).astype(np.float32)
    f["pool_is_s3"] = (pool.source[cols] == 3).astype(np.float32)
    return f


def number_features(
    num_s1: sp.csr_matrix, num_pool: sp.csr_matrix, rows: np.ndarray, cols: np.ndarray
) -> dict[str, np.ndarray]:
    inter = row_dot(num_s1, num_pool, rows, cols)
    na = row_nnz(num_s1)[rows].astype(np.float32)
    nb = row_nnz(num_pool)[cols].astype(np.float32)
    union = na + nb - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        jac = np.where(union > 0, inter / union, np.nan).astype(np.float32)
    return {
        "num_jaccard": jac,
        "num_shared": inter,
        "num_conflict": ((na > 0) & (nb > 0) & (inter == 0)).astype(np.float32),
    }


def assemble(base: dict[str, np.ndarray], parts: list[dict[str, np.ndarray]]) -> pd.DataFrame:
    cols: dict[str, np.ndarray] = dict(base)
    for p in parts:
        cols.update(p)
    return pd.DataFrame(cols)
