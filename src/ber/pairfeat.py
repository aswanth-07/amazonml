"""Vectorised pair features for tens of millions of candidate pairs.

Everything here is array-at-a-time: string similarities run in RapidFuzz's
multi-threaded C++ (``cpdist``), token overlaps are sparse row dot products,
and group context (how a pair ranks among an entity's candidates and among a
pool record's suitors) is computed with sorts over the whole candidate set.
No feature names a country or a source, so the model transfers to France.
"""

from __future__ import annotations

import re
import zlib

import numpy as np
import pandas as pd
import scipy.sparse as sp
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein
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


def _digits(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0, np.floor(np.log10(np.maximum(x, 1))) + 1, 1).astype(np.int64)


def detail_features(
    s1: Table, pool: Table, rows: np.ndarray, cols: np.ndarray
) -> dict[str, np.ndarray]:
    """Fine differences that separate a business from its neighbour or namesake.

    Wrong merges on train are mostly the same name at a nearby house number
    ("450 ma ave" vs "471 ma ave", "1015" vs "1018") or a name one short token
    apart ("wpq realtors" vs "rpq realtors"). Equality alone cannot tell a typo
    from a neighbour: how far apart, and whether one number is a truncation of
    the other, can.
    """
    fa, fb = s1.first_num[rows], pool.first_num[cols]
    both = (fa >= 0) & (fb >= 0)
    f: dict[str, np.ndarray] = {}
    diff = np.abs(fa - fb).astype(np.float64)
    f["first_num_logdiff"] = np.where(both, np.log1p(diff), np.nan).astype(np.float32)
    da, db = _digits(fa), _digits(fb)
    shift = np.abs(da - db)
    longer, shorter = np.where(da >= db, fa, fb), np.where(da >= db, fb, fa)
    prefix = (longer // np.power(10, np.minimum(shift, 18))) == shorter
    f["first_num_prefix"] = np.where(both & (shift > 0), prefix, np.nan).astype(np.float32)
    f["first_num_digits_diff"] = np.where(both, shift, np.nan).astype(np.float32)
    sa = np.where(both, fa, 0).astype(str).astype(object)
    sb = np.where(both, fb, 0).astype(str).astype(object)
    edit = cpdist(sa, sb, scorer=Levenshtein.distance, workers=-1, dtype=np.int32)
    f["first_num_edit"] = np.where(both, edit, np.nan).astype(np.float32)

    n = len(rows)
    extra_a = np.empty(n, np.float32)
    extra_b = np.empty(n, np.float32)
    short = np.empty(n, np.float32)
    for i, (a, b) in enumerate(zip(s1.name[rows], pool.name[cols])):
        ta, tb = set(a.split()), set(b.split())
        xa, xb = ta - tb, tb - ta
        extra_a[i] = len(xa)
        extra_b[i] = len(xb)
        short[i] = any(len(t) <= 3 for t in xa) or any(len(t) <= 3 for t in xb)
    f["name_extra_tok_s1"] = extra_a
    f["name_extra_tok_pool"] = extra_b
    f["name_short_tok_diff"] = short
    return f


_DIGIT = re.compile(r"\d")


def _street_keys(addrs: np.ndarray) -> np.ndarray:
    """A hash of an address's non-numeric tokens, order-free (reordered copies agree)."""
    return np.fromiter(
        (
            zlib.crc32(" ".join(sorted({t for t in a.split() if not _DIGIT.search(t)})).encode())
            for a in addrs
        ),
        np.int64,
        len(addrs),
    )


def _count_in(keys: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """For each key, how many times it occurs in ``reference``."""
    uniq, counts = np.unique(reference, return_counts=True)
    at = np.minimum(np.searchsorted(uniq, keys), max(len(uniq) - 1, 0))
    return np.where((len(uniq) > 0) & (uniq[at] == keys), counts[at], 0).astype(np.float32)


_MIX = 1_000_000_007


def _counter(keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    uniq, counts = np.unique(keys, return_counts=True)
    return uniq, counts.astype(np.float32)


def _lookup(keys: np.ndarray, counter: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    uniq, counts = counter
    if len(uniq) == 0:
        return np.zeros(len(keys), dtype=np.float32)
    at = np.minimum(np.searchsorted(uniq, keys), len(uniq) - 1)
    return np.where(uniq[at] == keys, counts[at], 0).astype(np.float32)


def cluster_features(rows: np.ndarray, cols: np.ndarray, stats: dict) -> dict[str, np.ndarray]:
    """Where the candidate's name lives, from its name cluster in the pool.

    Over the other pool records with the candidate's (country, name): how many sit at the
    S1 record's address (support), on its street, and at another non-empty address
    (conflict). An address-less candidate cannot be placed by its own address, but its name
    cluster can: a name that only ever appears at the S1 address hands its address-less
    member to that business, one that also appears elsewhere has a namesake. In v3, 44% of
    missed true pairs lost their record to another suitor, typically a namesake.
    """
    nk = stats["pool_name_key"][cols]
    a_s, st_s = stats["s1_addr_key"][rows], stats["s1_street_key"][rows]
    own_full = ~stats["pool_empty"][cols]
    own_addr = (own_full & (stats["pool_addr_key"][cols] == a_s)).astype(np.float32)
    own_street = (own_full & (stats["pool_street_key"][cols] == st_s)).astype(np.float32)
    support = _lookup(nk * _MIX + a_s, stats["cluster_addr"]) - own_addr
    street = _lookup(nk * _MIX + st_s, stats["cluster_street"]) - own_street
    total = _lookup(nk, stats["cluster_total"]) - own_full.astype(np.float32)
    return {
        "cluster_addr_support": support,
        "cluster_street_support": street,
        "cluster_conflict": total - street,
        "cluster_located": total,
        "pool_addr_empty": (~own_full).astype(np.float32),
    }


def collective_stats(s1: Table, pool: Table) -> dict[str, np.ndarray]:
    """Split-wide statistics of each record's address and name (computed once per split).

    Keys combine the country, the order-free street tokens and the first house number, so
    "12 MG Rd, Pune" and "Pune, 12 MG Road" share a key.
    """
    codes, _ = pd.factorize(np.concatenate([s1.country, pool.country]))
    c1, cp = codes[: len(s1)].astype(np.int64), codes[len(s1) :].astype(np.int64)
    st1, stp = _street_keys(s1.addr), _street_keys(pool.addr)
    k_street1, k_streetp = c1 * 4_294_967_311 + st1, cp * 4_294_967_311 + stp
    k_addr1 = k_street1 * 1_000_003 + s1.first_num
    k_addrp = k_streetp * 1_000_003 + pool.first_num
    n1 = pd.Series(s1.name).astype(str).to_numpy()
    npool = pd.Series(pool.name).astype(str).to_numpy()
    aliases: dict[str, list[str]] = {}
    for core, alias in zip(npool, pool.alias):
        if alias:
            aliases.setdefault(core, []).append(alias)
    # name clusters: where the pool records of each (country, name) sit
    name_p = cp * 4_294_967_311 + pd.util.hash_array(npool).astype(np.int64)
    empty_p = np.fromiter((not a for a in pool.addr), dtype=bool, count=len(pool))
    full = ~empty_p
    return {
        "pool_name_key": name_p,
        "pool_addr_key": k_addrp,
        "pool_street_key": k_streetp,
        "pool_empty": empty_p,
        "s1_addr_key": k_addr1,
        "s1_street_key": k_street1,
        "cluster_addr": _counter(name_p[full] * _MIX + k_addrp[full]),
        "cluster_street": _counter(name_p[full] * _MIX + k_streetp[full]),
        "cluster_total": _counter(name_p[full]),
        "s1_at_addr_of_pool": _count_in(k_addrp, k_addr1),  # Source 1 businesses at p's address
        "s1_at_own_addr": _count_in(k_addr1, k_addr1),  # ... at s's own address
        "pool_at_addr_of_pool": _count_in(k_addrp, k_addrp),  # pool records at p's address
        "s1_on_street_of_pool": _count_in(
            k_streetp, k_street1
        ),  # Source 1 businesses on p's street
        "s1_named_as_pool": _count_in(pd.util.hash_array(npool), pd.util.hash_array(n1)),
        "pool_named_as_pool": _count_in(pd.util.hash_array(npool), pd.util.hash_array(npool)),
        "aliases": aliases,
    }


def collective_features(
    s1: Table, pool: Table, rows: np.ndarray, cols: np.ndarray, stats: dict
) -> dict[str, np.ndarray]:
    """Evidence from the rest of the split rather than the pair alone.

    Measured on train: with the same street and number but different names, P(match) is
    0.92 when one Source 1 business has that address and 0.08 when six or more do; a
    differing house number that recurs across other pool records marks a real value (0.95
    as top candidate) rather than a one-off; and a rebranded copy is linked to its entity
    by a sibling record whose "formerly X" alias names it (0.71 against 0.11).
    """
    f = {
        "addr_s1_density": stats["s1_at_addr_of_pool"][cols],
        "addr_s1_density_self": stats["s1_at_own_addr"][rows],
        "addr_pool_recurrence": stats["pool_at_addr_of_pool"][cols],
        "street_s1_density": stats["s1_on_street_of_pool"][cols],
        "name_s1_namesakes": stats["s1_named_as_pool"][cols],
        "name_pool_recurrence": stats["pool_named_as_pool"][cols],
    }
    aliases = stats["aliases"]
    link = np.full(len(rows), np.nan, dtype=np.float32)
    for i, (s, p) in enumerate(zip(s1.name[rows], pool.name[cols])):
        found = aliases.get(p)
        if found:
            link[i] = max(fuzz.token_set_ratio(s, a) for a in found[:8]) / 100.0
    f["alias_link"] = link
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
