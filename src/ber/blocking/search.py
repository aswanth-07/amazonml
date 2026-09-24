"""Memory-bounded top-k cosine search and pairwise cosine lookup."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import scipy.sparse as sp

Partition = tuple[np.ndarray, np.ndarray]


def country_partitions(
    s1_country: np.ndarray, pool_country: np.ndarray, enabled: bool
) -> Iterator[Partition]:
    """Yield (s1 positions, pool positions) blocks that may be compared.

    With partitioning on, an S1 record sees pool records of its own country and
    pool records with no country; an S1 record with no country sees everything.
    Country labels are an open set: nothing here knows which countries exist.
    """
    all_s1 = np.arange(len(s1_country))
    all_pool = np.arange(len(pool_country))
    if not enabled:
        yield all_s1, all_pool
        return
    unknown_pool = all_pool[pool_country == ""]
    for label in np.unique(s1_country):
        s1_pos = all_s1[s1_country == label]
        if label == "":
            yield s1_pos, all_pool
        else:
            yield s1_pos, np.concatenate([all_pool[pool_country == label], unknown_pool])


def _chunk_rows(n_cols: int, budget_mb: int) -> int:
    return max(1, int(budget_mb * 2**20 // (4 * max(n_cols, 1))))


def topk_cosine(
    queries: sp.csr_matrix | np.ndarray,
    docs: sp.csr_matrix | np.ndarray,
    k: int,
    min_sim: float = 0.0,
    budget_mb: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each query row, the ``k`` most similar doc rows with similarity > ``min_sim``.

    Rows of both inputs must be L2-normalised. Works for sparse and dense inputs,
    materialising at most ``budget_mb`` of dense similarities at a time.
    Returns (query_row, doc_row, similarity) arrays.
    """
    n_q, n_d = queries.shape[0], docs.shape[0]
    if n_q == 0 or n_d == 0 or k <= 0:
        return _empty()
    k = min(k, n_d)
    sparse = sp.issparse(docs)
    docs_t = docs.T.tocsr() if sparse else np.ascontiguousarray(docs.T)
    step = _chunk_rows(n_d, budget_mb)
    out_r, out_c, out_s = [], [], []
    for start in range(0, n_q, step):
        stop = min(start + step, n_q)
        block = queries[start:stop] @ docs_t
        sims = block.toarray() if sp.issparse(block) else np.asarray(block)
        sims = sims.astype(np.float32, copy=False)
        if k < n_d:
            idx = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        else:
            idx = np.broadcast_to(np.arange(n_d), (stop - start, n_d))
        top = np.take_along_axis(sims, idx, axis=1)
        keep = top > min_sim
        rows = np.broadcast_to(np.arange(start, stop)[:, None], idx.shape)
        out_r.append(rows[keep])
        out_c.append(idx[keep])
        out_s.append(top[keep])
    return (
        np.concatenate(out_r).astype(np.int64),
        np.concatenate(out_c).astype(np.int64),
        np.concatenate(out_s).astype(np.float32),
    )


def topk_partitioned(
    queries: sp.csr_matrix | np.ndarray,
    docs: sp.csr_matrix | np.ndarray,
    partitions: list[Partition],
    k: int,
    min_sim: float,
    budget_mb: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, cols, sims = [], [], []
    for s1_pos, pool_pos in partitions:
        if len(s1_pos) == 0 or len(pool_pos) == 0:
            continue
        r, c, s = topk_cosine(queries[s1_pos], docs[pool_pos], k, min_sim, budget_mb)
        rows.append(s1_pos[r])
        cols.append(pool_pos[c])
        sims.append(s)
    if not rows:
        return _empty()
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(sims)


def pair_cosine(
    left: sp.csr_matrix | np.ndarray,
    right: sp.csr_matrix | np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    chunk: int = 200_000,
) -> np.ndarray:
    """Cosine of (left[rows[i]], right[cols[i]]) for every i, for L2-normalised rows."""
    out = np.empty(len(rows), dtype=np.float32)
    for start in range(0, len(rows), chunk):
        stop = start + chunk
        a, b = left[rows[start:stop]], right[cols[start:stop]]
        if sp.issparse(a):
            out[start:stop] = np.asarray(a.multiply(b).sum(axis=1)).ravel()
        else:
            out[start:stop] = np.einsum("ij,ij->i", a, b)
    return out


def _empty() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.float32)
