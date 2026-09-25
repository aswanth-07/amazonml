"""Exact sparse top-k retrieval at scale: an inverted index expressed as sparse matmul.

For each query (Source 1 record) score every pool record sharing at least one
token by the dot product of L2-normalised TF-IDF vectors, and keep the k best.
Query-side columns whose pool document frequency exceeds a cap are dropped
before the product (they are uninformative and dominate its cost); they still
count in both vectors' norms, so scores stay comparable. Work is split into row
chunks run on threads: SciPy releases the GIL inside the sparse product and
NumPy inside the sort.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import scipy.sparse as sp

log = logging.getLogger(__name__)


def _topk_chunk(q: sp.csr_matrix, dt: sp.csr_matrix, k: int, offset: int):
    s = (q @ dt).tocsr()
    if s.nnz == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.float32)
    counts = np.diff(s.indptr)
    rows = np.repeat(np.arange(s.shape[0], dtype=np.int64), counts)
    order = np.lexsort((-s.data, rows))
    rank = np.arange(len(order), dtype=np.int64) - s.indptr[rows[order]]
    sel = order[rank < k]
    return rows[sel] + offset, s.indices[sel].astype(np.int64), s.data[sel].astype(np.float32)


def sparse_topk(
    queries: sp.csr_matrix,
    docs: sp.csr_matrix,
    k: int,
    n_threads: int = 12,
    chunk_rows: int = 2000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Top-k docs per query row by dot product. Returns (query_row, doc_row, score)."""
    if queries.shape[0] == 0 or docs.shape[0] == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.float32)
    dt = docs.T.tocsr()
    bounds = [
        (s, min(s + chunk_rows, queries.shape[0])) for s in range(0, queries.shape[0], chunk_rows)
    ]
    with ThreadPoolExecutor(n_threads) as ex:
        parts = list(ex.map(lambda b: _topk_chunk(queries[b[0] : b[1]], dt, k, b[0]), bounds))
    return (
        np.concatenate([p[0] for p in parts]),
        np.concatenate([p[1] for p in parts]),
        np.concatenate([p[2] for p in parts]),
    )


def partitioned_topk(
    queries: sp.csr_matrix,
    docs: sp.csr_matrix,
    q_country: np.ndarray,
    d_country: np.ndarray,
    k: int,
    n_threads: int = 12,
    chunk_rows: int = 2000,
    enabled: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """sparse_topk within each country label (records without a label join every block)."""
    if not enabled:
        return sparse_topk(queries, docs, k, n_threads, chunk_rows)
    rows, cols, scores = [], [], []
    all_q = np.arange(queries.shape[0])
    all_d = np.arange(docs.shape[0])
    unknown_d = all_d[d_country == ""]
    for label in np.unique(q_country):
        qi = all_q[q_country == label]
        di = all_d if label == "" else np.concatenate([all_d[d_country == label], unknown_d])
        if len(qi) == 0 or len(di) == 0:
            continue
        log.info("top-%d for %d queries against %d docs (country %r)", k, len(qi), len(di), label)
        r, c, s = sparse_topk(queries[qi], docs[di], k, n_threads, chunk_rows)
        rows.append(qi[r])
        cols.append(di[c])
        scores.append(s)
    if not rows:
        return np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.float32)
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(scores)
