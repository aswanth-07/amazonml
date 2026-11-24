"""Multi-view inverted-index search, built once and queried in chunks.

The combined score of the blocking vector is ``wn^2 * name_cosine + wa^2 *
address_cosine``, so one pair of sparse products per query chunk (queries x
name index, queries x address index) yields three candidate lists at once:
top-m by name, top-m by address, and top-m by their weighted sum. Per-country
transposed indexes are built once and reused by every chunk.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import scipy.sparse as sp

log = logging.getLogger(__name__)

_EMPTY = (np.empty(0, np.int64), np.empty(0, np.int64))


def _topk_rows(s: sp.csr_matrix, k: int) -> tuple[np.ndarray, np.ndarray]:
    if k <= 0 or s.nnz == 0:
        return _EMPTY
    counts = np.diff(s.indptr)
    rows = np.repeat(np.arange(s.shape[0], dtype=np.int64), counts)
    order = np.lexsort((-s.data, rows))
    rank = np.arange(len(order), dtype=np.int64) - s.indptr[rows[order]]
    sel = order[rank < k]
    return rows[sel], s.indices[sel].astype(np.int64)


class CountryIndex:
    """Name and address indexes over documents, partitioned by country label.

    A query sees documents of its own label plus documents without one; a query
    without a label sees every document. Labels are an open set.
    """

    def __init__(
        self, dn: sp.csr_matrix, da: sp.csr_matrix, country: np.ndarray, partition: bool = True
    ) -> None:
        self.n_docs = dn.shape[0]
        self.partition = partition
        self._dn, self._da, self._country = dn, da, country
        self._groups: dict[str, tuple[np.ndarray, sp.csr_matrix, sp.csr_matrix]] = {}
        labels = [lab for lab in np.unique(country) if lab != ""] if partition else ["*"]
        for label in labels:
            self._build(label)

    def _build(self, label: str) -> None:
        all_docs = np.arange(self.n_docs)
        if label == "*" or (label == "" and self.partition):
            di = all_docs
        else:
            di = np.concatenate([all_docs[self._country == label], all_docs[self._country == ""]])
        self._groups[label] = (di, self._dn[di].T.tocsr(), self._da[di].T.tocsr())
        log.info("index %r: %d documents", label, len(di))

    def _group(self, label: str):
        key = label if self.partition else "*"
        if key not in self._groups:
            self._build(key)
        return self._groups[key]

    def search(
        self,
        qn: sp.csr_matrix,
        qa: sp.csr_matrix,
        q_country: np.ndarray,
        m_comb: int,
        m_name: int,
        m_addr: int,
        wn2: float,
        wa2: float,
        n_threads: int = 16,
        chunk_rows: int = 500,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Candidate (query row, document) pairs from all three lists, possibly with duplicates."""
        labels = np.unique(q_country) if self.partition else np.array(["*"])
        rows_out, cols_out = [], []
        for label in labels:
            qi = (
                np.arange(qn.shape[0]) if not self.partition else np.flatnonzero(q_country == label)
            )
            if len(qi) == 0:
                continue
            di, dnt, dat = self._group(label)
            qn_l, qa_l = qn[qi], qa[qi]

            def work(bounds, qn_l=qn_l, qa_l=qa_l, dnt=dnt, dat=dat):
                a, b = bounds
                sn = (qn_l[a:b] @ dnt).tocsr()
                sa = (qa_l[a:b] @ dat).tocsr()
                parts = [_topk_rows(sn, m_name), _topk_rows(sa, m_addr)]
                if m_comb > 0:
                    parts.append(_topk_rows((wn2 * sn + wa2 * sa).tocsr(), m_comb))
                r = np.concatenate([p[0] for p in parts]) + a
                c = np.concatenate([p[1] for p in parts])
                return r, c

            bounds = [(s, min(s + chunk_rows, len(qi))) for s in range(0, len(qi), chunk_rows)]
            with ThreadPoolExecutor(n_threads) as ex:
                for r, c in ex.map(work, bounds):
                    rows_out.append(qi[r])
                    cols_out.append(di[c])
        if not rows_out:
            return _EMPTY
        return np.concatenate(rows_out), np.concatenate(cols_out)
