"""Hashed, IDF-weighted token matrices for blocking and pair similarities.

Tokens are hashed (MurmurHash3 through scikit-learn's HashingVectorizer), so
there is no vocabulary to hold in memory, and the hashing is deterministic
across processes and runs. IDF is fitted on the records being matched (both
sides together), which is unsupervised and uses no external data.

Views
- ``name``: core-name and alias tokens plus their consonant keys, so a typo or
  a romanised Indic spelling still shares a key with the English spelling.
- ``addr``: address tokens (street words, numbers with leading zeros stripped,
  city and state codes, landmark words).
- ``num``: address numbers alone, for number-overlap features.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from itertools import pairwise

import numpy as np
import scipy.sparse as sp
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import HashingVectorizer

from ber.normalize.text import consonant_key
from ber.table import Table

N_FEATURES = 1 << 22


def _split(doc: str) -> list[str]:
    return doc.split()


_HASHER = HashingVectorizer(
    analyzer=_split,
    n_features=N_FEATURES,
    alternate_sign=False,
    norm=None,
    binary=True,
    dtype=np.float32,
)


GRAM = 4


def _grams(squashed: str) -> list[str]:
    if len(squashed) <= GRAM:
        return [squashed] if squashed else []
    return [squashed[i : i + GRAM] for i in range(len(squashed) - GRAM + 1)]


def name_docs(t: Table) -> list[str]:
    """Words, consonant keys, the squashed name (for "dentalcare" / "pinktraders"
    style names) and its character 4-grams (typos: "sepcialsss")."""
    out = []
    for name, alias in zip(t.name, t.alias):
        toks = f"{name} {alias}".split()
        doc = [f"n:{w}" for w in toks]
        doc += [f"k:{k}" for k in (consonant_key(w) for w in toks) if len(k) >= 2]
        for part in (name, alias):
            sq = part.replace(" ", "")
            if sq:
                doc.append(f"s:{sq}")
                doc += [f"g:{g}" for g in set(_grams(sq))]
        out.append(" ".join(doc))
    return out


def addr_docs(t: Table) -> list[str]:
    """Address words plus adjacent-word bigrams ("206_walnut", "walnut_st"): a bigram is
    rare where both of its words are common, which is what makes it searchable."""
    out = []
    for a, lm in zip(t.addr, t.landmark):
        words = a.split()
        doc = [f"a:{w}" for w in words]
        doc += [f"b:{x}_{y}" for x, y in pairwise(words)]
        doc += [f"a:{w}" for w in lm.split()]
        out.append(" ".join(doc))
    return out


def num_docs(t: Table) -> list[str]:
    return list(t.numbers)


def _hash(docs: list[str]) -> sp.csr_matrix:
    return _HASHER.transform(docs)


def hash_docs(docs: list[str], n_jobs: int = 12, chunk: int = 250_000) -> sp.csr_matrix:
    """Binary hashed token matrix, built in parallel processes (the analyzer is Python)."""
    parts = [docs[i : i + chunk] for i in range(0, len(docs), chunk)] or [[]]
    if len(parts) == 1:
        mats = [_hash(parts[0])]
    else:
        mats = Parallel(n_jobs=n_jobs, batch_size=1)(delayed(_hash)(p) for p in parts)
    return sp.vstack(mats, format="csr").astype(np.float32)


def idf_weights(*mats: sp.csr_matrix) -> np.ndarray:
    n = sum(m.shape[0] for m in mats)
    df = np.zeros(N_FEATURES, dtype=np.int64)
    for m in mats:
        df += np.bincount(m.indices, minlength=N_FEATURES)
    return (np.log((n + 1) / (df + 1)) + 1.0).astype(np.float32), df


def weight_and_normalize(m: sp.csr_matrix, idf: np.ndarray) -> sp.csr_matrix:
    m = m.copy()
    m.data *= idf[m.indices]
    norms = np.sqrt(np.asarray(m.multiply(m).sum(axis=1)).ravel())
    norms[norms == 0] = 1.0
    m.data /= np.repeat(norms, np.diff(m.indptr)).astype(np.float32)
    return m


def prefix_filter(m: sp.csr_matrix, df: np.ndarray, budget: int, cap: int) -> sp.csr_matrix:
    """Keep, per row, its rarest tokens (df <= cap) until their summed df reaches ``budget``.

    This bounds the work of an inverted-index search per query (the sum of the
    posting lengths it touches) while searching on the most discriminative
    tokens. The rarest token is always kept. Scores computed on the filtered
    query are only used to generate candidates; they are re-scored in full.
    """
    m = m.tocsr(copy=True)
    n = m.shape[0]
    counts = np.diff(m.indptr)
    rows = np.repeat(np.arange(n), counts)
    tok_df = df[m.indices].astype(np.int64)
    order = np.lexsort((tok_df, rows))
    sorted_df = tok_df[order]
    csum = np.cumsum(sorted_df)
    start = np.repeat(m.indptr[:-1], counts)
    before = (
        csum - sorted_df - np.concatenate([[0], csum])[start]
    )  # df summed over rarer tokens in the row
    first = np.arange(len(order)) == start
    keep_sorted = (sorted_df <= cap) & ((before + sorted_df <= budget) | first)
    keep = np.empty(len(order), dtype=bool)
    keep[order] = keep_sorted
    m.data[~keep] = 0
    m.eliminate_zeros()
    return m


def prune_columns(m: sp.csr_matrix, keep: np.ndarray) -> sp.csr_matrix:
    """Zero the entries of columns where ``keep`` is False (used on the query side)."""
    m = m.copy()
    m.data[~keep[m.indices]] = 0
    m.eliminate_zeros()
    return m


def row_dot(
    a: sp.csr_matrix,
    b: sp.csr_matrix,
    rows: np.ndarray,
    cols: np.ndarray,
    chunk: int = 500_000,
    n_threads: int = 12,
) -> np.ndarray:
    """a[rows[i]] . b[cols[i]] for every i, in chunks run on threads (SciPy releases the GIL)."""
    out = np.empty(len(rows), dtype=np.float32)

    def work(s: int) -> None:
        e = s + chunk
        out[s:e] = np.asarray(a[rows[s:e]].multiply(b[cols[s:e]]).sum(axis=1)).ravel()

    starts = range(0, len(rows), chunk)
    if len(rows) <= chunk or n_threads <= 1:
        for s in starts:
            work(s)
    else:
        with ThreadPoolExecutor(n_threads) as ex:
            list(ex.map(work, starts))
    return out


def row_nnz(m: sp.csr_matrix) -> np.ndarray:
    return np.diff(m.indptr).astype(np.int32)
