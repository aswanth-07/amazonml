"""The large-scale pipeline: millions of Source 1 records against ten million others.

1. **Tables.** Parallel normalisation, cached per split and source.
2. **Blocking.** Hashed TF-IDF over name, consonant-key and address tokens;
   exact inverted-index top-k per Source 1 record within its country, with
   frequent tokens excluded from the search (not from the norms). The top-k is
   the candidate set: it is what the matcher scores and what
   ``candidate_pairs.tsv`` reports.
3. **Context.** Cosines per view, then every pair's rank, gap and margin among
   its entity's candidates and among its pool record's suitors, over the whole
   candidate set.
4. **Stage 1.** LightGBM on a sample of Source 1 entities (grouped folds), so
   every sampled pair gets an out-of-fold probability and every other pair a
   fold-average one from models that never saw its entity.
5. **Stage 2 (stacking).** The same context features recomputed on stage-1
   probabilities: whether this is the entity's best candidate and the pool
   record's best suitor, and by how much.
6. **Decision.** Exclusive assignment, then one threshold tuned for the exact
   macro F0.5 on the sample; reported on the untouched holdout entities.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp

from ber import __version__, schema
from ber.blocking.multiview import CountryIndex
from ber.config import Config, dump_config
from ber.decision import macro_from_mask
from ber.io import load_split, read_source
from ber.matcher import group_folds
from ber.metrics import (
    breakdown,
    cardinality_labels,
    fbeta_from_counts,
    macro_fbeta,
    precision_recall,
)
from ber.pairfeat import (
    cluster_features,
    collective_features,
    collective_stats,
    detail_features,
    group_features,
    group_stats,
    number_features,
    string_features,
)
from ber.submission import validate_outputs, write_outputs
from ber.table import Table, build_table, release_workers
from ber.utils import git_revision, seed_everything, timed
from ber.vectorize import (
    addr_docs,
    hash_docs,
    idf_weights,
    name_docs,
    num_docs,
    prefix_filter,
    prune_columns,
    row_dot,
    weight_and_normalize,
)

log = logging.getLogger(__name__)

STAGE2_KEEP = (
    "score",
    "cos_name",
    "cos_addr",
    "name_tset",
    "name_best",
    "addr_tset",
    "key_tset",
    "first_num_eq",
    "squash_partial",
    "num_jaccard",
    "cos_emb",
)  # fmt: skip  (those present in the feature matrix)


BLOCKING_VERSION = 1  # bump when tokens, weighting or search change which pairs blocking returns
FEATURE_VERSION = 4  # bump when the stage-1 pair features change


# ----------------------------------------------------------------------------- tables
def _sha(raw: str) -> str:
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


def _cache_key(path: Path, cfg: Config) -> str:
    """Identify a source file's cached table by its location, size, mtime and normaliser settings."""
    st = path.stat()
    return _sha(
        f"{path.resolve()}|{st.st_size}|{st.st_mtime_ns}|{cfg.normalize.transliterate}|{__version__}"
    )


def split_key(data_dir: str | Path, split: str, cfg: Config) -> str:
    """Identify a split's candidate set: its three tables plus every setting blocking reads."""
    sc = cfg.scale
    tables = [
        _cache_key(Path(data_dir) / split / schema.source_file(split, s), cfg) for s in (1, 2, 3)
    ]
    settings = (
        sc.k, sc.k_name, sc.k_addr, sc.k_reverse, sc.generate, sc.generate_name, sc.generate_addr,
        sc.generate_reverse, sc.name_weight, sc.df_cap, sc.query_budget,
        cfg.blocking.partition_by_country, BLOCKING_VERSION,
    )  # fmt: skip
    return _sha(f"{split}|{tables}|{settings}")


def load_tables(data_dir: str | Path, split: str, cfg: Config) -> tuple[Table, Table]:
    cache = Path(cfg.scale.cache_dir)
    tabs = {}
    for s in (1, 2, 3):
        src = Path(data_dir) / split / schema.source_file(split, s)
        path = cache / f"{split}_s{s}_{_cache_key(src, cfg)}.table"
        if path.exists():
            tabs[s] = Table.load(path)
            continue
        with timed(f"{split}: normalise source {s}"):
            df = read_source(src, s, cfg.data.quoting)
            tabs[s] = build_table(df, s, cfg.normalize.transliterate, cfg.scale.n_jobs)
            del df
        tabs[s].save(path)
    release_workers()
    return tabs[1], Table.concat([tabs[2], tabs[3]])


# ----------------------------------------------------------------------------- blocking
@dataclass
class SplitData:
    name: str
    s1: Table
    pool: Table
    rows: np.ndarray  # candidate pairs: S1 position
    cols: np.ndarray  # candidate pairs: pool position
    base: dict[str, np.ndarray]  # score, cosines and group context, aligned with rows/cols
    num_q: sp.csr_matrix
    num_d: sp.csr_matrix
    stats: dict[str, float] = field(default_factory=dict)
    key: str = ""  # identifies the candidate set for the caches; empty: do not cache
    # the token-view candidate set this one extends (key, rows, cols): its cached features
    # are reused for every pair the two sets share
    donor: tuple[str, np.ndarray, np.ndarray] | None = None
    extras: dict[str, Any] = field(default_factory=dict)  # split-wide statistics, computed once


def block_split(
    s1: Table, pool: Table, cfg: Config, name: str, key: str = "", drop: np.ndarray | None = None
) -> SplitData:
    """Candidate pairs, their cosines and group context. With a ``key``, the pairs, cosines
    and number matrices are cached, so a rerun skips hashing and search entirely."""
    path = Path(cfg.scale.cache_dir) / f"{name}_blocking_{key}.joblib" if key else None
    views = None
    if path is not None and path.exists():
        with timed(f"{name}: load cached blocking {path.name}"):
            c = joblib.load(path)
        rows, cols, base, mq, md = c["rows"], c["cols"], c["base"], c["num_q"], c["num_d"]
    else:
        views, mq, md = _views(s1, pool, cfg, name)
        rows, cols, base = _block(views, s1, pool, cfg, name)
        if path is not None:
            with timed(f"{name}: save blocking cache {path.name}"):
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(".tmp")
                joblib.dump(
                    {"rows": rows, "cols": cols, "base": base, "num_q": mq, "num_d": md}, tmp
                )
                tmp.replace(path)
    donor = (key, rows, cols) if key else None
    if drop is not None and drop.any():
        # businesses removed from Source 1 (a simulation of test's orphaned copies): their
        # pairs go, their pool copies stay and fall to the next-best suitor below
        keep = ~drop[rows]
        rows, cols = rows[keep], cols[keep]
        base = {k: v[keep] for k, v in base.items()}
        key = _sha(f"{key}|drop|{int(drop.sum())}|{_sha(str(np.flatnonzero(drop)[:64].tolist()))}") if key else ""  # fmt: skip
    if cfg.scale.emb_dir:
        from ber.embed import embedding_key

        rows, cols, base = _with_embeddings(s1, pool, cfg, name, rows, cols, base, views, drop)
        sc = cfg.scale
        key = _sha(f"{key}|{sc.k_emb}|{sc.k_emb_reverse}|{embedding_key(sc.emb_dir, name)}") if key else ""  # fmt: skip
    elif drop is None or not drop.any():
        donor = None  # nothing to extend: the candidate set is the cached one itself
    del views
    with timed(f"{name}: group context"):
        base.update(group_features(rows, cols, base))
    stats = {
        "pairs": float(len(rows)),
        "candidates_per_s1": float(len(rows) / max(len(s1), 1)),
        "s1_without_candidates": float(len(s1) - len(np.unique(rows))),
        "reduction_ratio": float(1 - len(rows) / max(len(s1) * len(pool), 1)),
    }
    log.info("%s: %s", name, stats)
    return SplitData(name, s1, pool, rows, cols, base, mq, md, stats, key, donor)


def _block(views: Views, s1: Table, pool: Table, cfg: Config, name: str):
    """Search: (rows, cols sorted by S1 then score, their cosines)."""
    rows, cols = candidate_pairs(views, s1.country, pool.country, cfg, name)
    with timed(f"{name}: cosines"):
        base = views.cosines(rows, cols)
    return _sorted(rows, cols, base)


def _sorted(rows: np.ndarray, cols: np.ndarray, base: dict[str, np.ndarray]):
    order = np.lexsort((-base["score"], rows))
    rows, cols = rows[order].astype(np.int32), cols[order].astype(np.int32)
    return rows, cols, {k: v[order] for k, v in base.items()}


def _with_embeddings(s1, pool, cfg: Config, name: str, rows, cols, base, views, drop=None):
    """Add each entity's best pairs by embedding (and each pool record's best suitor),
    re-score the new pairs on the token views, and give every pair its ``cos_emb``.
    Businesses in ``drop`` are treated as absent from Source 1: a pool record's best
    remaining suitor takes their place in the reverse lists."""
    from ber.embed import dense_row_dot, load_embeddings, load_knn

    sc = cfg.scale
    knn = load_knn(sc.emb_dir, name, s1.ids, pool.ids)
    fr, fc = knn["fwd_rows"].astype(np.int64), knn["fwd_cols"].astype(np.int64)
    rr, rc = knn["rev_rows"].astype(np.int64), knn["rev_cols"].astype(np.int64)
    fs, rs = knn["fwd_sim"].astype(np.float32), knn["rev_sim"].astype(np.float32)
    if drop is not None and drop.any():
        kf, kr = ~drop[fr], ~drop[rr]
        fr, fc, fs, rr, rc, rs = fr[kf], fc[kf], fs[kf], rr[kr], rc[kr], rs[kr]
    keep_f = group_stats(fr, fs)[0] < sc.k_emb
    keep_r = group_stats(rc, rs)[0] < sc.k_emb_reverse
    n_d = len(pool)
    cand = np.unique(np.concatenate([fr[keep_f] * n_d + fc[keep_f], rr[keep_r] * n_d + rc[keep_r]]))
    new = np.setdiff1d(cand, rows.astype(np.int64) * n_d + cols, assume_unique=False)
    log.info(
        "%s: embedding view adds %d pairs (%.2f per S1)", name, len(new), len(new) / max(len(s1), 1)
    )
    if len(new):
        if views is None:
            views, _, _ = _views(s1, pool, cfg, name)
        nr, nc = new // n_d, new % n_d
        with timed(f"{name}: cosines of embedding pairs"):
            cos = views.cosines(nr, nc)
        base = {k: np.concatenate([base[k], cos[k]]) for k in base}
        rows, cols, base = _sorted(np.concatenate([rows, nr]), np.concatenate([cols, nc]), base)
    with timed(f"{name}: embedding cosines"):
        e1, ep = load_embeddings(sc.emb_dir, name, s1.ids, pool.ids)
        base["cos_emb"] = dense_row_dot(e1, ep, rows, cols)
    return rows, cols, base


def _views(s1: Table, pool: Table, cfg: Config, name: str):
    """Hash and weight both sides: (views, S1 number matrix, pool number matrix)."""
    sc = cfg.scale
    with timed(f"{name}: hash tokens"):
        nq, nd = hash_docs(name_docs(s1), sc.n_jobs), hash_docs(name_docs(pool), sc.n_jobs)
        aq, ad = hash_docs(addr_docs(s1), sc.n_jobs), hash_docs(addr_docs(pool), sc.n_jobs)
        mq, md = hash_docs(num_docs(s1), sc.n_jobs), hash_docs(num_docs(pool), sc.n_jobs)
        release_workers()
    idf_n, _ = idf_weights(nq, nd)
    idf_a, _ = idf_weights(aq, ad)
    pool_df = np.concatenate(
        [
            np.bincount(nd.indices, minlength=nd.shape[1]),
            np.bincount(ad.indices, minlength=ad.shape[1]),
        ]
    )
    nq, nd = weight_and_normalize(nq, idf_n), weight_and_normalize(nd, idf_n)
    aq, ad = weight_and_normalize(aq, idf_a), weight_and_normalize(ad, idf_a)
    views = Views(nq, nd, aq, ad, pool_df[: nd.shape[1]], pool_df[nd.shape[1] :], cfg)
    return views, mq, md


@dataclass
class Views:
    """The name and address views of both sides, L2-normalised, and their combination."""

    nq: sp.csr_matrix
    nd: sp.csr_matrix
    aq: sp.csr_matrix
    ad: sp.csr_matrix
    pool_df_n: np.ndarray
    pool_df_a: np.ndarray
    cfg: Config

    @property
    def weights(self) -> tuple[float, float]:
        wn = self.cfg.scale.name_weight
        return wn, float(np.sqrt(max(0.0, 1.0 - wn * wn)))

    def combined(self, side: str) -> sp.csr_matrix:
        wn, wa = self.weights
        n, a = (self.nq, self.aq) if side == "q" else (self.nd, self.ad)
        return sp.hstack([wn * n, wa * a], format="csr")

    def cosines(self, rows: np.ndarray, cols: np.ndarray) -> dict[str, np.ndarray]:
        wn, wa = self.weights
        cn = row_dot(self.nq, self.nd, rows, cols)
        ca = row_dot(self.aq, self.ad, rows, cols)
        return {
            "score": (wn * wn * cn + wa * wa * ca).astype(np.float32),
            "cos_name": cn,
            "cos_addr": ca,
        }


def _filtered(m: sp.csr_matrix, df: np.ndarray, budget: int, cap: int) -> sp.csr_matrix:
    return prefix_filter(m, df, budget, cap) if budget > 0 else prune_columns(m, df <= cap)


def candidate_pairs(
    views: Views, q_country: np.ndarray, d_country: np.ndarray, cfg: Config, name: str
):
    """Unique (S1 position, pool position) candidate pairs.

    Forward: for each chunk of S1 queries, search name and address indexes on each
    query's rarest tokens, take top-m by name, by address and by their weighted sum,
    re-score every generated pair with the full cosines, and keep each query's best
    (``select``). Reverse: each pool record searches the S1 index the same way; a
    reverse pair is kept when it is among its pool record's best ``k_reverse``
    suitors by full combined cosine.
    """
    sc = cfg.scale
    part = cfg.blocking.partition_by_country
    wn, wa = views.weights
    n_q, n_d = views.nq.shape[0], views.nd.shape[0]
    with timed(f"{name}: forward index"):
        qn = _filtered(views.nq, views.pool_df_n, sc.query_budget, sc.df_cap)
        qa = _filtered(views.aq, views.pool_df_a, sc.query_budget, sc.df_cap)
        index = CountryIndex(views.nd, views.ad, d_country, part)
    fr, fc = [], []
    for a in range(0, n_q, sc.block_chunk):
        b = min(a + sc.block_chunk, n_q)
        with timed(f"{name}: forward search {a:,}-{b:,}"):
            r, c = index.search(qn[a:b], qa[a:b], q_country[a:b], sc.generate, sc.generate_name, sc.generate_addr,
                                wn * wn, wa * wa, sc.n_threads, sc.chunk_rows)  # fmt: skip
            r, c = _unique_pairs(r + a, c, n_d)
            keep = select(r, c, views.cosines(r, c), cfg)
            fr.append(r[keep])
            fc.append(c[keep])
    del index, qn, qa
    rows, cols = np.concatenate(fr), np.concatenate(fc)
    log.info("%s: forward kept %d pairs (%.1f per S1)", name, len(rows), len(rows) / max(n_q, 1))
    if sc.generate_reverse <= 0 or sc.k_reverse <= 0:
        return rows, cols

    with timed(f"{name}: reverse index"):
        s1_df_n = np.bincount(views.nq.indices, minlength=views.nq.shape[1])
        s1_df_a = np.bincount(views.aq.indices, minlength=views.aq.shape[1])
        dn = _filtered(views.nd, s1_df_n, max(1, sc.query_budget // 4), max(1, sc.df_cap // 4))
        da = _filtered(views.ad, s1_df_a, max(1, sc.query_budget // 4), max(1, sc.df_cap // 4))
        rindex = CountryIndex(views.nq, views.aq, q_country, part)
    rr, rc = [], []
    step = sc.block_chunk * 4
    for a in range(0, n_d, step):
        b = min(a + step, n_d)
        with timed(f"{name}: reverse search {a:,}-{b:,}"):
            c, r = rindex.search(dn[a:b], da[a:b], d_country[a:b], sc.generate_reverse, 0, 0,
                                 wn * wn, wa * wa, sc.n_threads, sc.chunk_rows * 4)  # fmt: skip
            rr.append(r)
            rc.append(c + a)
    del rindex, dn, da
    fwd_key = rows.astype(np.int64) * n_d + cols
    all_r, all_c = _unique_pairs(np.concatenate([rows, *rr]), np.concatenate([cols, *rc]), n_d)
    is_fwd = np.isin(all_r.astype(np.int64) * n_d + all_c, fwd_key, assume_unique=True)
    score = views.cosines(all_r, all_c)["score"]
    pool_rank = group_stats(all_c, score)[0]
    keep = is_fwd | (pool_rank < sc.k_reverse)
    log.info("%s: reverse added %d pairs", name, int(keep.sum() - is_fwd.sum()))
    return all_r[keep], all_c[keep]


def select(
    rows: np.ndarray, cols: np.ndarray, cos: dict[str, np.ndarray], cfg: Config
) -> np.ndarray:
    """Per S1 record: its best by combined, by name and by address cosine (full re-score)."""
    sc = cfg.scale
    keep = np.zeros(len(rows), dtype=bool)
    for key, k in (("score", sc.k), ("cos_name", sc.k_name), ("cos_addr", sc.k_addr)):
        if k > 0:
            keep |= group_stats(rows, cos[key])[0] < k
    return keep


def _unique_pairs(rows: np.ndarray, cols: np.ndarray, n_cols: int) -> tuple[np.ndarray, np.ndarray]:
    key = np.unique(rows.astype(np.int64) * n_cols + cols.astype(np.int64))
    return (key // n_cols).astype(np.int64), (key % n_cols).astype(np.int64)


def pair_features(sd: SplitData, idx: np.ndarray) -> pd.DataFrame:
    cols_out: dict[str, np.ndarray] = {k: v[idx] for k, v in sd.base.items()}
    cols_out.update(intrinsic_features(sd, idx))
    return pd.DataFrame(cols_out)


# Record-pair features in named groups. A cached matrix from an earlier FEATURE_VERSION
# supplies every column it has, and only the missing groups are computed, so adding a group
# costs that group alone. Changing what an existing column means therefore needs a new
# column name, not just a version bump.
FEATURE_GROUPS: dict[str, Any] = {
    "string": lambda sd, r, c: string_features(sd.s1, sd.pool, r, c),
    "number": lambda sd, r, c: number_features(sd.num_q, sd.num_d, r, c),
    "detail": lambda sd, r, c: detail_features(sd.s1, sd.pool, r, c),
    "collective": lambda sd, r, c: collective_features(sd.s1, sd.pool, r, c, _collective(sd)),
    "cluster": lambda sd, r, c: cluster_features(r, c, _collective(sd)),
}


def _collective(sd: SplitData) -> dict[str, Any]:
    if "collective" not in sd.extras:
        with timed(f"{sd.name}: collective statistics"):
            sd.extras["collective"] = collective_stats(sd.s1, sd.pool)
    return sd.extras["collective"]


def intrinsic_features(
    sd: SplitData, idx: np.ndarray, groups: list[str] | None = None
) -> dict[str, np.ndarray]:
    """Features that depend only on the two records of a pair (and split-wide statistics),
    not on the candidate set; ``groups`` limits which groups are computed."""
    rows, cols = sd.rows[idx], sd.cols[idx]
    out: dict[str, np.ndarray] = {}
    for name, fn in FEATURE_GROUPS.items():
        if groups is None or name in groups:
            out.update(fn(sd, rows, cols))
    return out


def _group_columns(sd: SplitData) -> dict[str, list[str]]:
    probe = np.arange(min(2, len(sd.rows)))
    return {g: list(intrinsic_features(sd, probe, [g])) for g in FEATURE_GROUPS}


def _feature_path(cfg: Config, split: str, key: str, version: int | None = None) -> Path:
    v = FEATURE_VERSION if version is None else version
    return Path(cfg.scale.cache_dir) / f"{split}_features_{key}_v{v}.npy"


def _load_features(path: Path):
    names_path = path.with_suffix(".json")
    if path.exists() and names_path.exists():
        return np.load(path, mmap_mode="r"), json.loads(names_path.read_text())
    return None


def _latest_features(cfg: Config, split: str, key: str, max_version: int):
    for v in range(max_version, 0, -1):
        if (loaded := _load_features(_feature_path(cfg, split, key, v))) is not None:
            return v, loaded
    return None


def _positions(sd: SplitData, d_rows: np.ndarray, d_cols: np.ndarray) -> np.ndarray:
    """For each pair of ``sd``, its position in the (d_rows, d_cols) list, or -1."""
    n_d = len(sd.pool)
    kd = d_rows.astype(np.int64) * n_d + d_cols
    order = np.argsort(kd, kind="stable")
    ks = kd[order]
    kn = sd.rows.astype(np.int64) * n_d + sd.cols
    at = np.minimum(np.searchsorted(ks, kn), len(ks) - 1)
    return np.where(ks[at] == kn, order[at], -1)


def _run_donor(sd: SplitData, cfg: Config):
    """A previous run's cached feature matrix for this split, located through the pair list
    that run saved ({split}_scores.npz, in feature-matrix order)."""
    run = Path(cfg.scale.feature_donor_run)
    scores = run / f"{sd.name}_scores.npz"
    if not scores.exists():
        return None
    with np.load(scores) as z:
        d_rows, d_cols = z["rows"], z["cols"]
    cache = Path(cfg.scale.cache_dir)
    for v in range(FEATURE_VERSION, 0, -1):
        for path in sorted(cache.glob(f"{sd.name}_features_*_v{v}.npy")):
            loaded = _load_features(path)
            if loaded is not None and loaded[0].shape[0] == len(d_rows):
                pos = _positions(sd, d_rows, d_cols)
                return loaded[0], loaded[1], pos, f"run {run.name}, v{v}"
    return None


def _reuse_source(sd: SplitData, cfg: Config):
    """(matrix, column names, row in it for each pair or -1, description) from, in order:
    a named earlier run (``scale.feature_donor_run``), the newest earlier-version matrix of
    this candidate set, or the token-view set it extends."""
    if cfg.scale.feature_donor_run and (src := _run_donor(sd, cfg)) is not None:
        return src
    if sd.key and (hit := _latest_features(cfg, sd.name, sd.key, FEATURE_VERSION - 1)):
        v, (feats, names) = hit
        if feats.shape[0] == len(sd.rows):
            return feats, names, np.arange(len(sd.rows)), f"same pairs, v{v}"
    if sd.donor is None:
        return None
    key, d_rows, d_cols = sd.donor
    hit = _latest_features(cfg, sd.name, key, FEATURE_VERSION)
    if hit is None or len(d_rows) == 0:
        return None
    v, (feats, names) = hit
    return feats, names, _positions(sd, d_rows, d_cols), f"token-view pairs, v{v}"


def _features_reused(sd: SplitData, idx: np.ndarray, src, group_cols) -> pd.DataFrame:
    feats, names, pos, _ = src
    have_col = set(names)
    p = pos[idx]
    have = p >= 0
    columns = [c for cs in group_cols.values() for c in cs]
    block = {c: np.full(len(idx), np.nan, np.float32) for c in columns}
    if have.any():
        raw = np.asarray(feats[p[have]])
        for c in columns:
            if c in have_col:
                block[c][have] = raw[:, names.index(c)]
        missing = [g for g, cs in group_cols.items() if any(c not in have_col for c in cs)]
        if missing:  # new groups, for the pairs the source covers
            for c, v in intrinsic_features(sd, idx[have], missing).items():
                block[c][have] = v
    if (~have).any():  # pairs the source does not cover: every group
        for c, v in intrinsic_features(sd, idx[~have]).items():
            block[c][~have] = v
    x = {k: v[idx] for k, v in sd.base.items()}
    x.update(block)
    return pd.DataFrame(x)


def feature_matrix(sd: SplitData, cfg: Config) -> tuple[np.ndarray, list[str]]:
    """Every candidate pair's stage-1 features as one float32 matrix, in pair order.

    With a split key it is a memory-mapped ``.npy`` in the cache directory; its column
    names are written last, so a file from an interrupted run is never mistaken for
    a finished one. Columns an earlier version of this candidate set (or the token-view
    set it extends) already computed are copied; only the rest is computed.
    """
    n = len(sd.rows)
    path = _feature_path(cfg, sd.name, sd.key) if sd.key else None
    names_path = path.with_suffix(".json") if path is not None else None
    if path is not None and (cached := _load_features(path)) is not None:
        log.info("%s: cached features %s", sd.name, path.name)
        return cached
    src = _reuse_source(sd, cfg) if n else None
    group_cols = _group_columns(sd) if src is not None else None
    if src is not None:
        log.info("%s: reusing cached features (%s) for %d/%d pairs",
                 sd.name, src[3], int((src[2] >= 0).sum()), n)  # fmt: skip
    out, names = None, []
    for idx in _chunks(n, cfg.scale.feature_chunk):
        with timed(f"{sd.name}: features {idx[0]:,}-{idx[-1]:,} of {n:,}"):
            x = (
                pair_features(sd, idx)
                if src is None
                else _features_reused(sd, idx, src, group_cols)
            )
            if out is None:
                names = list(x.columns)
                shape = (n, len(names))
                if path is not None:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    out = np.lib.format.open_memmap(path, "w+", np.float32, shape)
                else:
                    out = np.empty(shape, np.float32)
            out[idx[0] : idx[-1] + 1] = x[names].to_numpy(np.float32)
    if out is None:
        names = list(pair_features(sd, np.arange(0)).columns)
        out = np.empty((0, len(names)), np.float32)
    if path is not None and n:
        out.flush()
        names_path.write_text(json.dumps(names))
    return out, names


# ----------------------------------------------------------------------------- labels
def pair_labels(sd: SplitData, truth: dict[str, list[str]]) -> tuple[np.ndarray, np.ndarray]:
    """(label per candidate pair, number of true matches per S1 position)."""
    pool_index = pd.Index(sd.pool.ids)
    s1_index = pd.Index(sd.s1.ids)
    owner = np.full(len(sd.pool), -1, dtype=np.int64)
    n_true = np.zeros(len(sd.s1), dtype=np.float64)
    s1_list, pool_list = [], []
    for sid, matches in truth.items():
        s1_list.extend([sid] * len(matches))
        pool_list.extend(matches)
    s1_pos = s1_index.get_indexer(s1_list)
    pool_pos = pool_index.get_indexer(pool_list)
    ok = (s1_pos >= 0) & (pool_pos >= 0)
    owner[pool_pos[ok]] = s1_pos[ok]
    np.add.at(n_true, s1_index.get_indexer(list(truth)), [len(v) for v in truth.values()])
    labels = owner[sd.cols] == sd.rows
    return labels, n_true


# ----------------------------------------------------------------------------- models
def _lgb_params(cfg: Config, seed: int) -> dict[str, Any]:
    return {
        **cfg.model.params,
        "seed": seed,
        "num_threads": cfg.scale.n_threads + 4,
        "deterministic": True,
        "force_row_wise": True,
    }


def reuse_folds(path: str | Path, x: pd.DataFrame, groups: np.ndarray, cfg: Config):
    """Stage 1 from an earlier run's saved model instead of retraining: the same grouped
    folds are rebuilt and each fold model predicts its own held-out part, which gives the
    out-of-fold probabilities that training would have. Refused unless the saved model
    used exactly these features and folds."""
    saved = joblib.load(path)
    if list(saved.feature_names) != list(x.columns) or len(saved.stage1) != cfg.model.n_folds:
        raise ValueError(f"{path}: stage 1 was trained on other features or folds")
    folds = group_folds(groups, cfg.model.n_folds, cfg.seed)
    oof = np.zeros(len(x), dtype=np.float32)
    for k, booster in enumerate(saved.stage1):
        va = folds == k
        with timed(f"stage1 fold {k + 1}: reuse and predict held-out part"):
            oof[va] = booster.predict(x[va], num_iteration=booster.best_iteration)
    log.info("reused stage 1 from %s", path)
    return oof, list(saved.stage1), [int(b.best_iteration) for b in saved.stage1]


def fit_folds(
    x: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    cfg: Config,
    tag: str,
    monotone: list[int] | None = None,
):
    """Grouped K-fold LightGBM. Returns (out-of-fold probabilities, fold models, best iterations)."""
    import lightgbm as lgb

    folds = group_folds(groups, cfg.model.n_folds, cfg.seed)
    oof = np.zeros(len(x), dtype=np.float32)
    models, iters = [], []
    extra = {"monotone_constraints": monotone} if monotone and any(monotone) else {}
    for k in range(cfg.model.n_folds):
        tr, va = folds != k, folds == k
        with timed(f"{tag} fold {k + 1}/{cfg.model.n_folds}"):
            dtr = lgb.Dataset(x[tr], label=y[tr], free_raw_data=True)
            dva = lgb.Dataset(x[va], label=y[va], reference=dtr, free_raw_data=True)
            booster = lgb.train(
                {**_lgb_params(cfg, cfg.seed + k), **extra},
                dtr,
                num_boost_round=cfg.model.num_boost_round,
                valid_sets=[dva],
                callbacks=[
                    lgb.early_stopping(cfg.model.early_stopping_rounds, verbose=False),
                    lgb.log_evaluation(0),
                ],
            )
            oof[va] = booster.predict(x[va], num_iteration=booster.best_iteration)
        models.append(booster)
        iters.append(int(booster.best_iteration))
        log.info("%s fold %d: best iteration %d", tag, k + 1, booster.best_iteration)
    return oof, models, iters


def predict_avg(models: list, x: pd.DataFrame) -> np.ndarray:
    out = np.zeros(len(x), dtype=np.float64)
    for m in models:
        out += m.predict(x, num_iteration=m.best_iteration)
    return (out / len(models)).astype(np.float32)


def stage2_columns(
    rows: np.ndarray, cols: np.ndarray, p1: np.ndarray, keep: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Stage-2 inputs: stage-1 probability in the context of both groups, plus kept stage-1 columns."""
    f: dict[str, np.ndarray] = {"p1": p1}
    for side, keys in (("s1", rows), ("pool", cols)):
        rank, best, second, size = group_stats(keys, p1)
        f[f"p1_rank_{side}"] = rank
        f[f"p1_gap_{side}"] = (p1 - best).astype(np.float32)
        f[f"p1_margin_{side}"] = np.where(rank == 0, p1 - second, p1 - best).astype(np.float32)
        sums = np.bincount(keys, weights=p1, minlength=int(keys.max()) + 1 if len(keys) else 0)
        f[f"p1_rest_{side}"] = (sums[keys] - p1).astype(np.float32)
        f[f"p1_n_{side}"] = size
        del rank, best, second, sums
    f.update(keep)
    return f


def cross_scores(
    sd: SplitData,
    cfg: Config,
    p1: np.ndarray,
    data_dir: str | Path,
    work_dir: Path,
    only: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Cross-encoder probabilities for every pair with stage-1 p1 >= ``ce_min_p1`` (NaN for
    the rest), run in the torch interpreter, one array per model in ``scale.ce_model``
    (comma-separated): "ce" for the first, then "ce2", "ce3", ... so stage 2 learns how to
    weigh several cross-encoders. ``only``: boolean mask of S1 positions to score."""
    import os
    import shlex
    import subprocess

    sc = cfg.scale
    models = [m.strip() for m in sc.ce_model.split(",") if m.strip()]
    if not models:
        return {}
    sel = p1 >= sc.ce_min_p1
    if only is not None:
        sel &= only[sd.rows]
    sel = np.flatnonzero(sel)
    pairs = work_dir / f"{sd.name}_ce_pairs.npz"
    # p1 orders a listwise model's groups; the pairwise scorer ignores it
    np.savez(pairs, rows=sd.rows[sel], cols=sd.cols[sel], p1=p1[sel].astype(np.float32))
    src = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(src)}
    out: dict[str, np.ndarray] = {}
    band = [float(x) for x in sc.ce_band.split(",")] if sc.ce_band else None
    seen = {"pairwise": 0, "listwise": 0, "listwise_pool": 0}
    for model in models:
        done = Path(model) / "crossenc.json"  # written last by ber.crossenc / ber.listwise train
        waited = 0
        while not done.exists() and waited < sc.ce_wait:  # the model may still be training
            if waited % 300 == 0:
                log.info("waiting for the cross-encoder at %s (%ds so far)", model, waited)
            time.sleep(30)
            waited += 30
        kind = scorer_kind(model)
        seen[kind] += 1
        stem = {"listwise": "lw", "listwise_pool": "lwp"}.get(kind, "ce")
        prefix = stem if seen[kind] == 1 else f"{stem}{seen[kind]}"
        module = "ber.crossenc" if kind == "pairwise" else "ber.listwise"
        scores = work_dir / f"{sd.name}_{prefix}_scores.npy"
        todo, todo_pairs = sel, pairs
        if kind == "pairwise" and prefix != "ce" and band is not None and "ce" in out:
            # a slower model scores only the pairs the first one is unsure of; elsewhere its
            # column repeats the first model's confident score (same rule in train and test).
            # A listwise model reads whole groups, so it always scores every pair.
            first = out["ce"][sel]
            todo = sel[(first > band[0]) & (first < band[1])]
            todo_pairs = work_dir / f"{sd.name}_{prefix}_pairs.npz"
            np.savez(todo_pairs, rows=sd.rows[todo], cols=sd.cols[todo],
                     p1=p1[todo].astype(np.float32))  # fmt: skip
        group = scores.with_name(f"{scores.stem}_grp.npy")  # listwise: P(the entity has a match)
        # the same model on the same pairs (and, for a listwise model, the same p1 that
        # groups them) gives the same scores: a later run reuses them instead of rescoring
        key = _score_key(model, sd.name, sd.rows[todo], sd.cols[todo],
                         p1[todo] if kind != "pairwise" else None)  # fmt: skip
        cached = Path(sc.cache_dir) / "scores" / f"{sd.name}_{key}.npy"
        cached_group = cached.with_name(f"{cached.stem}_grp.npy")
        if cached.exists():
            log.info("%s: reusing cached %s scores of %s (%s)", sd.name, kind, model, cached.name)
            shutil.copyfile(cached, scores)
            if cached_group.exists():
                shutil.copyfile(cached_group, group)
        else:
            cmd = [*shlex.split(sc.ce_python), "-m", module, "score", "--model", model,
                   "--data-dir", str(data_dir), "--split", sd.name, "--pairs", str(todo_pairs), "--out", str(scores)]  # fmt: skip
            with timed(f"{sd.name}: {kind} scorer {model} on {len(todo):,} pairs"):
                subprocess.run(cmd, check=True, env=env)
            cached.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(scores, cached.with_suffix(".tmp"))
            if group.exists():
                shutil.copyfile(group, cached_group)
            cached.with_suffix(".tmp").replace(cached)  # last: marks the entry complete
        ce = np.full(len(sd.rows), np.nan, dtype=np.float32)
        if len(todo) < len(sel):
            ce[sel] = out["ce"][sel]
        ce[todo] = np.load(scores).astype(np.float32)
        out[prefix] = ce
        if group.exists():
            g = np.full(len(sd.rows), np.nan, dtype=np.float32)
            g[todo] = np.load(group).astype(np.float32)
            out[f"{prefix}_grp"] = g
    return out


def _score_key(model: str, split: str, rows, cols, p1=None) -> str:
    """Identify a scorer's output: the model (its path and the crossenc.json its training
    wrote last, which changes on every retraining) and the exact pairs it scored."""
    done = Path(model) / "crossenc.json"
    h = hashlib.sha1(f"{Path(model).resolve()}|{split}".encode())
    h.update(done.read_bytes() if done.exists() else b"")
    for a in (rows, cols, p1):
        if a is not None:
            h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:16]


def stage2_monotone(names: list[str]) -> list[int]:
    """LightGBM monotone constraints: a higher probability from stage 1 or any cross-encoder
    or listwise model may only raise the stage-2 probability. Keeps the stack from learning
    interactions that flip under the train-to-test shift."""
    return [1 if n == "p1" or re.fullmatch(r"(?:ce|lwp|lw)\d*", n) else 0 for n in names]


def scorer_kind(model: str | Path) -> str:
    """The scorer a model directory needs: "listwise" (ber.listwise, Source 1 side),
    "listwise_pool" (ber.listwise, pool side: C7) or "pairwise" (ber.crossenc)."""
    done = Path(model) / "crossenc.json"
    if not done.exists():
        return "pairwise"
    try:
        info = json.loads(done.read_text())
    except json.JSONDecodeError:
        return "pairwise"
    kind = str(info.get("kind", "pairwise"))
    return "listwise_pool" if kind == "listwise" and info.get("side") == "pool" else kind


def scorer_columns(
    rows: np.ndarray, cols: np.ndarray, arrays: dict[str, np.ndarray], pool_for: set[str]
) -> dict[str, np.ndarray]:
    """Stage-2 columns from ``cross_scores``: each score in context of the entity's
    candidates, and of the pool record's suitors for the prefixes in ``pool_for``; a listwise
    model also gets the rest of the entity's mass (its expected number of other matches)
    and its group probability as is."""
    out: dict[str, np.ndarray] = {}
    for prefix, v in arrays.items():
        if prefix.endswith("_grp"):
            out[prefix] = v
            continue
        out.update(ce_columns(rows, v, prefix, cols if prefix in pool_for else None))
        if prefix.startswith("lw") and len(rows):
            scored = ~np.isnan(v)
            mass = np.where(scored, v, 0.0).astype(np.float64)
            sums = np.bincount(rows, weights=mass, minlength=int(rows.max()) + 1)
            out[f"{prefix}_rest_s1"] = np.where(scored, sums[rows] - mass, np.nan).astype(
                np.float32
            )
    return out


_SCORER = re.compile(r"^((?:ce|lwp|lw)\d*)(?:_(.*))?$")


def stage2_family(name: str) -> str:
    """The family of a stage-2 column, the unit ``scale.stage2_exclude`` and
    ``scripts/stage2_search.py`` switch on and off: "p1" (stage-1 probability in context),
    "coh" (coherence), a scorer prefix ("ce", "ce2", "lw", ...) with "_pool" for its
    pool-side context and "_grp" for a listwise group probability, else "keep" (the
    stage-1 similarities carried over)."""
    if name == "p1" or name.startswith("p1_"):
        return "p1"
    if name in ("coh_emb", "partner_p1"):
        return "coh"
    m = _SCORER.match(name)
    if m:
        prefix, rest = m.group(1), m.group(2) or ""
        if rest.endswith("_pool"):
            return f"{prefix}_pool"
        if rest == "grp":
            return f"{prefix}_grp"
        return prefix
    return "keep"


def _stage2_names(columns: dict[str, np.ndarray], cfg: Config) -> list[str]:
    excluded = {s.strip() for s in cfg.scale.stage2_exclude.split(",") if s.strip()}
    return [n for n in columns if stage2_family(n) not in excluded]


def _uses_scorers(names: list[str]) -> bool:
    return any(_SCORER.match(n) for n in names)


def coherence_columns(
    rows: np.ndarray, cols: np.ndarray, p1: np.ndarray, emb_pool: np.ndarray
) -> dict[str, np.ndarray]:
    """How much a candidate resembles its entity's most probable other candidate.

    An entity's true matches are variants of one business, so a candidate close
    to a confident match is likelier to be one too. Partner: the entity's best
    pair by stage-1 probability, or its second best for the best pair itself.
    """
    from ber.embed import dense_row_dot

    n = len(rows)
    order = np.lexsort((-p1, rows))
    r = rows[order]
    new = np.ones(n, dtype=bool)
    new[1:] = r[1:] != r[:-1]
    starts = np.flatnonzero(new)
    sizes = np.diff(np.append(starts, n))
    first = order[starts]
    second = np.where(sizes > 1, order[np.minimum(starts + 1, n - 1)], -1)
    gid = np.empty(n, dtype=np.int64)
    gid[order] = np.cumsum(new) - 1
    partner = np.where(np.arange(n) == first[gid], second[gid], first[gid])
    has = np.flatnonzero(partner >= 0)
    coh = np.full(n, np.nan, dtype=np.float32)
    coh[has] = dense_row_dot(emb_pool, emb_pool, cols[has], cols[partner[has]])
    partner_p1 = np.full(n, np.nan, dtype=np.float32)
    partner_p1[has] = p1[partner[has]]
    return {"coh_emb": coh, "partner_p1": partner_p1}


def _pool_embeddings(sd: SplitData, cfg: Config) -> np.ndarray:
    from ber.embed import load_embeddings

    return load_embeddings(cfg.scale.emb_dir, sd.name, sd.s1.ids, sd.pool.ids)[1]


def ce_columns(
    rows: np.ndarray, ce: np.ndarray, prefix: str = "ce", cols: np.ndarray | None = None
) -> dict[str, np.ndarray]:
    """The cross-encoder probability and how it ranks among the entity's scored candidates
    and, with ``cols``, among the pool record's scored suitors: whether another business got
    a higher cross-encoder score for the same record decides who owns it (44% of v3's missed
    true pairs lost their record to another suitor)."""
    v = np.nan_to_num(ce, nan=-1.0)
    scored = ~np.isnan(ce)
    out = {prefix: ce}
    sides = [("s1", rows)] + ([("pool", cols)] if cols is not None else [])
    for side, keys in sides:
        rank, best, second, _ = group_stats(keys, v)
        out[f"{prefix}_rank_{side}"] = np.where(scored, rank, np.nan).astype(np.float32)
        out[f"{prefix}_gap_{side}"] = np.where(scored, v - best, np.nan).astype(np.float32)
        out[f"{prefix}_margin_{side}"] = np.where(
            scored, np.where(rank == 0, v - second, v - best), np.nan
        ).astype(np.float32)
    return out


def _frame(columns: dict[str, np.ndarray], idx: np.ndarray, names: list[str]) -> pd.DataFrame:
    return pd.DataFrame({n: columns[n][idx] for n in names})


def predict_columns(
    models: list, columns: dict[str, np.ndarray], names: list[str], n: int, chunk: int
) -> np.ndarray:
    out = np.empty(n, dtype=np.float32)
    for idx in _chunks(n, chunk):
        out[idx] = predict_avg(models, _frame(columns, idx, names))
    return out


def exclusive_adjust(rows: np.ndarray, cols: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Zero every pair whose pool record has a better S1 suitor (ties: lower S1 position)."""
    order = np.lexsort((rows, -p, cols))
    first = np.ones(len(order), dtype=bool)
    first[1:] = cols[order][1:] != cols[order][:-1]
    keep = np.empty(len(order), dtype=bool)
    keep[order] = first
    return np.where(keep, p, 0.0).astype(np.float32)


def id_lists(
    rows: np.ndarray, cols: np.ndarray, key: np.ndarray, s1_ids: np.ndarray, pool_ids: np.ndarray
) -> dict[str, list[str]]:
    """{S1 id: [pool ids by descending key]} for every S1 id, built with one sort."""
    order = np.lexsort((-key, rows))
    r = rows[order]
    ids = pool_ids[cols[order]]
    bounds = np.searchsorted(r, np.arange(len(s1_ids) + 1))
    return {s1_ids[i]: ids[bounds[i] : bounds[i + 1]].tolist() for i in range(len(s1_ids))}


# ----------------------------------------------------------------------------- orchestration
@dataclass
class TrainedModel:
    stage1: list
    stage2: list
    feature_names: list[str]
    stage2_names: list[str]
    threshold: float
    config: dict[str, Any]
    version: str = __version__
    decision: dict[str, Any] | None = None  # tuned rule (ber.tuning); None: plain threshold
    train_prior: float | None = None  # mean final probability over eligible train pairs

    def rule(self):
        from ber.tuning import Rule

        if self.decision is None:
            return Rule("threshold", {"t": self.threshold})
        return Rule(self.decision["strategy"], self.decision["params"])


def decide(rows: np.ndarray, adj: np.ndarray, rule, beta: float) -> np.ndarray:
    from ber.tuning import Grouped, select

    return select(Grouped.build(rows, adj), rule, beta)


def _chunks(n: int, size: int):
    for s in range(0, n, size):
        yield np.arange(s, min(s + size, n))


def score_all(
    sd: SplitData,
    cfg: Config,
    stage1: list,
    feature_names: list[str],
    feats: np.ndarray,
    feats_names: list[str],
    override: dict | None = None,
):
    """Stage-1 probabilities for every pair (chunked), plus the columns stage 2 keeps."""
    n = len(sd.rows)
    pos = [feats_names.index(c) for c in feature_names]
    p1 = np.empty(n, dtype=np.float32)
    kept = [k for k in STAGE2_KEEP if k in feats_names]
    keep = {k: np.empty(n, dtype=np.float32) for k in kept}
    for idx in _chunks(n, cfg.scale.feature_chunk):
        s, e = idx[0], idx[-1] + 1
        with timed(f"{sd.name}: predict {s:,}-{e:,}"):
            block = np.asarray(feats[s:e])
            p1[s:e] = predict_avg(stage1, pd.DataFrame(block[:, pos], columns=feature_names))
            for k in kept:
                keep[k][s:e] = block[:, feats_names.index(k)]
    if override is not None:
        p1[override["idx"]] = override["p1"]
    return p1, keep


def adjusted(sd: SplitData, probs: np.ndarray, cfg: Config) -> np.ndarray:
    return exclusive_adjust(sd.rows, sd.cols, probs) if cfg.decision.exclusive else probs


def subset_fbeta(
    sd: SplitData,
    adj: np.ndarray,
    labels: np.ndarray,
    n_true: np.ndarray,
    threshold: float,
    subset: np.ndarray,
    beta: float,
) -> float:
    """Leaderboard macro F-beta over the S1 positions in ``subset`` (every competitor still decided)."""
    in_sub = np.zeros(len(sd.s1), dtype=bool)
    in_sub[subset] = True
    sel = in_sub[sd.rows]
    mask = (adj[sel] >= threshold) & (adj[sel] > 0)
    return macro_from_mask(
        np.searchsorted(subset, sd.rows[sel]), labels[sel], mask, n_true[subset], beta
    )


def tune_threshold(
    sd: SplitData,
    adj: np.ndarray,
    labels: np.ndarray,
    n_true: np.ndarray,
    subset: np.ndarray,
    cfg: Config,
) -> tuple[float, float]:
    grid = np.round(
        np.arange(cfg.decision.grid_min, cfg.decision.grid_max + 1e-9, cfg.decision.grid_step), 6
    )
    scores = [subset_fbeta(sd, adj, labels, n_true, float(t), subset, cfg.beta) for t in grid]
    best = int(np.argmax(scores))
    log.info(
        "threshold %.3f gives macro F%.1f %.5f on %d entities",
        grid[best],
        cfg.beta,
        scores[best],
        len(subset),
    )
    return float(grid[best]), float(scores[best])


def run(
    cfg: Config,
    data_dir: str | Path,
    out_dir: str | Path,
    tag: str = "run",
    predict_test: bool = True,
) -> dict[str, Any]:
    seed_everything(cfg.seed)
    t0 = time.time()
    run_dir = Path(cfg.runs_dir) / f"{time.strftime('%Y%m%d-%H%M%S')}-{tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    dump_config(cfg, run_dir / "config.yaml")
    summary: dict[str, Any] = {"version": __version__, "git": git_revision()}

    # ---- train: block every S1 entity against the whole pool
    s1, pool = load_tables(data_dir, "train", cfg)
    truth = load_split_truth(data_dir)
    eligible, drop = train_entities(s1.ids, cfg)
    if drop is not None:
        summary["dropped_s1"] = int(drop.sum())
        log.info("train: %d S1 businesses removed to simulate orphaned copies", int(drop.sum()))
    sd = block_split(s1, pool, cfg, "train", split_key(data_dir, "train", cfg), drop)
    labels, n_true = pair_labels(sd, truth)
    summary["train_blocking"] = {
        **sd.stats,
        "eligible_entities": int(eligible.sum()),
        **blocking_recall(sd, labels, n_true, cfg.beta, eligible),
    }
    log.info("train blocking: %s", summary["train_blocking"])
    _write_json(run_dir / "metrics.json", summary)

    # ---- stage 1 on a sample of entities; every other eligible entity is an untouched holdout
    # (minus the listwise share, which trains ber.listwise)
    sample, in_sample, lw, holdout = split_entities(s1.ids, eligible, cfg)
    if lw.any():
        summary["listwise_share"] = int(lw.sum())
    sidx = np.flatnonzero(in_sample[sd.rows])
    feats, feature_names = feature_matrix(sd, cfg)
    with timed("train: sample features"):
        xs = pd.DataFrame(np.asarray(feats[sidx]), columns=feature_names)
    ys = labels[sidx].astype(np.int8)
    if cfg.scale.stage1_model:  # reuse a previous run's stage 1 (same features and sample)
        oof1, stage1, iters1 = reuse_folds(cfg.scale.stage1_model, xs, sd.rows[sidx], cfg)
    else:
        oof1, stage1, iters1 = fit_folds(xs, ys, sd.rows[sidx], cfg, "stage1")
    del xs
    _feature_importance(stage1, feature_names).to_csv(
        run_dir / "feature_importance.tsv", sep="\t", index=False
    )
    p1, keep = score_all(
        sd, cfg, stage1, feature_names, feats, feature_names, override={"idx": sidx, "p1": oof1}
    )
    del feats
    sd.base = {"score": sd.base["score"]}  # group context no longer needed; free it
    summary["stage1"] = _stage_report(sd, p1, labels, n_true, sample, holdout, cfg, iters1)
    _write_json(run_dir / "metrics.json", summary)
    if lw.any():  # training groups for ber.listwise (and its validation on holdout entities)
        dump_listwise_groups(run_dir / "lw_groups_train.npz", sd, p1, labels, n_true, lw,
                             holdout, cfg)  # fmt: skip
    final_p, stage2, s2_names, thr = p1, [], [], summary["stage1"]["threshold"]

    # ---- cross-encoders on uncertain pairs of entities they were not trained on
    # scoring every suitor (not only eligible entities') makes pool-side competition in
    # training look like test, where every suitor is scored
    ce_only = None if cfg.scale.ce_train_all else eligible
    scored = cross_scores(sd, cfg, p1, data_dir, run_dir, only=ce_only)
    for prefix, ce in scored.items():
        if not prefix.endswith("_grp"):
            summary.setdefault("scored_pairs", {})[prefix] = int((~np.isnan(ce)).sum())
    if scored:
        summary["cross_encoder_pairs"] = int((~np.isnan(next(iter(scored.values())))).sum())
    pool_for = set(scored) if cfg.scale.ce_pool_context else set()
    keep.update(scorer_columns(sd.rows, sd.cols, scored, pool_for))
    del scored
    if cfg.scale.emb_dir:
        with timed("train: coherence"):
            keep.update(coherence_columns(sd.rows, sd.cols, p1, _pool_embeddings(sd, cfg)))

    # ---- stage 2: stage-1 probabilities in context
    if cfg.scale.stage2:
        cols2 = stage2_columns(sd.rows, sd.cols, p1, keep)
        del keep
        if cfg.scale.save_stage2:
            save_stage2_inputs(run_dir, sd, cols2, p1, labels, n_true, eligible, in_sample, lw,
                               holdout, cfg)  # fmt: skip
        s2_names = _stage2_names(cols2, cfg)
        oof2, stage2, iters2 = fit_folds(
            _frame(cols2, sidx, s2_names), ys, sd.rows[sidx], cfg, "stage2",
            monotone=stage2_monotone(s2_names) if cfg.scale.stage2_monotone else None,
        )  # fmt: skip
        p2 = predict_columns(stage2, cols2, s2_names, len(sd.rows), cfg.scale.feature_chunk)
        p2[sidx] = oof2
        del cols2
        _feature_importance(stage2, s2_names).to_csv(
            run_dir / "feature_importance_stage2.tsv", sep="\t", index=False
        )
        summary["stage2"] = _stage_report(sd, p2, labels, n_true, sample, holdout, cfg, iters2)
        if summary["stage2"]["sample_oof"] >= summary["stage1"]["sample_oof"]:
            final_p, thr = p2, summary["stage2"]["threshold"]
        else:  # stacking did not help on the sample: ship stage 1 alone
            stage2, s2_names = [], []
        _write_json(run_dir / "metrics.json", summary)

    # ---- decision rule: tuned on the sample's out-of-fold probabilities, reported on holdout
    from ber.tuning import tune

    with timed("train: decision rules"):
        rule, summary["decision"] = tune(
            sd.rows, adjusted(sd, final_p, cfg), labels, n_true, sample, holdout, cfg.beta
        )
    _write_json(run_dir / "metrics.json", summary)
    _report_train(sd, final_p, rule, truth, cfg, run_dir, summary, eligible & ~lw)
    # every pair's final probability, for tuning decision rules offline (``eligible`` leaves
    # out the listwise share, whose listwise scores are in-sample)
    np.savez(
        run_dir / "train_scores.npz", rows=sd.rows, cols=sd.cols, p=final_p, labels=labels,
        n_true=n_true, eligible=eligible & ~lw, in_sample=in_sample, lw_share=lw,
        country=s1.country.astype("U"),
    )  # fmt: skip
    model = TrainedModel(
        stage1, stage2, feature_names, s2_names, thr, cfg.to_dict(),
        decision={"strategy": rule.strategy, "params": rule.params},
        train_prior=float(final_p[eligible[sd.rows]].mean()),
    )  # fmt: skip
    Path(cfg.artifacts_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(model, Path(cfg.artifacts_dir) / "scale_model.joblib")
    joblib.dump(model, run_dir / "scale_model.joblib")
    _write_json(run_dir / "metrics.json", summary)
    del sd, s1, pool, p1, final_p

    if predict_test:
        summary["test"] = predict_split(model, cfg, data_dir, out_dir, save_to=run_dir)
    summary["minutes"] = round((time.time() - t0) / 60, 1)
    _write_json(run_dir / "metrics.json", summary)
    log.info("run recorded at %s (%.1f min)", run_dir, summary["minutes"])
    return summary


def train_entities(s1_ids: np.ndarray, cfg: Config) -> tuple[np.ndarray, np.ndarray | None]:
    """(eligible, drop) over train Source 1 positions. Eligible: the entities the matcher may
    learn from and be judged on, i.e. all except those the embedding encoder was fine-tuned
    on (their embedding similarity is optimistic) and those removed to simulate orphans."""
    eligible = np.ones(len(s1_ids), dtype=bool)
    if cfg.scale.emb_dir:
        from ber.embed import encoder_mask

        eligible &= ~encoder_mask(s1_ids)
    drop = _drop_mask(len(s1_ids), eligible, cfg)
    if drop is not None:
        eligible &= ~drop
    return eligible, drop


def listwise_mask(s1_ids: np.ndarray, candidates: np.ndarray, share: float) -> np.ndarray:
    """The listwise share: ``share`` of the boolean ``candidates`` (eligible entities outside
    the stage-1/2 sample), picked by a fixed hash of the id, so every run agrees on it."""
    out = np.zeros(len(s1_ids), dtype=bool)
    if share <= 0:
        return out
    cut = int(share * 1_000_000)
    idx = np.flatnonzero(candidates)
    out[idx] = np.fromiter(
        (zlib.crc32(f"lw|{s1_ids[i]}".encode()) % 1_000_000 < cut for i in idx), bool, len(idx)
    )
    return out


def split_entities(s1_ids: np.ndarray, eligible: np.ndarray, cfg: Config):
    """(sample, in_sample, listwise share, holdout): the stage-1/2 sample is drawn from the
    eligible entities, the listwise share (``scale.lw_share``) from the rest, and the holdout
    is what remains, so no model's score on a holdout entity is in-sample."""
    rng = np.random.default_rng(cfg.seed)
    pick = np.flatnonzero(eligible)
    sample = np.sort(rng.choice(pick, min(cfg.scale.train_sample, len(pick)), replace=False))
    in_sample = np.zeros(len(s1_ids), dtype=bool)
    in_sample[sample] = True
    lw = listwise_mask(s1_ids, eligible & ~in_sample, cfg.scale.lw_share)
    holdout = np.flatnonzero(eligible & ~in_sample & ~lw)
    return sample, in_sample, lw, holdout


def dump_listwise_groups(
    path: Path,
    sd: SplitData,
    p1: np.ndarray,
    labels: np.ndarray,
    n_true: np.ndarray,
    lw: np.ndarray,
    holdout: np.ndarray,
    cfg: Config,
) -> None:
    """The pairs ``ber.listwise train`` learns from (listwise share) and validates on
    (holdout), with stage-1 probability >= ``scale.lw_dump_min_p1``: rows, cols, p1,
    labels, every entity's true-match count, and the two entity masks."""
    valid = np.zeros(len(sd.s1), dtype=bool)
    valid[holdout] = True
    sel = (lw | valid)[sd.rows] & (p1 >= cfg.scale.lw_dump_min_p1)
    with timed(f"write {path.name} ({int(sel.sum()):,} pairs)"):
        np.savez(path, rows=sd.rows[sel], cols=sd.cols[sel], p1=p1[sel].astype(np.float32),
                 labels=labels[sel], n_true=n_true, train=lw, valid=valid)  # fmt: skip


def save_stage2_inputs(
    run_dir: Path,
    sd: SplitData,
    cols2: dict[str, np.ndarray],
    p1: np.ndarray,
    labels: np.ndarray,
    n_true: np.ndarray,
    eligible: np.ndarray,
    in_sample: np.ndarray,
    lw: np.ndarray,
    holdout: np.ndarray,
    cfg: Config,
    min_p1: float = 0.001,
) -> None:
    """Every stage-2 column of the pairs with p1 >= ``min_p1`` (the others are ~0 after
    stage 2), for ``scripts/stage2_search.py``: stage2_inputs.npy (memory-mapped float32)
    with its column names in stage2_inputs.json, and stage2_meta.npz with the pairs, labels
    and entity masks."""
    idx = np.flatnonzero(p1 >= min_p1)
    names = list(cols2)
    path = run_dir / "stage2_inputs.npy"
    with timed(f"write stage-2 inputs ({len(idx):,} pairs x {len(names)} columns)"):
        mat = np.lib.format.open_memmap(path, "w+", np.float32, (len(idx), len(names)))
        for s in range(0, len(idx), 1_000_000):
            part = idx[s : s + 1_000_000]
            mat[s : s + len(part)] = np.column_stack([cols2[n][part] for n in names])
        mat.flush()
        del mat
        hold = np.zeros(len(sd.s1), dtype=bool)
        hold[holdout] = True
        np.savez(run_dir / "stage2_meta.npz", idx=idx, rows=sd.rows[idx], cols=sd.cols[idx],
                 labels=labels[idx], n_true=n_true, eligible=eligible, in_sample=in_sample,
                 lw_share=lw, holdout=hold, n_folds=cfg.model.n_folds, seed=cfg.seed)  # fmt: skip
        (run_dir / "stage2_inputs.json").write_text(json.dumps(names))


def _drop_mask(n: int, eligible: np.ndarray, cfg: Config) -> np.ndarray | None:
    """S1 positions removed from train to reproduce test's share of orphaned pool copies
    (test has ~5.75 pool records per S1 against 4.68 in train). Taken from the entities the
    matcher never trains on or is judged on when there are enough of them."""
    share = cfg.scale.drop_s1_share
    if share <= 0:
        return None
    k = round(share * n)
    candidates = np.flatnonzero(~eligible) if (~eligible).sum() >= k else np.arange(n)
    rng = np.random.default_rng(cfg.scale.drop_seed)
    drop = np.zeros(n, dtype=bool)
    drop[rng.choice(candidates, min(k, len(candidates)), replace=False)] = True
    return drop


def _stage_report(sd, probs, labels, n_true, sample, holdout, cfg, iters) -> dict[str, Any]:
    adj = adjusted(sd, probs, cfg)
    thr, f_sample = tune_threshold(sd, adj, labels, n_true, sample, cfg)
    out = {"threshold": thr, "sample_oof": f_sample, "best_iterations": iters}
    if len(holdout):
        out["holdout"] = subset_fbeta(sd, adj, labels, n_true, thr, holdout, cfg.beta)
    log.info("stage report: %s", out)
    return out


def blocking_recall(
    sd: SplitData,
    labels: np.ndarray,
    n_true: np.ndarray,
    beta: float,
    subset: np.ndarray | None = None,
) -> dict[str, float]:
    """Recall of the candidate set and the F-beta a perfect matcher would reach on it,
    over the S1 entities in the boolean ``subset`` (all when None)."""
    if subset is None:
        subset = np.ones(len(sd.s1), dtype=bool)
    tp = np.bincount(sd.rows, weights=labels.astype(float), minlength=len(sd.s1))
    oracle_f = float(fbeta_from_counts(tp, tp, n_true, beta)[subset].mean())
    lab = labels & subset[sd.rows]
    total = max(n_true[subset].sum(), 1)
    recall_at = {}
    for name in ("score", "cos_emb"):
        if f"{name}_rank_s1" not in sd.base:
            continue
        rank = sd.base[f"{name}_rank_s1"]
        for k in (1, 3, 5, 10, 20):
            recall_at[f"{name}_recall@{k}"] = float((lab & (rank < k)).sum() / total)
    return {"pair_recall": float(lab.sum() / total), "oracle_fbeta": oracle_f, **recall_at}


def _report_train(sd, probs, rule, truth, cfg, run_dir: Path, summary: dict, eligible) -> None:
    adj = adjusted(sd, probs, cfg)
    mask = decide(sd.rows, adj, rule, cfg.beta)
    pred = id_lists(sd.rows[mask], sd.cols[mask], adj[mask], sd.s1.ids, sd.pool.ids)
    keep_ids = set(sd.s1.ids[eligible].tolist())
    truth = {k: v for k, v in truth.items() if k in keep_ids}
    pred = {k: v for k, v in pred.items() if k in keep_ids}
    countries = dict(zip(sd.s1.ids, sd.s1.country))
    summary["train_full"] = {
        "fbeta": macro_fbeta(pred, truth, cfg.beta),
        **precision_recall(pred, truth),
        "by_country": breakdown(pred, truth, countries, cfg.beta),
        "by_true_cardinality": breakdown(pred, truth, cardinality_labels(truth), cfg.beta),
    }
    log.info("train (sample OOF + holdout) macro F: %.5f", summary["train_full"]["fbeta"])
    _error_samples(sd, pred, truth, run_dir)


def _error_samples(sd, pred, truth, run_dir: Path, n: int = 3000) -> None:
    pool_pos = pd.Index(sd.pool.ids)
    s1_pos = pd.Index(sd.s1.ids)
    fp, fn = [], []
    for sid, t in truth.items():
        p = set(pred.get(sid, ()))
        ts = set(t)
        fp += [(sid, x) for x in p - ts]
        fn += [(sid, x) for x in ts - p]
        if len(fp) > n and len(fn) > n:
            break
    for name, rows in (
        ("errors_false_positive.tsv", fp[:n]),
        ("errors_false_negative.tsv", fn[:n]),
    ):
        a = s1_pos.get_indexer([r[0] for r in rows])
        b = pool_pos.get_indexer([r[1] for r in rows])
        ok = (a >= 0) & (b >= 0)
        pd.DataFrame(
            {
                "s1_id": sd.s1.ids[a[ok]], "s1_name": sd.s1.name[a[ok]], "s1_addr": sd.s1.addr[a[ok]],
                "pool_id": sd.pool.ids[b[ok]], "pool_name": sd.pool.name[b[ok]], "pool_alias": sd.pool.alias[b[ok]],
                "pool_addr": sd.pool.addr[b[ok]],
            }
        ).to_csv(run_dir / name, sep="\t", index=False)  # fmt: skip


def _feature_importance(models: list, names: list[str]) -> pd.DataFrame:
    gain = np.mean([m.feature_importance("gain") for m in models], axis=0)
    df = pd.DataFrame({"feature": names, "gain": gain})
    df["share"] = df["gain"] / df["gain"].sum()
    return df.sort_values("gain", ascending=False)


def load_split_truth(data_dir: str | Path) -> dict[str, list[str]]:
    from ber.io import read_id_lists

    return read_id_lists(
        Path(data_dir) / "train" / schema.ground_truth_file("train"),
        schema.S1_ID,
        schema.MATCHED_IDS,
    )


def predict_split(
    model: TrainedModel,
    cfg: Config,
    data_dir: str | Path,
    out_dir: str | Path,
    split: str = "test",
    save_to: Path | None = None,
) -> dict[str, Any]:
    s1, pool = load_tables(data_dir, split, cfg)
    sd = block_split(s1, pool, cfg, split, split_key(data_dir, split, cfg))
    feats, feats_names = feature_matrix(sd, cfg)
    p1, keep = score_all(sd, cfg, model.stage1, model.feature_names, feats, feats_names)
    del feats
    probs = p1
    if model.stage2:
        cand_score = sd.base["score"]
        sd.base = {"score": cand_score}
        if _uses_scorers(model.stage2_names):
            scored = cross_scores(sd, cfg, p1, data_dir, Path(save_to or out_dir))
            pool_for = {k for k in scored if f"{k}_rank_pool" in model.stage2_names}
            keep.update(scorer_columns(sd.rows, sd.cols, scored, pool_for))
            del scored
        if "coh_emb" in model.stage2_names:
            keep.update(coherence_columns(sd.rows, sd.cols, p1, _pool_embeddings(sd, cfg)))
        cols2 = stage2_columns(sd.rows, sd.cols, p1, keep)
        del keep
        probs = predict_columns(
            model.stage2, cols2, model.stage2_names, len(sd.rows), cfg.scale.feature_chunk
        )
        del cols2
    if save_to is not None:
        np.savez(Path(save_to) / f"{split}_scores.npz", rows=sd.rows, cols=sd.cols, p=probs,
                 s1_ids=sd.s1.ids.astype("U"), pool_ids=sd.pool.ids.astype("U"))  # fmt: skip
    prior = None
    if cfg.decision.prior_adjust and model.train_prior:
        from ber.tuning import em_prior, rescale_prior

        new = em_prior(probs, model.train_prior)
        log.info(
            "%s: prior adjustment %.5f (train) -> %.5f (EM estimate)", split, model.train_prior, new
        )
        probs = rescale_prior(probs, model.train_prior, new)
        prior = {"train": model.train_prior, "estimated": new}
    adj = adjusted(sd, probs, cfg)
    mask = decide(sd.rows, adj, model.rule(), cfg.beta)
    matches = id_lists(sd.rows[mask], sd.cols[mask], adj[mask], sd.s1.ids, sd.pool.ids)
    candidates = id_lists(sd.rows, sd.cols, sd.base["score"], sd.s1.ids, sd.pool.ids)
    s1_ids = list(sd.s1.ids)
    m_path, c_path = write_outputs(out_dir, matches, candidates, s1_ids)
    report = validate_outputs(m_path, c_path, Path(data_dir) / split)
    print(report.render())
    out = {
        "n_s1": len(s1), "n_pool": len(pool), **sd.stats, "prior_adjust": prior,
        "predicted_pairs": int(mask.sum()),
        "predicted_singletons": int(sum(1 for v in matches.values() if not v)),
        "validation_ok": report.ok, "errors": report.errors[:5], "warnings": report.warnings[:5],
    }  # fmt: skip
    if (Path(data_dir) / split / schema.ground_truth_file(split)).exists():
        truth = load_split(data_dir, split).truth
        out["scored"] = {
            "fbeta": macro_fbeta(matches, truth, cfg.beta),
            **precision_recall(matches, truth),
        }
    return out


def _write_json(path: Path, obj: Any) -> None:
    def default(o):
        if isinstance(o, (np.integer, np.floating)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    path.write_text(json.dumps(obj, indent=2, default=default), encoding="utf-8")
