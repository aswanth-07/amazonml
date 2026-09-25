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
import time
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
from ber.pairfeat import group_features, group_stats, number_features, string_features
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
    "score", "cos_name", "cos_addr", "name_tset", "name_best", "addr_tset", "key_tset",
    "first_num_eq", "squash_partial", "num_jaccard",
)  # fmt: skip


BLOCKING_VERSION = 1  # bump when tokens, weighting or search change which pairs blocking returns
FEATURE_VERSION = 1  # bump when the stage-1 pair features change


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


def block_split(s1: Table, pool: Table, cfg: Config, name: str, key: str = "") -> SplitData:
    """Candidate pairs, their cosines and group context. With a ``key``, the pairs, cosines
    and number matrices are cached, so a rerun skips hashing and search entirely."""
    path = Path(cfg.scale.cache_dir) / f"{name}_blocking_{key}.joblib" if key else None
    if path is not None and path.exists():
        with timed(f"{name}: load cached blocking {path.name}"):
            c = joblib.load(path)
        rows, cols, base, mq, md = c["rows"], c["cols"], c["base"], c["num_q"], c["num_d"]
    else:
        rows, cols, base, mq, md = _block(s1, pool, cfg, name)
        if path is not None:
            with timed(f"{name}: save blocking cache {path.name}"):
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(".tmp")
                joblib.dump(
                    {"rows": rows, "cols": cols, "base": base, "num_q": mq, "num_d": md}, tmp
                )
                tmp.replace(path)
    with timed(f"{name}: group context"):
        base.update(group_features(rows, cols, base))
    stats = {
        "pairs": float(len(rows)),
        "candidates_per_s1": float(len(rows) / max(len(s1), 1)),
        "s1_without_candidates": float(len(s1) - len(np.unique(rows))),
        "reduction_ratio": float(1 - len(rows) / max(len(s1) * len(pool), 1)),
    }
    log.info("%s: %s", name, stats)
    return SplitData(name, s1, pool, rows, cols, base, mq, md, stats, key)


def _block(s1: Table, pool: Table, cfg: Config, name: str):
    """Hash, weight, search: (rows, cols sorted by S1 then score, cosines, number matrices)."""
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
    rows, cols = candidate_pairs(views, s1.country, pool.country, cfg, name)
    with timed(f"{name}: cosines"):
        base = views.cosines(rows, cols)
    order = np.lexsort((-base["score"], rows))
    rows, cols = rows[order].astype(np.int32), cols[order].astype(np.int32)
    base = {k: v[order] for k, v in base.items()}
    return rows, cols, base, mq, md


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
    rows, cols = sd.rows[idx], sd.cols[idx]
    cols_out: dict[str, np.ndarray] = {k: v[idx] for k, v in sd.base.items()}
    cols_out.update(string_features(sd.s1, sd.pool, rows, cols))
    cols_out.update(number_features(sd.num_q, sd.num_d, rows, cols))
    return pd.DataFrame(cols_out)


def feature_matrix(sd: SplitData, cfg: Config) -> tuple[np.ndarray, list[str]]:
    """Every candidate pair's stage-1 features as one float32 matrix, in pair order.

    With a split key it is a memory-mapped ``.npy`` in the cache directory; its column
    names are written last, so a file from an interrupted run is never mistaken for
    a finished one.
    """
    n = len(sd.rows)
    path = (
        Path(cfg.scale.cache_dir) / f"{sd.name}_features_{sd.key}_v{FEATURE_VERSION}.npy"
        if sd.key
        else None
    )
    names_path = path.with_suffix(".json") if path is not None else None
    if path is not None and path.exists() and names_path.exists():
        log.info("%s: cached features %s", sd.name, path.name)
        return np.load(path, mmap_mode="r"), json.loads(names_path.read_text())
    out, names = None, []
    for idx in _chunks(n, cfg.scale.feature_chunk):
        with timed(f"{sd.name}: features {idx[0]:,}-{idx[-1]:,} of {n:,}"):
            x = pair_features(sd, idx)
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


def fit_folds(x: pd.DataFrame, y: np.ndarray, groups: np.ndarray, cfg: Config, tag: str):
    """Grouped K-fold LightGBM. Returns (out-of-fold probabilities, fold models, best iterations)."""
    import lightgbm as lgb

    folds = group_folds(groups, cfg.model.n_folds, cfg.seed)
    oof = np.zeros(len(x), dtype=np.float32)
    models, iters = [], []
    for k in range(cfg.model.n_folds):
        tr, va = folds != k, folds == k
        with timed(f"{tag} fold {k + 1}/{cfg.model.n_folds}"):
            dtr = lgb.Dataset(x[tr], label=y[tr], free_raw_data=True)
            dva = lgb.Dataset(x[va], label=y[va], reference=dtr, free_raw_data=True)
            booster = lgb.train(
                _lgb_params(cfg, cfg.seed + k),
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
    keep = {k: np.empty(n, dtype=np.float32) for k in STAGE2_KEEP}
    for idx in _chunks(n, cfg.scale.feature_chunk):
        s, e = idx[0], idx[-1] + 1
        with timed(f"{sd.name}: predict {s:,}-{e:,}"):
            block = np.asarray(feats[s:e])
            p1[s:e] = predict_avg(stage1, pd.DataFrame(block[:, pos], columns=feature_names))
            for k in STAGE2_KEEP:
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
    sd = block_split(s1, pool, cfg, "train", split_key(data_dir, "train", cfg))
    labels, n_true = pair_labels(sd, truth)
    summary["train_blocking"] = {**sd.stats, **blocking_recall(sd, labels, n_true, cfg.beta)}
    log.info("train blocking: %s", summary["train_blocking"])
    _write_json(run_dir / "metrics.json", summary)

    # ---- stage 1 on a sample of entities; every other entity is an untouched holdout
    rng = np.random.default_rng(cfg.seed)
    sample = np.sort(rng.choice(len(s1), min(cfg.scale.train_sample, len(s1)), replace=False))
    in_sample = np.zeros(len(s1), dtype=bool)
    in_sample[sample] = True
    holdout = np.flatnonzero(~in_sample)
    sidx = np.flatnonzero(in_sample[sd.rows])
    feats, feature_names = feature_matrix(sd, cfg)
    with timed("train: sample features"):
        xs = pd.DataFrame(np.asarray(feats[sidx]), columns=feature_names)
    ys = labels[sidx].astype(np.int8)
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
    final_p, stage2, s2_names, thr = p1, [], [], summary["stage1"]["threshold"]

    # ---- stage 2: stage-1 probabilities in context
    if cfg.scale.stage2:
        cols2 = stage2_columns(sd.rows, sd.cols, p1, keep)
        del keep
        s2_names = list(cols2)
        oof2, stage2, iters2 = fit_folds(
            _frame(cols2, sidx, s2_names), ys, sd.rows[sidx], cfg, "stage2"
        )
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

    _report_train(sd, final_p, thr, truth, cfg, run_dir, summary)
    model = TrainedModel(stage1, stage2, feature_names, s2_names, thr, cfg.to_dict())
    Path(cfg.artifacts_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(model, Path(cfg.artifacts_dir) / "scale_model.joblib")
    joblib.dump(model, run_dir / "scale_model.joblib")
    _write_json(run_dir / "metrics.json", summary)
    del sd, s1, pool, p1, final_p

    if predict_test:
        summary["test"] = predict_split(model, cfg, data_dir, out_dir)
    summary["minutes"] = round((time.time() - t0) / 60, 1)
    _write_json(run_dir / "metrics.json", summary)
    log.info("run recorded at %s (%.1f min)", run_dir, summary["minutes"])
    return summary


def _stage_report(sd, probs, labels, n_true, sample, holdout, cfg, iters) -> dict[str, Any]:
    adj = adjusted(sd, probs, cfg)
    thr, f_sample = tune_threshold(sd, adj, labels, n_true, sample, cfg)
    out = {"threshold": thr, "sample_oof": f_sample, "best_iterations": iters}
    if len(holdout):
        out["holdout"] = subset_fbeta(sd, adj, labels, n_true, thr, holdout, cfg.beta)
    log.info("stage report: %s", out)
    return out


def blocking_recall(
    sd: SplitData, labels: np.ndarray, n_true: np.ndarray, beta: float
) -> dict[str, float]:
    tp = np.bincount(sd.rows, weights=labels.astype(float), minlength=len(sd.s1))
    oracle_f = float(
        fbeta_from_counts(tp, tp, n_true, beta).mean()
    )  # perfect matcher on these candidates
    recall_at = {}
    rank = sd.base["score_rank_s1"]
    for k in (1, 3, 5, 10, 20):
        recall_at[f"recall@{k}"] = float((labels & (rank < k)).sum() / max(n_true.sum(), 1))
    return {
        "pair_recall": float(labels.sum() / max(n_true.sum(), 1)),
        "oracle_fbeta": oracle_f,
        **recall_at,
    }


def _report_train(sd, probs, thr, truth, cfg, run_dir: Path, summary: dict) -> None:
    adj = adjusted(sd, probs, cfg)
    mask = (adj >= thr) & (adj > 0)
    pred = id_lists(sd.rows[mask], sd.cols[mask], adj[mask], sd.s1.ids, sd.pool.ids)
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
    model: TrainedModel, cfg: Config, data_dir: str | Path, out_dir: str | Path, split: str = "test"
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
        cols2 = stage2_columns(sd.rows, sd.cols, p1, keep)
        del keep
        probs = predict_columns(
            model.stage2, cols2, model.stage2_names, len(sd.rows), cfg.scale.feature_chunk
        )
        del cols2
    adj = adjusted(sd, probs, cfg)
    mask = (adj >= model.threshold) & (adj > 0)
    matches = id_lists(sd.rows[mask], sd.cols[mask], adj[mask], sd.s1.ids, sd.pool.ids)
    candidates = id_lists(sd.rows, sd.cols, sd.base["score"], sd.s1.ids, sd.pool.ids)
    s1_ids = list(sd.s1.ids)
    m_path, c_path = write_outputs(out_dir, matches, candidates, s1_ids)
    report = validate_outputs(m_path, c_path, Path(data_dir) / split)
    print(report.render())
    out = {
        "n_s1": len(s1), "n_pool": len(pool), **sd.stats,
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
