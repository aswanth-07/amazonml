"""Candidate generation: the union of several blockers, capped per Source 1 entity.

The capped set is exactly what the matcher scores and what
``candidate_pairs.tsv`` reports, so final matches are always a subset of it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ber import schema
from ber.blocking.blockers import EmbeddingBlocker, TfidfBlocker, run_key_blocker
from ber.blocking.search import country_partitions, pair_cosine, topk_partitioned
from ber.config import BlockingConfig
from ber.utils import timed

log = logging.getLogger(__name__)

S1_IDX = "s1_idx"
POOL_IDX = "pool_idx"


@dataclass
class CandidateSet:
    """Candidate pairs as positions into the normalised S1 and pool frames."""

    pairs: (
        pd.DataFrame
    )  # s1_idx, pool_idx, <tfidf>_score..., hit_<blocker>..., n_blockers, cap_score
    score_columns: list[str]
    hit_columns: list[str]
    stats: dict[str, float] = field(default_factory=dict)
    precap: pd.DataFrame | None = None  # s1_idx, pool_idx, hit_* before the cap (diagnostics)

    def id_lists(
        self, s1_ids: np.ndarray, pool_ids: np.ndarray, frame: pd.DataFrame | None = None
    ) -> dict[str, list[str]]:
        """{S1 id: [pool ids]} ordered by cap score, for every S1 id (empty lists included)."""
        frame = self.pairs if frame is None else frame
        ordered = frame.sort_values([S1_IDX, "cap_score"], ascending=[True, False], kind="stable")
        out: dict[str, list[str]] = {sid: [] for sid in s1_ids}
        for s1_pos, group in ordered.groupby(S1_IDX, sort=False):
            out[s1_ids[s1_pos]] = pool_ids[group[POOL_IDX].to_numpy()].tolist()
        return out


class CandidateGenerator:
    def __init__(self, cfg: BlockingConfig) -> None:
        self.cfg = cfg

    def generate(self, s1n: pd.DataFrame, pooln: pd.DataFrame) -> CandidateSet:
        cfg = self.cfg
        partitions = list(
            country_partitions(
                s1n["country_norm"].to_numpy(),
                pooln["country_norm"].to_numpy(),
                cfg.partition_by_country,
            )
        )
        frames: list[pd.DataFrame] = []
        vectors: dict[str, tuple[sp.csr_matrix | np.ndarray, sp.csr_matrix | np.ndarray]] = {}

        for bcfg in cfg.tfidf:
            with timed(f"blocker {bcfg.name}"):
                q, d = TfidfBlocker(bcfg).fit_transform(
                    s1n[bcfg.field].tolist(), pooln[bcfg.field].tolist()
                )
                vectors[bcfg.name] = (q, d)
                r, c, _ = topk_partitioned(
                    q, d, partitions, bcfg.top_k, bcfg.min_sim, cfg.memory_budget_mb
                )
                frames.append(_hits(r, c, bcfg.name))

        for kcfg in cfg.keys:
            with timed(f"blocker {kcfg.name}"):
                r, c = run_key_blocker(kcfg, s1n, pooln)
                if cfg.partition_by_country:
                    r, c = _same_country(r, c, s1n, pooln)
                frames.append(_hits(r, c, kcfg.name))

        if cfg.embedding.enabled:
            ecfg = cfg.embedding
            with timed("blocker embedding"):
                encoder = EmbeddingBlocker(ecfg)
                q = encoder.encode(s1n[ecfg.field].tolist())
                d = encoder.encode(pooln[ecfg.field].tolist())
                vectors["emb"] = (q, d)
                r, c, _ = topk_partitioned(
                    q, d, partitions, ecfg.top_k, ecfg.min_sim, cfg.memory_budget_mb
                )
                frames.append(_hits(r, c, "emb"))

        hit_columns = [f"hit_{name}" for name in _blocker_names(cfg)]
        union = pd.concat(frames, ignore_index=True)
        union = union.groupby([S1_IDX, POOL_IDX], sort=True).max().reset_index()
        for col in hit_columns:
            union[col] = union[col].fillna(0).astype(np.int8) if col in union else np.int8(0)
        union["n_blockers"] = union[hit_columns].sum(axis=1).astype(np.int8)

        rows, cols = union[S1_IDX].to_numpy(), union[POOL_IDX].to_numpy()
        score_columns = []
        for name, (q, d) in vectors.items():
            column = f"{name}_score"
            union[column] = pair_cosine(q, d, rows, cols)
            score_columns.append(column)

        weights = {b.name: b.cap_weight for b in cfg.tfidf}
        union["cap_score"] = (
            sum(weights.get(n, 0.0) * union[f"{n}_score"] for n in weights)
            + 0.05 * union["n_blockers"]
        )
        union = union.sort_values([S1_IDX, "cap_score"], ascending=[True, False], kind="stable")
        union["cap_rank"] = union.groupby(S1_IDX).cumcount() + 1
        precap = union[[S1_IDX, POOL_IDX, *hit_columns]].copy()
        capped = union[union["cap_rank"] <= cfg.max_candidates].reset_index(drop=True)

        n_s1, n_pool = len(s1n), len(pooln)
        stats = {
            "pairs_before_cap": float(len(union)),
            "pairs_after_cap": float(len(capped)),
            "mean_candidates_per_s1": float(len(capped) / max(n_s1, 1)),
            "s1_without_candidates": float(n_s1 - capped[S1_IDX].nunique()),
            "reduction_ratio": float(1.0 - len(capped) / max(n_s1 * n_pool, 1)),
        }
        log.info(
            "candidates: %d before cap, %d after (%.1f per S1), reduction ratio %.6f",
            len(union),
            len(capped),
            stats["mean_candidates_per_s1"],
            stats["reduction_ratio"],
        )
        return CandidateSet(capped, score_columns, hit_columns, stats, precap)


def _blocker_names(cfg: BlockingConfig) -> list[str]:
    names = [b.name for b in cfg.tfidf] + [k.name for k in cfg.keys]
    return [*names, "emb"] if cfg.embedding.enabled else names


def _hits(rows: np.ndarray, cols: np.ndarray, name: str) -> pd.DataFrame:
    return pd.DataFrame(
        {S1_IDX: rows, POOL_IDX: cols, f"hit_{name}": np.ones(len(rows), dtype=np.int8)}
    )


def _same_country(
    rows: np.ndarray, cols: np.ndarray, s1n: pd.DataFrame, pooln: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    a = s1n["country_norm"].to_numpy()[rows]
    b = pooln["country_norm"].to_numpy()[cols]
    keep = (a == b) | (a == "") | (b == "")
    return rows[keep], cols[keep]


def pool_ids(pooln: pd.DataFrame) -> np.ndarray:
    return pooln[schema.ENTITY_ID].to_numpy()
