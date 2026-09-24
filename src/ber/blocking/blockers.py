"""Individual blockers: TF-IDF nearest neighbours, shared keys, dense embeddings."""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from collections.abc import Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from ber.config import EmbeddingBlockerConfig, KeyBlockerConfig, TfidfBlockerConfig

log = logging.getLogger(__name__)


class TfidfBlocker:
    """Fit TF-IDF on both sides together (unsupervised) and expose normalised vectors."""

    def __init__(self, cfg: TfidfBlockerConfig) -> None:
        self.cfg = cfg

    def _vectorizer(self, min_df: int, max_df: float) -> TfidfVectorizer:
        kwargs = {
            "analyzer": self.cfg.analyzer,
            "ngram_range": tuple(self.cfg.ngram_range),
            "min_df": min_df,
            "max_df": max_df,
            "sublinear_tf": True,
            "lowercase": False,
            "dtype": np.float32,
        }
        if self.cfg.analyzer == "word":
            kwargs["token_pattern"] = r"(?u)\b\w+\b"
        return TfidfVectorizer(**kwargs)

    def fit_transform(
        self, s1_texts: Sequence[str], pool_texts: Sequence[str]
    ) -> tuple[sp.csr_matrix, sp.csr_matrix]:
        corpus = list(s1_texts) + list(pool_texts)
        try:
            matrix = self._vectorizer(self.cfg.min_df, self.cfg.max_df).fit_transform(corpus)
        except ValueError:  # tiny or degenerate corpus: every term pruned
            log.warning(
                "%s: document-frequency pruning removed every term; refitting unpruned",
                self.cfg.name,
            )
            try:
                matrix = self._vectorizer(1, 1.0).fit_transform(corpus)
            except ValueError:
                matrix = sp.csr_matrix((len(corpus), 1), dtype=np.float32)
        matrix = sp.csr_matrix(matrix, dtype=np.float32)
        n1 = len(s1_texts)
        return matrix[:n1], matrix[n1:]


def key_sets(frame: pd.DataFrame, key: str) -> list[set[str]]:
    """Blocking keys per normalised record for a named key function."""
    if key == "postal":
        return [set(p.split()) for p in frame["postal"]]
    if key == "postal_name":
        return [
            {f"{p}|{core[:2]}" for p in postal.split()} if core else set()
            for postal, core in zip(frame["postal"], frame["name_core"])
        ]
    if key == "acronym":
        out = []
        for acronym, core in zip(frame["name_acronym"], frame["name_core"]):
            keys = {acronym} if len(acronym) >= 3 else set()
            # A one-word name that could itself be an acronym ("ibm"); "i b m" is covered above.
            if " " not in core and 3 <= len(core) <= 6 and core.isalpha():
                keys.add(core)
            out.append(keys)
        return out
    if key == "name_first":
        return [
            {core.split()[0]} if core and len(core.split()[0]) >= 3 else set()
            for core in frame["name_core"]
        ]
    if key == "skeleton":
        return [{s} if s else set() for s in frame["name_skeleton"]]
    raise ValueError(
        f"unknown blocking key {key!r}; known: postal, postal_name, acronym, name_first, skeleton"
    )


def key_candidates(
    s1_keys: list[set[str]], pool_keys: list[set[str]], max_block_size: int
) -> tuple[np.ndarray, np.ndarray]:
    index: dict[str, list[int]] = defaultdict(list)
    for j, keys in enumerate(pool_keys):
        for k in keys:
            index[k].append(j)
    rows: list[int] = []
    cols: list[int] = []
    for i, keys in enumerate(s1_keys):
        hits: set[int] = set()
        for k in keys:
            members = index.get(k)
            if members and len(members) <= max_block_size:
                hits.update(members)
        rows.extend([i] * len(hits))
        cols.extend(hits)
    return np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)


def run_key_blocker(
    cfg: KeyBlockerConfig, s1n: pd.DataFrame, pooln: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    return key_candidates(key_sets(s1n, cfg.key), key_sets(pooln, cfg.key), cfg.max_block_size)


class EmbeddingBlocker:
    """Dense sentence embeddings from a locally stored, licence-checked model.

    Runs offline: the model must already be in ``cfg.local_path`` (``ber fetch-model``).
    Nothing about any business is ever sent anywhere.
    """

    def __init__(self, cfg: EmbeddingBlockerConfig) -> None:
        from ber.pretrained import check_allowed

        check_allowed(cfg.model_name)
        self.cfg = cfg
        self._model = None

    def _load(self):  # pragma: no cover - needs torch + weights
        if self._model is None:
            if not os.path.isdir(self.cfg.local_path):
                raise FileNotFoundError(
                    f"{self.cfg.local_path} not found; run `ber fetch-model {self.cfg.model_name}` first"
                )
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.cfg.local_path, device=self.cfg.device)
        return self._model

    def encode(
        self, texts: Sequence[str]
    ) -> np.ndarray:  # pragma: no cover - needs torch + weights
        model = self._load()
        prefixed = [f"{self.cfg.prefix}{t}" for t in texts]
        vectors = model.encode(
            prefixed,
            batch_size=self.cfg.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)
