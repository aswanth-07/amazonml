"""Pool-heavy name words: words a country's Source 2/3 names use far more than its Source 1.

Computed per split from that split's own inputs (no labels). Two kinds show up:

- words pool-heavy in every country ("com", "www", "dba", "formerly", "incorporated"):
  copy noise, which train's true copies add to their business's name 9-13% of the time;
- words pool-heavy in one country only. In France these are développement, participations,
  distribution, holding, international, associés: 10% of French pool names, 0.35% of
  French S1 names, never in train. The pool records carrying them are decoys (a
  neighbouring house number, or one name word changed at the same address), and no model
  has seen them as a decoy signal.

So only a country's *specific* words mark a likely decoy: a pool name that adds one to its
candidate business's name. ``scripts/make_pseudo_labels_r2.py`` keeps them out of the
round-2 French pseudo-positives.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Sequence

import numpy as np

from ber.normalize.name import LEGAL_SET
from ber.normalize.text import fold_latin

_TOKEN = re.compile(r"[a-z0-9]{3,}")


def name_tokens(name: str) -> frozenset[str]:
    """Lowercase, accent-folded word tokens of three or more characters, legal forms left out."""
    return frozenset(t for t in _TOKEN.findall(fold_latin(name).lower()) if t not in LEGAL_SET)


def _doc_freq(names: Iterable[str]) -> tuple[Counter, int]:
    df: Counter = Counter()
    n = 0
    for name in names:
        df.update(name_tokens(name))
        n += 1
    return df, n


def pool_heavy(
    s1_names: Sequence[str],
    s1_country: Sequence[str],
    pool_names: Sequence[str],
    pool_country: Sequence[str],
    min_share: float = 1e-3,
    ratio: float = 5.0,
    min_count: int = 50,
) -> dict[str, set[str]]:
    """Per country: the words in at least ``min_share`` (and ``min_count``) of its pool names
    that are at least ``ratio`` times as frequent there as in its Source 1 names. The count
    floor keeps one-off typos out on small data; on the real test set (1.4M French pool
    names) the share alone already asks for 1,400."""
    s1_country, pool_country = np.asarray(s1_country), np.asarray(pool_country)
    s1_names, pool_names = np.asarray(s1_names, dtype=object), np.asarray(pool_names, dtype=object)
    out: dict[str, set[str]] = {}
    for c in sorted(set(pool_country.tolist())):
        dfp, n_pool = _doc_freq(pool_names[pool_country == c])
        df1, n_s1 = _doc_freq(s1_names[s1_country == c])
        words = set()
        for t, k in dfp.items():
            share = k / n_pool
            if (
                k >= min_count
                and share >= min_share
                and share / (df1[t] / max(n_s1, 1) + 1e-4) >= ratio
            ):
                words.add(t)
        out[str(c)] = words
    return out


def specific_words(heavy: dict[str, set[str]]) -> dict[str, set[str]]:
    """Each country's pool-heavy words that no other country's pool leans on."""
    return {
        c: words - set().union(*(w for k, w in heavy.items() if k != c))
        for c, words in heavy.items()
    }


def added_words(s1_name: str, pool_name: str, words: set[str]) -> list[str]:
    """The ``words`` the pool name has and the Source 1 name lacks."""
    return sorted((name_tokens(pool_name) - name_tokens(s1_name)) & words)
