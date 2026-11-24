"""Copies that only add words their country's pool names lean on, at the same house number.

On train, a pool name that is its candidate's Source 1 name plus words from its country's
pool-heavy list (``ber.vocab.pool_heavy``: "riverside", "incorporated", "aka" in the US;
"shri", "overseas", "infratech" in India) is a true copy 99.2% (US, 27.5k pairs) and 97.3%
(India, 31.8k pairs) of the time when both give the same first house number, and mostly
not at a nearby number (22.8% and 70.4%): the generator's planted lookalikes add such a
word *and* move the house number.

France has no labels. Its pool-heavy words (associés, holding, international,
développement, ...) come from the same generator, but the France-adapted scorers were
self-trained on pseudo labels that never had one as a positive
(``scripts/make_pseudo_labels_r2.py`` leaves them out), so they reject the same-address
copies too. ``rescue_mask`` finds these pairs; the decision floors their probability at
``decision.rescue_floor``. Only the split's own inputs are used (no labels).
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence

import numpy as np
import pandas as pd

from ber.normalize.text import fold_latin
from ber.vocab import name_tokens, pool_heavy


def first_numbers(addresses: Sequence[str]) -> np.ndarray:
    """Each address's first run of digits without leading zeros ("" when it has none)."""
    s = pd.Series(np.asarray(addresses, dtype=object)).fillna("").astype(str)
    return s.str.extract(r"(\d+)", expand=False).fillna("").str.lstrip("0").to_numpy(dtype=object)


def rescue_mask(
    rows: np.ndarray,
    cols: np.ndarray,
    s1_names: Sequence[str],
    s1_addr: Sequence[str],
    s1_country: Sequence[str],
    pool_names: Sequence[str],
    pool_addr: Sequence[str],
    pool_country: Sequence[str],
    heavy: dict[str, set[str]] | None = None,
) -> np.ndarray:
    """Pairs (rows = Source 1 positions, cols = pool positions) whose pool name is the Source
    1 name plus only its country's pool-heavy words, at the same first house number."""
    s1_names = np.asarray(s1_names, dtype=object)
    pool_names = np.asarray(pool_names, dtype=object)
    c1 = pd.Series(np.asarray(s1_country, dtype=object)).fillna("").str.strip().str.lower()
    cp = pd.Series(np.asarray(pool_country, dtype=object)).fillna("").str.strip().str.lower()
    c1, cp = c1.to_numpy(dtype=object), cp.to_numpy(dtype=object)
    if heavy is None:
        heavy = pool_heavy(s1_names, c1, pool_names, cp)
    n1, n2 = first_numbers(s1_addr), first_numbers(pool_addr)
    idx = np.flatnonzero((n1[rows] != "") & (n1[rows] == n2[cols]))
    tok2: dict[int, frozenset[str]] = {}
    keep = np.zeros(len(rows), dtype=bool)
    for i in idx:
        r, q = int(rows[i]), int(cols[i])
        words = heavy.get(c1[r])
        if not words:
            continue
        t2 = tok2.get(q)
        if t2 is None:
            t2 = tok2[q] = name_tokens(pool_names[q])
        if not (t2 & words):
            continue
        t1 = name_tokens(s1_names[r])
        added = t2 - t1
        keep[i] = bool(t1) and bool(added) and t1 <= t2 and added <= words
    return keep


def replace_mask(
    rows: np.ndarray,
    cols: np.ndarray,
    s1_names: Sequence[str],
    s1_addr: Sequence[str],
    s1_country: Sequence[str],
    pool_names: Sequence[str],
    pool_addr: Sequence[str],
    words: dict[str, set[str]],
) -> np.ndarray:
    """Pairs whose pool name drops one Source 1 name word and adds one or two of its country's
    replacement words (``words``), at the same first house number.

    The generator's "replace" copies: "Nexio Wisconsin Inc" -> "Nexio Inc Center". On train
    the added word is one of center/services/service/partners and such a pair is a true copy
    99.5% (US) and 93% (India) of the time. France uses its own words ("Roubaix Federation
    EURL" -> "Roubaix EURL & Associés"). They were identified without labels: their pairs
    share the business's altered address rendering with its other copies from the same
    source as often as train's replace copies do (a fifth to a third of the time, where
    train's decoy words never do), and their pool records almost never belong to another
    business."""
    s1_names = np.asarray(s1_names, dtype=object)
    pool_names = np.asarray(pool_names, dtype=object)
    c1 = pd.Series(np.asarray(s1_country, dtype=object)).fillna("").str.strip().str.lower()
    c1 = c1.to_numpy(dtype=object)
    n1, n2 = first_numbers(s1_addr), first_numbers(pool_addr)
    idx = np.flatnonzero((n1[rows] != "") & (n1[rows] == n2[cols]))
    tok2: dict[int, frozenset[str]] = {}
    keep = np.zeros(len(rows), dtype=bool)
    for i in idx:
        r, q = int(rows[i]), int(cols[i])
        w = words.get(c1[r])
        if not w:
            continue
        t2 = tok2.get(q)
        if t2 is None:
            t2 = tok2[q] = name_tokens(pool_names[q])
        if not (t2 & w):
            continue
        t1 = name_tokens(s1_names[r])
        added = t2 - t1
        keep[i] = len(t1) >= 2 and len(t1 - t2) == 1 and 1 <= len(added) <= 2 and added <= w
    return keep


def swap_decoy_mask(
    rows: np.ndarray,
    cols: np.ndarray,
    s1_names: Sequence[str],
    s1_addr: Sequence[str],
    s1_country: Sequence[str],
    pool_names: Sequence[str],
    pool_addr: Sequence[str],
    countries: set[str],
    keep_words: set[str],
    min_vocab: int = 50,
    anywhere: bool = False,
) -> np.ndarray:
    """Pairs whose pool name swaps one Source 1 name word, in place (not as the last word),
    for another word common in its country's Source 1 names, at the same first house number.

    On train such swaps are true 10.1% (US) and 0.8% (India) of the time: they are the
    generator's same-address decoys ("Smith", "Blue", "Harbor"), unlike the replace copies
    that append a generic word at the end. France's self-trained scorers accept a third or
    more of theirs ("Forme Ecole SA" -> "Forme Amicale SA"). ``keep_words`` (the country's
    replace words and generic words like "centre") are never capped."""
    s1_names = np.asarray(s1_names, dtype=object)
    pool_names = np.asarray(pool_names, dtype=object)
    c1 = pd.Series(np.asarray(s1_country, dtype=object)).fillna("").str.strip().str.lower()
    c1 = c1.to_numpy(dtype=object)
    vocab: dict[str, Counter] = {}
    for c in countries:
        cnt: Counter = Counter()
        for n in s1_names[c1 == c]:
            cnt.update(name_tokens(n))
        vocab[c] = cnt
    n1, n2 = first_numbers(s1_addr), first_numbers(pool_addr)
    on = np.isin(c1[rows], sorted(countries)) & (n1[rows] != "") & (n1[rows] == n2[cols])
    last = re.compile(r"[a-z0-9]+")  # every token: the replace op appends after "SA" too
    keep = np.zeros(len(rows), dtype=bool)
    for i in np.flatnonzero(on):
        r, q = int(rows[i]), int(cols[i])
        t1, t2 = name_tokens(s1_names[r]), name_tokens(pool_names[q])
        dropped, added = t1 - t2, t2 - t1
        if len(t1) < 2 or len(dropped) != 1 or len(added) != 1:
            continue
        w = next(iter(added))
        if w in keep_words or vocab[c1[r]].get(w, 0) < min_vocab:
            continue
        words = last.findall(fold_latin(pool_names[q]).lower())
        keep[i] = anywhere or (bool(words) and words[-1] != w)
    return keep


def namesake_mask(
    rows: np.ndarray,
    cols: np.ndarray,
    s1_names: Sequence[str],
    s1_country: Sequence[str],
    pool_names: Sequence[str],
    pool_addr: Sequence[str],
    min_namesakes: int = 2,
) -> np.ndarray:
    """Pairs whose pool record has no address and the same name tokens as a Source 1 name
    that at least ``min_namesakes`` Source 1 businesses of its country share.

    On train such a copy belongs to one of the k namesakes, about 1/k each (0.49 for two,
    0.28 for three to five), and label-trained scorers accept it rarely (US/India test:
    21%/15% for two); France's self-trained ones accept it 2-3x as often (41% for two),
    betting on a lottery that F0.5 punishes."""
    s1_names = np.asarray(s1_names, dtype=object)
    c1 = pd.Series(np.asarray(s1_country, dtype=object)).fillna("").str.strip().str.lower()
    t1 = pd.Series([" ".join(sorted(name_tokens(n))) for n in s1_names])
    key = c1 + "|" + t1
    count = key.map(key.value_counts()).to_numpy()
    t1 = t1.to_numpy(dtype=object)
    empty = pd.Series(np.asarray(pool_addr, dtype=object)).fillna("").astype(str).str.strip().eq("")
    empty = empty.to_numpy()
    idx = np.flatnonzero(empty[cols] & (count[rows] >= min_namesakes) & (t1[rows] != ""))
    pool_names = np.asarray(pool_names, dtype=object)
    tok2: dict[int, str] = {}
    keep = np.zeros(len(rows), dtype=bool)
    for i in idx:
        q = int(cols[i])
        t2 = tok2.get(q)
        if t2 is None:
            t2 = tok2[q] = " ".join(sorted(name_tokens(pool_names[q])))
        keep[i] = t2 == t1[rows[i]]
    return keep
