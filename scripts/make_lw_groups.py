"""Listwise training groups straight from the raw TSVs: no pipeline run, no caches.

``ber.listwise train`` needs, for each training entity, a group of candidates with labels:
the entity's true copies plus the wrong records a matcher finds plausible. A finished run
gives those (``scripts/dump_lw_groups.py``); this script builds them from the data alone,
so the model can be trained on any machine with the dataset.

- **Entities**: a sample of the *encoder share* (``ber.embed.encoder_mask``, the 40% of
  train S1 the bi-encoder and the cross-encoders already train on). The matcher never
  trains or is judged on them, so the holdout of every run stays exactly as it was and a
  run with the listwise model compares directly with one without it.
- **Positives**: every true copy.
- **Negatives**: pool records of the same country that share the entity's rarest name or
  address tokens (namesakes, neighbours at another house number, other businesses at the
  same address, rebrands), ranked by how many rare tokens they share and whether the
  house number agrees; each entity keeps a random 0-7 of the best, so groups vary in size
  and difficulty as the p1 >= ``ce_min_p1`` groups of a run do.
- ``p1``: a proxy >= 0.05 (the ranking above), so every dumped pair enters the group.

``--side pool`` (C7) builds the mirror image: groups keyed on pool records, each with its
true owner and its best other Source 1 suitors, suitors drawn from the encoder share only
(so no holdout entity's label is learned), for copies owned by encoder-share entities and
for records nobody owns (``--unowned`` of them; their answer is "none of these").
``ber.listwise train`` also drops the owner from a fifth of the groups, as test's orphans.

Memory stays at a few GB: sources are read in chunks and tokens are kept as 64-bit hashes.
Also writes ``--valid`` keys for ``ber.listwise eval``.

    python scripts/make_lw_groups.py --data-dir DATA --out artifacts/lw_groups_enc.npz --train 400000
    python scripts/make_lw_groups.py --data-dir DATA --out artifacts/lwp_groups_enc.npz --side pool --train 800000
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from ber import schema

log = logging.getLogger("make_lw_groups")
TOKEN = re.compile(r"[^\W_]+")
NAME_STOP = frozenset(
    ["llc", "inc", "ltd", "limited", "pvt", "private", "corp", "corporation", "company", "the", "and", "of", "sarl", "sas", "eurl", "sasu", "snc", "sci", "llp", "plc", "incorporated"]
)  # fmt: skip
ADDR_STOP = frozenset(
    ["null", "n/a", "street", "road", "avenue", "drive", "lane", "unit", "floor", "suite", "box", "near", "opp", "opposite", "the", "and", "rue", "boulevard", "ave", "blvd"]
)  # fmt: skip
READ = {"sep": "\t", "dtype": str, "keep_default_na": False, "na_filter": False,
        "quoting": csv.QUOTE_NONE, "encoding": "utf-8-sig"}  # fmt: skip


def chunks(path: Path, cols: list[str], size: int = 500_000):
    for c in pd.read_csv(path, usecols=cols, chunksize=size, **READ):
        yield c.apply(lambda s: s.str.strip())


def id_key(ids: np.ndarray) -> np.ndarray:
    """Source digit and numeric part of an id as one int64 ("S2-681193310" -> 2_681193310)."""
    s = pd.Series(ids)
    return s.str[1].astype(np.int64).to_numpy() * 10**12 + s.str[3:].astype(np.int64).to_numpy()


def hashes(tokens: list[str]) -> np.ndarray:
    if not tokens:
        return np.empty(0, np.uint64)
    return pd.util.hash_array(np.asarray(tokens, dtype=object))


def tokenize(names, addrs):
    """Per record: name key-token hashes, address key-token hashes (CSR), first number."""
    nh, nlen, ah, alen, first = [], [], [], [], []
    for n, a in zip(names, addrs):
        nt = {t for t in TOKEN.findall(n.lower()) if len(t) >= 3 and t not in NAME_STOP}
        al = a.lower()
        at = {
            t for t in TOKEN.findall(al) if len(t) >= 3 and not t.isdigit() and t not in ADDR_STOP
        }
        m = re.search(r"\d+", al)
        nh.append(hashes(sorted(nt)))
        ah.append(hashes(sorted(at)))
        nlen.append(len(nt))
        alen.append(len(at))
        first.append(int(m.group()[:9]) if m else -1)
    cat = lambda xs: np.concatenate(xs) if xs else np.empty(0, np.uint64)  # noqa: E731
    return cat(nh), np.asarray(nlen), cat(ah), np.asarray(alen), np.asarray(first, np.int64)


class Index:
    """Token hash -> sorted record positions, and each token's document frequency."""

    def __init__(self, h: np.ndarray, lengths: np.ndarray):
        pos = np.repeat(np.arange(len(lengths), dtype=np.int64), lengths)
        order = np.argsort(h, kind="stable")
        self.h, self.pos = h[order], pos[order]

    def df(self, q: np.ndarray) -> np.ndarray:
        return np.searchsorted(self.h, q, "right") - np.searchsorted(self.h, q, "left")

    def postings(self, q: np.ndarray, owner: np.ndarray):
        """All (owner[i], pool position) for the query hashes q (owner: query's entity)."""
        lo, hi = np.searchsorted(self.h, q, "left"), np.searchsorted(self.h, q, "right")
        n = hi - lo
        if n.sum() == 0:
            return np.empty(0, np.int64), np.empty(0, np.int64)
        start = np.repeat(lo - np.concatenate([[0], np.cumsum(n)[:-1]]), n)
        idx = start + np.arange(n.sum())
        return np.repeat(owner, n), self.pos[idx]


def rarest(h: np.ndarray, lengths: np.ndarray, index: Index, cap: int, keep: int):
    """Per record, its ``keep`` rarest tokens with 1 <= df <= cap: (hashes, owner record)."""
    owner = np.repeat(np.arange(len(lengths)), lengths)
    df = index.df(h)
    ok = (df >= 1) & (df <= cap)
    h, owner, df = h[ok], owner[ok], df[ok]
    order = np.lexsort((df, owner))
    h, owner = h[order], owner[order]
    rank = np.arange(len(owner)) - np.searchsorted(owner, owner)
    sel = rank < keep
    return h[sel], owner[sel]


def subset(h: np.ndarray, lengths: np.ndarray, idx: np.ndarray):
    """The token hashes (CSR) of the records ``idx``, in that order."""
    start = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    n = lengths[idx]
    if n.sum() == 0:
        return np.empty(0, h.dtype), n
    at = np.repeat(start[idx] - np.concatenate([[0], np.cumsum(n)[:-1]]), n) + np.arange(n.sum())
    return h[at], n


def neighbours(query, index_n, index_a, t_first, t_country, cap: int, step: int = 20_000):
    """For query records (name hashes, lengths, address hashes, lengths, first numbers,
    countries), every same-country target sharing one of their 3 rarest name or address
    tokens, with a score: 2 x name hits + address hits + 1.5 if the house numbers agree.
    Yields (query index, target position, score) per chunk of queries."""
    qn, qnl, qa, qal, q_first, q_country = query
    hn, on = rarest(qn, qnl, index_n, cap, 3)
    ha, oa = rarest(qa, qal, index_a, cap, 3)
    for a in range(0, len(qnl), step):
        b = min(a + step, len(qnl))
        mn, ma = (on >= a) & (on < b), (oa >= a) & (oa < b)
        r1, c1 = index_n.postings(hn[mn], on[mn])
        r2, c2 = index_a.postings(ha[ma], oa[ma])
        key = np.concatenate([r1 * 2**34 + c1 * 2 + 1, r2 * 2**34 + c2 * 2])  # low bit: name hit
        pair = (key >> 34) * 2**33 + ((key >> 1) & (2**33 - 1))
        uniq, inv = np.unique(pair, return_inverse=True)
        name_hits = np.bincount(inv, weights=(key & 1), minlength=len(uniq))
        addr_hits = np.bincount(inv, weights=1 - (key & 1), minlength=len(uniq))
        q, t = uniq >> 33, uniq & (2**33 - 1)
        same = q_country[q] == t_country[t]
        num = (q_first[q] >= 0) & (q_first[q] == t_first[t])
        score = 2 * name_hits + addr_hits + 1.5 * num
        yield q[same], t[same], score[same]


def keep_best(q: np.ndarray, score: np.ndarray, n_keep: np.ndarray) -> np.ndarray:
    """Mask of each query's ``n_keep[q]`` best-scoring rows."""
    order = np.lexsort((-score, q))
    rank = np.empty(len(q), np.int64)
    rank[order] = np.arange(len(q)) - np.searchsorted(q[order], q[order])
    return rank < n_keep[q]


def proxy_p1(score: np.ndarray) -> np.ndarray:
    return (0.05 + 0.4 * np.minimum(score, 8) / 8).astype(np.float32)


def load(root: Path):
    """Source 1 frame, pool keys/countries/tokens (S2 then S3 = ber's pool order), owners."""
    t0 = time.time()
    s1 = pd.read_csv(root / schema.source_file("train", 1), **READ).apply(lambda s: s.str.strip())
    pk, pc, NH, NL, AH, AL, FN = [], [], [], [], [], [], []
    cols = [schema.ENTITY_ID, schema.NAME, schema.ADDRESS, schema.COUNTRY]
    for s in (2, 3):
        for c in chunks(root / schema.source_file("train", s), cols):
            pk.append(id_key(c[schema.ENTITY_ID].to_numpy()))
            pc.append(c[schema.COUNTRY].str.lower().to_numpy())
            nh, nl, ah, al, fn = tokenize(c[schema.NAME].tolist(), c[schema.ADDRESS].tolist())
            NH.append(nh), NL.append(nl), AH.append(ah), AL.append(al), FN.append(fn)
        log.info("pool source %d tokenised (%.0fs)", s, time.time() - t0)
    pool = {"key": np.concatenate(pk), "country": np.concatenate(pc), "first": np.concatenate(FN),
            "nh": np.concatenate(NH), "nl": np.concatenate(NL), "ah": np.concatenate(AH),
            "al": np.concatenate(AL)}  # fmt: skip
    gt = pd.read_csv(root / schema.ground_truth_file("train"), **READ)
    gt = gt.assign(p=gt[schema.MATCHED_IDS].str.split(",")).explode("p")
    gt = gt[gt["p"].str.strip() != ""]
    order = np.argsort(pool["key"])
    ppos = order[np.searchsorted(pool["key"][order], id_key(gt["p"].str.strip().to_numpy()))]
    s1_key = id_key(s1[schema.ENTITY_ID].to_numpy())
    s1_order = np.argsort(s1_key)
    spos = s1_order[np.searchsorted(s1_key[s1_order], id_key(gt[schema.S1_ID].to_numpy()))]
    owner = np.full(len(pool["key"]), -1, np.int64)
    owner[ppos] = spos
    n_true = np.bincount(spos, minlength=len(s1)).astype(np.float64)
    log.info("pool %d records, %d true pairs (%.0fs)", len(owner), len(spos), time.time() - t0)
    return s1, pool, owner, n_true


def s1_side(s1, pool, owner, args, rng, enc):
    """Groups keyed on Source 1 entities: every true copy + the best 0..max_neg neighbours."""
    n = len(enc)
    train = np.zeros(n, bool)
    train[rng.choice(np.flatnonzero(enc), min(args.train, int(enc.sum())), replace=False)] = True
    valid = np.zeros(n, bool)
    valid[rng.choice(np.flatnonzero(~enc), min(args.valid, int((~enc).sum())), replace=False)] = (
        True
    )
    pick = np.flatnonzero(train | valid)
    sub = s1.iloc[pick]
    nh, nl, ah, al, fn = tokenize(sub[schema.NAME].tolist(), sub[schema.ADDRESS].tolist())
    query = (nh, nl, ah, al, fn, sub[schema.COUNTRY].str.lower().to_numpy())
    name_idx, addr_idx = Index(pool["nh"], pool["nl"]), Index(pool["ah"], pool["al"])
    n_neg = rng.integers(0, args.max_neg + 1, len(pick))
    rows, cols, p1s, labs = [], [], [], []
    found = neighbours(query, name_idx, addr_idx, pool["first"], pool["country"], args.df_cap)
    for q, t, score in found:
        neg = owner[t] != pick[q]
        q, t, score = q[neg], t[neg], score[neg]
        k = keep_best(q, score, n_neg)
        rows.append(pick[q[k]]), cols.append(t[k]), p1s.append(proxy_p1(score[k]))
        labs.append(np.zeros(k.sum(), bool))
    pos = np.flatnonzero((owner >= 0) & (train | valid)[np.maximum(owner, 0)])
    rows.append(owner[pos]), cols.append(pos), labs.append(np.ones(len(pos), bool))
    p1s.append(np.full(len(pos), 0.9, np.float32))
    return rows, cols, p1s, labs, train, valid


def pool_side(s1, pool, owner, args, rng, enc):
    """Groups keyed on pool records (C7): the record's true owner + its best 0..max_neg other
    suitors, all from the encoder share, for copies owned by encoder-share entities and for
    records nobody owns (the answer there: none of these)."""
    n = len(owner)
    owned = np.flatnonzero((owner >= 0) & enc[np.maximum(owner, 0)])
    unowned = np.flatnonzero(owner < 0)
    n_un = int(args.train * args.unowned)
    pick_tr = np.concatenate([
        rng.choice(owned, min(args.train - n_un, len(owned)), replace=False),
        rng.choice(unowned, min(n_un, len(unowned)), replace=False),
    ])  # fmt: skip
    rest = np.setdiff1d(np.concatenate([owned, unowned]), pick_tr)
    pick_va = rng.choice(rest, min(args.valid, len(rest)), replace=False)
    train, valid = np.zeros(n, bool), np.zeros(n, bool)
    train[pick_tr], valid[pick_va] = True, True
    pick = np.flatnonzero(train | valid)
    encp = np.flatnonzero(enc)  # suitors: encoder-share Source 1 only
    sub = s1.iloc[encp]
    nh, nl, ah, al, fn = tokenize(sub[schema.NAME].tolist(), sub[schema.ADDRESS].tolist())
    name_idx, addr_idx = Index(nh, nl), Index(ah, al)
    qn, qnl = subset(pool["nh"], pool["nl"], pick)
    qa, qal = subset(pool["ah"], pool["al"], pick)
    query = (qn, qnl, qa, qal, pool["first"][pick], pool["country"][pick])
    s1_country = sub[schema.COUNTRY].str.lower().to_numpy()
    n_neg = rng.integers(0, args.max_neg + 1, len(pick))
    rows, cols, p1s, labs = [], [], [], []
    for q, t, score in neighbours(query, name_idx, addr_idx, fn, s1_country, args.df_cap):
        s1p = encp[t]
        neg = owner[pick[q]] != s1p
        q, s1p, score = q[neg], s1p[neg], score[neg]
        k = keep_best(q, score, n_neg)
        rows.append(s1p[k]), cols.append(pick[q[k]]), p1s.append(proxy_p1(score[k]))
        labs.append(np.zeros(k.sum(), bool))
    has = pick[owner[pick] >= 0]
    rows.append(owner[has]), cols.append(has), labs.append(np.ones(len(has), bool))
    p1s.append(np.full(len(has), 0.9, np.float32))
    return rows, cols, p1s, labs, train, valid


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--side", choices=["s1", "pool"], default="s1")
    ap.add_argument(
        "--train", type=int, default=400_000, help="training keys (entities or records)"
    )
    ap.add_argument("--valid", type=int, default=20_000, help="validation keys")
    ap.add_argument("--unowned", type=float, default=0.35, help="pool side: share nobody owns")
    ap.add_argument("--df-cap", type=int, default=3000)
    ap.add_argument("--max-neg", type=int, default=7)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    s1, pool, owner, n_true = load(Path(args.data_dir) / "train")
    ids = s1[schema.ENTITY_ID].to_numpy()
    enc = np.fromiter((zlib.crc32(str(i).encode()) % 1000 < 400 for i in ids), bool, len(ids))
    build = pool_side if args.side == "pool" else s1_side
    rows, cols, p1s, labs, train, valid = build(s1, pool, owner, args, rng, enc)
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    p1, labels = np.concatenate(p1s).astype(np.float32), np.concatenate(labs)
    keys = cols if args.side == "pool" else rows
    order = np.lexsort((-p1, keys))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, rows=rows[order].astype(np.int64), cols=cols[order].astype(np.int64),
             p1=p1[order], labels=labels[order], n_true=n_true, train=train, valid=valid,
             side=np.array(args.side))  # fmt: skip
    per = np.bincount(keys, minlength=len(train))[np.flatnonzero(train | valid)]
    log.info("%s (%s side): %d pairs (%.1f%% positive), %.2f per key (max %d); %d train, "
             "%d valid keys; %.0fs", out, args.side, len(rows), 100 * labels.mean(), per.mean(),
             per.max(), train.sum(), valid.sum(), time.time() - t0)  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
