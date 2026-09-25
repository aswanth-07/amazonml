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

Memory stays at a few GB: sources are read in chunks and tokens are kept as 64-bit hashes.
Also writes ``--valid`` entities from outside the encoder share for ``ber.listwise eval``.

    python scripts/make_lw_groups.py --data-dir DATA --out artifacts/lw_groups_enc.npz --train 400000
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
    """Token hash -> sorted pool positions, and each token's document frequency."""

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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train", type=int, default=400_000, help="encoder-share entities")
    ap.add_argument("--valid", type=int, default=20_000, help="entities outside the share")
    ap.add_argument("--df-cap", type=int, default=3000)
    ap.add_argument("--max-neg", type=int, default=7)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    t0 = time.time()
    root = Path(args.data_dir) / "train"
    rng = np.random.default_rng(args.seed)

    # ---- Source 1: ids, countries; pick the entities
    s1 = pd.read_csv(root / schema.source_file("train", 1), **READ)
    s1 = s1.apply(lambda s: s.str.strip())
    ids = s1[schema.ENTITY_ID].to_numpy()
    enc = np.fromiter((zlib.crc32(str(i).encode()) % 1000 < 400 for i in ids), bool, len(ids))
    train = np.zeros(len(ids), bool)
    train[rng.choice(np.flatnonzero(enc), min(args.train, int(enc.sum())), replace=False)] = True
    valid = np.zeros(len(ids), bool)
    valid[rng.choice(np.flatnonzero(~enc), min(args.valid, int((~enc).sum())), replace=False)] = (
        True
    )
    pick = np.flatnonzero(train | valid)
    s1_key = id_key(ids)
    s1_country = s1[schema.COUNTRY].str.lower().to_numpy()
    log.info("S1 %d; train %d, valid %d", len(ids), train.sum(), valid.sum())

    # ---- pool: keys, countries, tokens (chunked; S2 then S3 = ber's pool order)
    pk, pc, NH, NL, AH, AL, FN = [], [], [], [], [], [], []
    for s in (2, 3):
        for c in chunks(root / schema.source_file("train", s),
                        [schema.ENTITY_ID, schema.NAME, schema.ADDRESS, schema.COUNTRY]):  # fmt: skip
            pk.append(id_key(c[schema.ENTITY_ID].to_numpy()))
            pc.append(c[schema.COUNTRY].str.lower().to_numpy())
            nh, nl, ah, al, fn = tokenize(c[schema.NAME].tolist(), c[schema.ADDRESS].tolist())
            NH.append(nh), NL.append(nl), AH.append(ah), AL.append(al), FN.append(fn)
        log.info("pool source %d tokenised (%.0fs)", s, time.time() - t0)
    pool_key, pool_country = np.concatenate(pk), np.concatenate(pc)
    pool_first = np.concatenate(FN)
    name_idx = Index(np.concatenate(NH), np.concatenate(NL))
    addr_idx = Index(np.concatenate(AH), np.concatenate(AL))
    del NH, AH
    log.info("pool %d records indexed (%.0fs)", len(pool_key), time.time() - t0)

    # ---- truth: owner per pool position
    gt = pd.read_csv(root / schema.ground_truth_file("train"), **READ)
    gt = gt.assign(p=gt[schema.MATCHED_IDS].str.split(",")).explode("p")
    gt = gt[gt["p"].str.strip() != ""]
    order = np.argsort(pool_key)
    at = np.searchsorted(pool_key[order], id_key(gt["p"].str.strip().to_numpy()))
    ppos = order[np.minimum(at, len(order) - 1)]
    s1_order = np.argsort(s1_key)
    spos = s1_order[np.searchsorted(s1_key[s1_order], id_key(gt[schema.S1_ID].to_numpy()))]
    owner = np.full(len(pool_key), -1, np.int64)
    owner[ppos] = spos
    n_true = np.bincount(spos, minlength=len(ids)).astype(np.float64)
    log.info("truth: %d pairs", len(spos))

    # ---- candidates for the picked entities
    sub = s1.iloc[pick]
    nh, nl, ah, al, fn = tokenize(sub[schema.NAME].tolist(), sub[schema.ADDRESS].tolist())
    qn, on = rarest(nh, nl, name_idx, args.df_cap, 3)
    qa, oa = rarest(ah, al, addr_idx, args.df_cap, 3)
    rows, cols, p1s, labs = [], [], [], []
    step = 20_000
    for a in range(0, len(pick), step):
        b = min(a + step, len(pick))
        mn, ma = (on >= a) & (on < b), (oa >= a) & (oa < b)
        r1, c1 = name_idx.postings(qn[mn], on[mn])
        r2, c2 = addr_idx.postings(qa[ma], oa[ma])
        key = np.concatenate([r1 * 2**34 + c1 * 2 + 1, r2 * 2**34 + c2 * 2])  # name hit: +1 bit
        r = key >> 34
        c = (key >> 1) & (2**33 - 1)
        pair = r * 2**33 + c
        uniq, inv = np.unique(pair, return_inverse=True)
        name_hits = np.bincount(inv, weights=(key & 1), minlength=len(uniq))
        addr_hits = np.bincount(inv, weights=1 - (key & 1), minlength=len(uniq))
        r, c = uniq >> 33, uniq & (2**33 - 1)
        s1p = pick[r]
        same = s1_country[s1p] == pool_country[c]
        num = (fn[r] >= 0) & (fn[r] == pool_first[c])
        score = 2 * name_hits + addr_hits + 1.5 * num
        neg = same & (owner[c] != s1p)
        r, c, s1p, score = r[neg], c[neg], s1p[neg], score[neg]
        order = np.lexsort((-score, r))
        r, c, s1p, score = r[order], c[order], s1p[order], score[order]
        rank = np.arange(len(r)) - np.searchsorted(r, r)
        n_neg = rng.integers(0, args.max_neg + 1, b - a)
        keep = rank < n_neg[r - a]
        rows.append(s1p[keep]), cols.append(c[keep]), labs.append(np.zeros(keep.sum(), bool))
        p1s.append(0.05 + 0.4 * np.minimum(score[keep], 8) / 8)
    pos = np.flatnonzero((owner >= 0) & (train | valid)[np.maximum(owner, 0)])
    rows.append(owner[pos]), cols.append(pos), labs.append(np.ones(len(pos), bool))
    p1s.append(np.full(len(pos), 0.9))
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    p1, labels = np.concatenate(p1s).astype(np.float32), np.concatenate(labs)
    order = np.lexsort((-p1, rows))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, rows=rows[order].astype(np.int64), cols=cols[order].astype(np.int64),
             p1=p1[order], labels=labels[order], n_true=n_true, train=train, valid=valid)  # fmt: skip
    per = np.bincount(rows, minlength=len(ids))[pick]
    log.info("%s: %d pairs (%.1f%% positive), %.2f per entity (max %d) in %.0fs", out,
             len(rows), 100 * labels.mean(), per.mean(), per.max(), time.time() - t0)  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
