"""Dense name + address embeddings: contrastive fine-tuning, encoding, GPU kNN.

The encoder (``intfloat/multilingual-e5-small``, MIT, 118M parameters) is
fine-tuned on train pairs so that a Source 1 record and its Source 2/3 matches
embed close together. It learns the noise the token views cannot see: other
scripts, transliteration, reordered and abbreviated addresses, trade names.
Only the provided training data is used, and only a pretrained open-licence
model; nothing about any business is looked up.

This module needs PyTorch and transformers, which live in a separate
interpreter here (CUDA PyTorch for Python 3.13). ``ber.scale`` reads only the
arrays written below (``load_embeddings``, ``load_knn``), so the main pipeline
does not import torch.

Leakage. The encoder is trained on a fixed subset of train Source 1 entities
(``encoder_mask``). ``ber.scale`` never trains, tunes or evaluates the matcher
on those entities, so the embedding similarity of every pair the matcher learns
from is as out-of-sample as it is on test.

Usage (from the repository root)::

    py -3.13 -m ber.embed all --data-dir student_resource/dataset --base models/multilingual-e5-small
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from ber import schema
from ber.io import read_source, read_tsv

log = logging.getLogger("ber.embed")

ENCODER_SHARE = 0.4  # share of train Source 1 entities reserved for fine-tuning the encoder
MAX_LEN = 64


# ----------------------------------------------------------------------------- shared with ber.scale
def encoder_mask(s1_ids) -> np.ndarray:
    """True for train Source 1 entities the encoder may learn from (a fixed hash of the id)."""
    cut = int(ENCODER_SHARE * 1000)
    return np.fromiter(
        (zlib.crc32(str(i).encode()) % 1000 < cut for i in s1_ids), dtype=bool, count=len(s1_ids)
    )


def emb_paths(emb_dir: str | Path, split: str) -> dict[str, Path]:
    d = Path(emb_dir)
    return {
        "s1": d / f"{split}_s1.npy",
        "pool": d / f"{split}_pool.npy",
        "s1_ids": d / f"{split}_s1_ids.npy",
        "pool_ids": d / f"{split}_pool_ids.npy",
        "knn": d / f"{split}_knn.npz",
    }


def load_embeddings(emb_dir: str | Path, split: str, s1_ids, pool_ids):
    """(S1, pool) float16 embeddings aligned with the given id arrays."""
    p = emb_paths(emb_dir, split)
    out = []
    for key, ids in (("s1", s1_ids), ("pool", pool_ids)):
        stored = np.load(p[f"{key}_ids"])
        emb = np.load(p[key], mmap_mode="r")
        if len(stored) == len(ids) and np.array_equal(stored, np.asarray(ids, dtype=stored.dtype)):
            out.append(np.asarray(emb))
        else:
            pos = pd.Index(stored).get_indexer(ids)
            if (pos < 0).any():
                raise ValueError(f"{split} {key}: {int((pos < 0).sum())} ids have no embedding")
            out.append(np.asarray(emb)[pos])
    return out[0], out[1]


def load_knn(emb_dir: str | Path, split: str, s1_ids, pool_ids) -> dict[str, np.ndarray]:
    """kNN lists with positions mapped onto the given id arrays (fwd_* are S1 -> pool
    lists, rev_* are pool -> S1 lists, both as (S1 position, pool position, sim))."""
    p = emb_paths(emb_dir, split)
    with np.load(p["knn"]) as z:
        out = {k: z[k] for k in z.files}
    for side, ids, keys in (("s1", s1_ids, ("fwd_rows", "rev_rows")),
                            ("pool", pool_ids, ("fwd_cols", "rev_cols"))):  # fmt: skip
        stored = np.load(p[f"{side}_ids"])
        if len(stored) == len(ids) and np.array_equal(stored, np.asarray(ids, dtype=stored.dtype)):
            continue
        to_new = pd.Index(ids).get_indexer(stored)
        for k in keys:
            out[k] = to_new[out[k]]
    return out


def embedding_key(emb_dir: str | Path, split: str) -> str:
    """Changes whenever the embeddings or kNN lists of a split are rewritten."""
    parts = []
    for key in ("s1", "pool", "knn"):
        st = emb_paths(emb_dir, split)[key].stat()
        parts.append(f"{st.st_size}:{st.st_mtime_ns}")
    return "|".join(parts)


def dense_row_dot(a, b, rows, cols, chunk: int = 1_000_000, n_threads: int = 8) -> np.ndarray:
    """a[rows[i]] . b[cols[i]] in float32 for float16 embedding matrices, chunked on threads."""
    from concurrent.futures import ThreadPoolExecutor

    out = np.empty(len(rows), dtype=np.float32)

    def work(s: int) -> None:
        e = s + chunk
        x = a[rows[s:e]].astype(np.float32)
        y = b[cols[s:e]].astype(np.float32)
        out[s:e] = np.einsum("ij,ij->i", x, y)

    with ThreadPoolExecutor(n_threads) as ex:
        list(ex.map(work, range(0, len(rows), chunk)))
    return out


# ----------------------------------------------------------------------------- data
def record_texts(df: pd.DataFrame) -> list[str]:
    return [
        f"query: {n} | {a}" for n, a in zip(df[schema.NAME].tolist(), df[schema.ADDRESS].tolist())
    ]


def load_frames(data_dir: str | Path, split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(Source 1, pool) frames in the order ber.scale uses: pool = Source 2 then Source 3."""
    root = Path(data_dir) / split
    s1 = read_source(root / schema.source_file(split, 1), 1)
    pool = pd.concat(
        [read_source(root / schema.source_file(split, s), s) for s in (2, 3)], ignore_index=True
    )
    return s1, pool


def countries(df: pd.DataFrame) -> np.ndarray:
    return df[schema.COUNTRY].str.strip().str.lower().to_numpy()


def truth_pairs(data_dir: str | Path, s1: pd.DataFrame, pool: pd.DataFrame):
    """(S1 position, pool position) for every true train pair."""
    gt = read_tsv(Path(data_dir) / "train" / schema.ground_truth_file("train"))
    s1_list, pool_list = [], []
    for sid, ids in zip(gt[schema.S1_ID], gt[schema.MATCHED_IDS]):
        for pid in ids.split(","):
            if pid.strip():
                s1_list.append(sid)
                pool_list.append(pid.strip())
    a = pd.Index(s1[schema.ENTITY_ID]).get_indexer(s1_list)
    b = pd.Index(pool[schema.ENTITY_ID]).get_indexer(pool_list)
    ok = (a >= 0) & (b >= 0)
    return a[ok].astype(np.int64), b[ok].astype(np.int64)


# ----------------------------------------------------------------------------- model
def _device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _pool(hidden, mask):
    import torch

    m = mask.unsqueeze(-1).to(hidden.dtype)
    v = (hidden * m).sum(1) / m.sum(1).clamp(min=1)
    return torch.nn.functional.normalize(v.float(), dim=-1)


def _batches_by_length(texts: list[str], batch: int):
    order = np.argsort([len(t) for t in texts], kind="stable")
    for s in range(0, len(order), batch):
        yield order[s : s + batch]


def encode(model_dir: str | Path, texts: list[str], batch: int = 2048, max_len: int = MAX_LEN):
    """L2-normalised float16 embeddings, one row per text (length-sorted batches)."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    dev = _device()
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir, dtype=torch.float16).to(dev).eval()
    out = np.empty((len(texts), model.config.hidden_size), dtype=np.float16)
    t0, done = time.time(), 0
    with torch.inference_mode():
        for idx in _batches_by_length(texts, batch):
            enc = tok([texts[i] for i in idx], padding=True, truncation=True, max_length=max_len,
                      return_tensors="pt").to(dev)  # fmt: skip
            h = model(**enc).last_hidden_state
            out[idx] = _pool(h, enc["attention_mask"]).half().cpu().numpy()
            done += len(idx)
            if done % (batch * 500) < batch:
                log.info("encoded %d/%d (%.0f/s)", done, len(texts), done / (time.time() - t0))
    log.info("encoded %d texts in %.0fs", len(texts), time.time() - t0)
    return out


def finetune(
    data_dir: str | Path,
    base: str | Path,
    out_dir: str | Path,
    epochs: float = 1.0,
    batch: int = 1024,
    lr: float = 5e-5,
    scale: float = 20.0,
    hard_negatives: str | Path | None = None,
    seed: int = 42,
) -> Path:
    """Symmetric in-batch contrastive fine-tuning on encoder-share train pairs.

    Each batch holds pairs from one country (so negatives are not trivially
    told apart by country). With a blocking cache, each query also brings its
    highest-scoring non-matching candidate as a hard negative. Logits between
    records of the same entity (other than the target) are masked.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    s1, pool = load_frames(data_dir, "train")
    a, b = truth_pairs(data_dir, s1, pool)
    owner = np.full(len(pool), -1, dtype=np.int64)
    owner[b] = a
    enc_ent = encoder_mask(s1[schema.ENTITY_ID].to_numpy())
    keep = enc_ent[a]
    a, b = a[keep], b[keep]
    log.info("fine-tuning on %d pairs of %d entities", len(a), len(np.unique(a)))

    hard = np.full(len(s1), -1, dtype=np.int64)
    if hard_negatives and Path(hard_negatives).exists():
        import joblib

        c = joblib.load(hard_negatives)
        r, col, sc = c["rows"].astype(np.int64), c["cols"].astype(np.int64), c["base"]["score"]
        neg = owner[col] != r
        r, col, sc = r[neg], col[neg], sc[neg]
        order = np.lexsort((-sc, r))
        first = np.ones(len(order), bool)
        first[1:] = r[order][1:] != r[order][:-1]
        hard[r[order][first]] = col[order][first]
        log.info("hard negatives for %d entities", int((hard >= 0).sum()))

    s1_text = record_texts(s1)
    pool_text = record_texts(pool)
    country = countries(s1)[a]
    # Hard in-batch negatives: sort by country, area (the last address component: state,
    # city or PIN) and then name, so a batch holds same-area businesses with similar names.
    # Random batches are too easy (the loss reaches ~0.01 within 300 steps).
    region = np.array(
        [x.rsplit(",", 1)[-1].strip().lower() for x in s1[schema.ADDRESS].to_numpy()[a]]
    )
    name = s1[schema.NAME].str.lower().to_numpy()[a]

    # batches from one country each, area- and name-sorted, in random order
    perm = rng.permutation(len(a))
    perm = perm[np.lexsort((name[perm], region[perm], country[perm]))]
    batches = []
    for c in np.unique(country):
        idx = perm[country[perm] == c]
        batches += [idx[s : s + batch] for s in range(0, len(idx) - batch // 2, batch)]
    n_steps = math.ceil(len(batches) * epochs)
    dev = _device()
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModel.from_pretrained(base).to(dev).train()
    model.gradient_checkpointing_enable()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    warm = max(1, n_steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warm, max(0.0, (n_steps - s) / max(1, n_steps - warm)))
    )

    def embed(texts):
        enc = tok(texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(
            dev
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = model(**enc).last_hidden_state
        return _pool(h, enc["attention_mask"])

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        model.save_pretrained(out)
        tok.save_pretrained(out)

    t0 = time.time()
    step = skipped = 0
    while step < n_steps:
        for bi in rng.permutation(len(batches)):
            if step >= n_steps:
                break
            idx = batches[bi]
            qa, pb = a[idx], b[idx]
            hn = hard[qa]
            hn = hn[hn >= 0]
            cols = np.concatenate([pb, hn])
            try:
                q = embed([s1_text[i] for i in qa])
                d = embed([pool_text[i] for i in cols])
                # column j belongs to entity owner[cols[j]]; mask same-entity columns but the target
                ent_q = torch.as_tensor(qa, device=dev)
                ent_d = torch.as_tensor(owner[cols], device=dev)
                n = len(idx)
                target = torch.arange(n, device=dev)
                logits = scale * q @ d.T
                same = ent_q[:, None] == ent_d[None, :]
                same[target, target] = False
                logits = logits.masked_fill(same, -1e4)
                loss_q = torch.nn.functional.cross_entropy(logits, target)
                logits_d = scale * d[:n] @ q.T
                same_d = ent_d[:n, None] == ent_q[None, :]
                same_d[target, target] = False
                loss_d = torch.nn.functional.cross_entropy(
                    logits_d.masked_fill(same_d, -1e4), target
                )
                loss = (loss_q + loss_d) / 2
                opt.zero_grad(set_to_none=True)
                loss.backward()
            except torch.OutOfMemoryError:  # a batch of unusually long texts: skip it
                opt.zero_grad(set_to_none=True)
                q = d = logits = logits_d = loss = None
                torch.cuda.empty_cache()
                skipped += 1
                log.warning("out of GPU memory on a batch; skipped (%d so far)", skipped)
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 50 == 0 or step == 1:
                log.info("step %d/%d loss %.4f (%.1f steps/s)", step, n_steps, loss.item(),
                         step / (time.time() - t0))  # fmt: skip
            if step % 250 == 0:
                save()  # a crash later still leaves a usable encoder
    save()
    (out / "finetune.json").write_text(
        json.dumps(
            {
                "base": str(base),
                "pairs": len(a),
                "steps": n_steps,
                "batch": batch,
                "lr": lr,
                "scale": scale,
                "encoder_share": ENCODER_SHARE,
                "hard_negatives": bool((hard >= 0).any()),
                "skipped_batches": skipped,
                "seconds": round(time.time() - t0),
            },
            indent=2,
        )
    )
    log.info("fine-tuned encoder saved to %s (%.0fs)", out, time.time() - t0)
    return out


# ----------------------------------------------------------------------------- kNN
def knn(q: np.ndarray, d: np.ndarray, q_country: np.ndarray, d_country: np.ndarray, k: int,
        q_chunk: int = 1024, d_block: int = 500_000):  # fmt: skip
    """Exact top-k inner product per query within its country (records with an empty
    country are searchable from every country). Returns (rows, cols, sims)."""
    import torch

    dev = _device()
    rows, cols, sims = [], [], []
    for c in np.unique(q_country):
        qi = np.flatnonzero(q_country == c)
        di = np.flatnonzero((d_country == c) | (d_country == ""))
        if len(di) == 0:
            continue
        kk = min(k, len(di))
        blocks = [
            torch.from_numpy(d[di[s : s + d_block]]).to(dev) for s in range(0, len(di), d_block)
        ]
        offs = list(range(0, len(di), d_block))
        for s in range(0, len(qi), q_chunk):
            qb = torch.from_numpy(q[qi[s : s + q_chunk]]).to(dev)
            vals, idxs = [], []
            for off, db in zip(offs, blocks):
                v, i = torch.topk(qb @ db.T, min(kk, db.shape[0]), dim=1)
                vals.append(v.float())
                idxs.append(i + off)
            v, j = torch.topk(torch.cat(vals, 1), kk, dim=1)
            i = torch.gather(torch.cat(idxs, 1), 1, j)
            rows.append(np.repeat(qi[s : s + q_chunk], kk))
            cols.append(di[i.cpu().numpy().ravel()])
            sims.append(v.cpu().numpy().ravel().astype(np.float16))
        del blocks
        torch.cuda.empty_cache()
        log.info("knn %s: %d queries x %d records", c, len(qi), len(di))
    if not rows:
        return np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.float16)
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(sims)


def recall_report(data_dir, s1, pool, rows, cols, only_heldout: bool = True) -> dict[str, float]:
    """Pair recall of forward kNN lists at several depths, on entities the encoder never saw."""
    a, b = truth_pairs(data_dir, s1, pool)
    if only_heldout:
        held = ~encoder_mask(s1[schema.ENTITY_ID].to_numpy())
        keep = held[a]
        a, b = a[keep], b[keep]
    n_pool = len(pool)
    true_key = a * n_pool + b
    k = int(np.bincount(rows).max()) if len(rows) else 0
    rank = np.arange(len(rows)) - np.searchsorted(rows, rows)
    out = {}
    for depth in (1, 5, 10, 20, 30, 50, 100):
        if depth > k:
            break
        sel = rank < depth
        found = np.isin(true_key, rows[sel].astype(np.int64) * n_pool + cols[sel])
        out[f"recall@{depth}"] = float(found.mean())
    return out


# ----------------------------------------------------------------------------- driver
def run_split(data_dir, split: str, model_dir, emb_dir, k: int, k_reverse: int) -> dict:
    p = emb_paths(emb_dir, split)
    Path(emb_dir).mkdir(parents=True, exist_ok=True)
    s1, pool = load_frames(data_dir, split)
    if not (p["s1"].exists() and p["pool"].exists()):
        for key, df in (("s1", s1), ("pool", pool)):
            e = encode(model_dir, record_texts(df))
            np.save(p[key], e)
            np.save(p[f"{key}_ids"], df[schema.ENTITY_ID].to_numpy().astype("U"))
    q = np.load(p["s1"])
    d = np.load(p["pool"])
    qc, dc = countries(s1), countries(pool)
    t0 = time.time()
    fr, fc, fs = knn(q, d, qc, dc, k)
    rc, rr, rs = knn(d, q, dc, qc, k_reverse) if k_reverse > 0 else (np.empty(0, int),) * 3
    np.savez(p["knn"], fwd_rows=fr.astype(np.int32), fwd_cols=fc.astype(np.int32), fwd_sim=fs,
             rev_rows=rr.astype(np.int32), rev_cols=rc.astype(np.int32), rev_sim=rs)  # fmt: skip
    report = {"split": split, "knn_seconds": round(time.time() - t0)}
    if split == "train":
        order = np.lexsort((-fs.astype(np.float32), fr))
        report.update(recall_report(data_dir, s1, pool, fr[order], fc[order]))
    log.info("%s", report)
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ber.embed")
    ap.add_argument("command", choices=["finetune", "encode", "all"])
    ap.add_argument("--data-dir", default="student_resource/dataset")
    ap.add_argument("--base", default="models/multilingual-e5-small")
    ap.add_argument("--model", default="models/e5-small-ft")
    ap.add_argument("--emb-dir", default="artifacts/emb")
    ap.add_argument("--splits", default="train,test")
    ap.add_argument("--hard-negatives", default="")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--k-reverse", type=int, default=3)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname).1s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")  # fmt: skip
    if args.command in ("finetune", "all"):
        finetune(args.data_dir, args.base, args.model, args.epochs, args.batch, args.lr,
                 hard_negatives=args.hard_negatives or None)  # fmt: skip
    if args.command in ("encode", "all"):
        reports = [run_split(args.data_dir, s, args.model, args.emb_dir, args.k, args.k_reverse)
                   for s in args.splits.split(",")]  # fmt: skip
        Path(args.emb_dir, "report.json").write_text(json.dumps(reports, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
