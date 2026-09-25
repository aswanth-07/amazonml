"""Listwise group reranker: one transformer pass over a Source 1 record and all its
uncertain candidates, so every candidate is judged in the context of the others.

Why. Every other model here sees one candidate at a time (the GBM features, the bi-encoder,
the pairwise cross-encoder). The errors left at ~0.98 are group-level: a neighbour at 471
among three copies that agree on 450; a namesake whose copies agree with each other but not
with the Source 1 record; a Source 1 record whose own house number is the typo, which the
copies outvote. Reading ``[CLS] S1 [SEP] c1 [SEP] c2 [SEP] ...`` lets each candidate attend
to every other one, so the consensus is learned instead of hand-built.

It is content-only: no probability or pool-side competition goes in. Its signal does not
change when a candidate's true owner is missing from Source 1 (test's orphaned copies),
which is where the pool-side features mislead stage 2.

Groups. An entity's candidates with stage-1 probability >= ``scale.ce_min_p1`` (the pairs
the cross-encoder scores), at most ``k`` per group in random order. An entity with more
gets overflow groups: its ``k // 2`` most probable candidates as anchors plus the next
chunk, so every pair is scored next to the entity's likeliest copies.

Leakage. The model trains on the *listwise share* (``scale.lw_share``): eligible entities
outside the stage-1/2 sample, whose stage-1 probabilities and cross-encoder scores are
out-of-sample exactly as on test. ``ber.scale`` removes the share from the holdout.

Needs PyTorch and transformers (the torch interpreter, like ``ber.crossenc``)::

    python -m ber.listwise train --groups runs/<run>/lw_groups_train.npz --init models/ce-small-v2 --model models/lw-small
    python -m ber.listwise eval  --groups runs/<run>/lw_groups_train.npz --model models/lw-small
    python -m ber.listwise score --model models/lw-small --split test --pairs pairs.npz --out scores.npy
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ber import schema

log = logging.getLogger("ber.listwise")

K = 16  # candidates per group
MAX_LEN = 512
LEN_S1 = 48  # tokens kept of the Source 1 record
LEN_CAND = 32  # tokens kept of each candidate (fewer when a large group must fit MAX_LEN)
THRESHOLDS = (0.002, 0.005, 0.01, 0.02, 0.02, 0.02, 0.05)  # training: p1 floor drawn per group


def texts(df) -> list[str]:
    return [f"{n} | {a}" for n, a in zip(df[schema.NAME].tolist(), df[schema.ADDRESS].tolist())]


# ----------------------------------------------------------------------------- groups (numpy only)
@dataclass
class Group:
    s1: int  # Source 1 position
    idx: np.ndarray  # pair positions in the caller's arrays, in the order the model reads them
    primary: np.ndarray  # bool: this group's score is the pair's score (False: anchor)


def build_groups(
    rows: np.ndarray, p1: np.ndarray, k: int = K, rng: np.random.Generator | None = None
) -> list[Group]:
    """Every pair in exactly one group as primary. Per entity, pairs by descending p1: the
    first ``k`` form one group; the rest come in chunks of ``k - k // 2`` behind the top
    ``k // 2`` as anchors. With ``rng`` each group's reading order is shuffled."""
    if len(rows) == 0:
        return []
    anchors = k // 2
    order = np.lexsort((-p1, rows))
    r = rows[order]
    starts = np.flatnonzero(np.r_[True, r[1:] != r[:-1]])
    ends = np.r_[starts[1:], len(r)]
    out: list[Group] = []
    for s, e in zip(starts.tolist(), ends.tolist()):
        idx = order[s:e]
        s1 = int(r[s])
        out.append(Group(s1, idx[:k], np.ones(min(k, len(idx)), dtype=bool)))
        rest = idx[k:]
        for t in range(0, len(rest), k - anchors):
            chunk = rest[t : t + k - anchors]
            prim = np.r_[np.zeros(anchors, dtype=bool), np.ones(len(chunk), dtype=bool)]
            out.append(Group(s1, np.r_[idx[:anchors], chunk], prim))
    if rng is not None:
        for g in out:
            perm = rng.permutation(len(g.idx))
            g.idx, g.primary = g.idx[perm], g.primary[perm]
    return out


def training_groups(
    rows: np.ndarray,
    p1: np.ndarray,
    entities: np.ndarray,
    rng: np.random.Generator,
    k: int = K,
    thresholds: tuple[float, ...] = THRESHOLDS,
    p_overflow: float = 0.3,
    labels: np.ndarray | None = None,
    drop_owner: float = 0.0,
) -> list[Group]:
    """Groups for one epoch over the entities in the boolean mask ``entities``: each entity
    draws its own p1 floor (so the model sees small and large groups), keeps its first
    group and, with probability ``p_overflow``, one of its overflow groups. With
    ``drop_owner`` (pool side), that share of groups loses its true pairs, as a pool record
    whose owner is missing from Source 1 (test's orphans) must be answered "none of these"."""
    floor = np.asarray(thresholds, dtype=np.float32)[
        rng.integers(0, len(thresholds), len(entities))
    ]
    keep = np.flatnonzero(entities[rows] & (p1 >= floor[rows]))
    groups = build_groups(rows[keep], p1[keep], k, rng)
    out: list[Group] = []
    i = 0
    while i < len(groups):
        j = i + 1
        while j < len(groups) and groups[j].s1 == groups[i].s1:
            j += 1
        out.append(groups[i])
        if j - i > 1 and rng.random() < p_overflow:
            out.append(groups[int(rng.integers(i + 1, j))])
        i = j
    for g in out:
        g.idx = keep[g.idx]
    if drop_owner > 0 and labels is not None:
        kept = []
        for g in out:
            if rng.random() < drop_owner:
                wrong = labels[g.idx] == 0
                g.idx, g.primary = g.idx[wrong], g.primary[wrong]
            if len(g.idx):
                kept.append(g)
        out = kept
    return out


class TokenStore:
    """Token ids of many texts in one flat int32 array (a Python list per text would need
    ~36 bytes per id)."""

    def __init__(self, tok, items: list[str], max_tokens: int, batch: int = 100_000):
        ids, lens = [], []
        for s in range(0, len(items), batch):
            enc = tok(items[s : s + batch], add_special_tokens=False, truncation=True,
                      max_length=max_tokens)["input_ids"]  # fmt: skip
            lens.append(np.fromiter((len(x) for x in enc), np.int64, len(enc)))
            ids.append(np.fromiter((t for x in enc for t in x), np.int32))
        self.ids = np.concatenate(ids) if ids else np.empty(0, np.int32)
        ln = np.concatenate(lens) if lens else np.empty(0, np.int64)
        self.start = np.r_[0, np.cumsum(ln)[:-1]].astype(np.int64)
        self.len = ln

    def get(self, i: int) -> np.ndarray:
        return self.ids[self.start[i] : self.start[i] + self.len[i]]


@dataclass
class Encoded:
    """What one group becomes: token ids, the slot of every token (0 special, 1 the Source 1
    record, 2 + j candidate j), and each candidate's pair position."""

    ids: np.ndarray
    seg: np.ndarray
    idx: np.ndarray
    primary: np.ndarray


def encode_group(
    g: Group,
    s1_tokens: np.ndarray,
    cand_tokens: list[np.ndarray],
    cls: int,
    sep: int,
    max_len: int = MAX_LEN,
    len_s1: int = LEN_S1,
    len_cand: int = LEN_CAND,
) -> Encoded:
    q = s1_tokens[:len_s1]
    n = len(cand_tokens)
    per = max(1, min(len_cand, (max_len - 2 - len(q)) // max(n, 1) - 1))
    parts, segs = [np.array([cls], np.int32), q, np.array([sep], np.int32)], [
        np.zeros(1, np.int32), np.ones(len(q), np.int32), np.zeros(1, np.int32)
    ]  # fmt: skip
    for j, c in enumerate(cand_tokens):
        c = c[:per]
        parts += [c, np.array([sep], np.int32)]
        segs += [np.full(len(c), j + 2, np.int32), np.zeros(1, np.int32)]
    return Encoded(np.concatenate(parts), np.concatenate(segs), g.idx, g.primary)


# ----------------------------------------------------------------------------- model (torch)
def _device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _autocast_dtype():
    """bfloat16 where the GPU has it, else float16; ``BER_LISTWISE_AMP`` (bf16, fp16, off)
    overrides, e.g. to test the float16 path or to run in full precision."""
    import torch

    if not torch.cuda.is_available():
        return None
    force = os.environ.get("BER_LISTWISE_AMP", "").lower()
    if force in ("bf16", "fp16", "off"):
        return {"bf16": torch.bfloat16, "fp16": torch.float16, "off": None}[force]
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def _model(init: str | Path, heads: Path | None = None, grad_ckpt: bool = False):
    import torch
    from transformers import AutoModel

    class Listwise(torch.nn.Module):
        """Encoder + a pair head on each candidate's mean-pooled span next to the Source 1
        span, + a group head (P(any candidate matches)) on [CLS] and the Source 1 span."""

        def __init__(self):
            super().__init__()
            self.encoder = AutoModel.from_pretrained(init)
            if grad_ckpt:
                self.encoder.gradient_checkpointing_enable()
            d = self.encoder.config.hidden_size
            self.pair_head = torch.nn.Sequential(
                torch.nn.Linear(4 * d, d), torch.nn.GELU(), torch.nn.Linear(d, 1)
            )
            self.group_head = torch.nn.Sequential(
                torch.nn.Linear(2 * d, d), torch.nn.GELU(), torch.nn.Linear(d, 1)
            )

        def forward(self, input_ids, attention_mask, seg, n_cand: int):
            h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            slots = torch.arange(1, n_cand + 2, device=seg.device)
            m = (seg.unsqueeze(1) == slots.view(1, -1, 1)).to(h.dtype)  # B, 1 + K, T
            pooled = torch.bmm(m, h) / m.sum(-1, keepdim=True).clamp(min=1)
            q, c = pooled[:, :1], pooled[:, 1:]
            x = torch.cat([c, q.expand_as(c), c * q, (c - q).abs()], dim=-1)
            pair = self.pair_head(x).squeeze(-1)
            group = self.group_head(torch.cat([h[:, 0], q[:, 0]], dim=-1)).squeeze(-1)
            return pair.float(), group.float()

    model = Listwise()
    if heads is not None:
        state = torch.load(heads, map_location="cpu")
        model.pair_head.load_state_dict(state["pair_head"])
        model.group_head.load_state_dict(state["group_head"])
    return model


def _special(tok) -> tuple[int, int, int]:
    cls = tok.cls_token_id if tok.cls_token_id is not None else tok.bos_token_id
    sep = tok.sep_token_id if tok.sep_token_id is not None else tok.eos_token_id
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    return int(cls), int(sep), int(pad)


def collate(batch: list[Encoded], pad: int, labels: np.ndarray | None = None):
    """Tensors for a list of encoded groups: input ids, attention mask, slots, and per
    candidate its pair position (-1 where a group has fewer candidates), primary flag and,
    with ``labels``, its label."""
    import torch

    t = max(len(e.ids) for e in batch)
    n = max(len(e.idx) for e in batch)
    ids = np.full((len(batch), t), pad, np.int64)
    att = np.zeros((len(batch), t), np.int64)
    seg = np.zeros((len(batch), t), np.int64)
    pos = np.full((len(batch), n), -1, np.int64)
    prim = np.zeros((len(batch), n), bool)
    for i, e in enumerate(batch):
        ids[i, : len(e.ids)] = e.ids
        att[i, : len(e.ids)] = 1
        seg[i, : len(e.seg)] = e.seg
        pos[i, : len(e.idx)] = e.idx
        prim[i, : len(e.idx)] = e.primary
    out = {
        "input_ids": torch.from_numpy(ids),
        "attention_mask": torch.from_numpy(att),
        "seg": torch.from_numpy(seg),
        "n_cand": n,
        "pos": pos,
        "primary": prim,
    }
    if labels is not None:
        y = np.zeros((len(batch), n), np.float32)
        y[pos >= 0] = labels[pos[pos >= 0]]
        out["y"] = torch.from_numpy(y)
        out["valid"] = torch.from_numpy(pos >= 0)
    return out


class _Encoder:
    """Turns groups into ``Encoded`` items; usable as a map-style dataset. The query of a
    group is the record at ``g.s1`` in ``q_store``; its candidates are the records at
    ``cand[g.idx]`` in ``c_store``. Source 1 side: S1 query, pool candidates. Pool side
    (C7): pool query, S1 suitors."""

    def __init__(self, groups, cand, q_store, q_slot, c_store, c_slot, special, max_len,
                 len_s1, len_cand):  # fmt: skip
        self.groups, self.cand = groups, cand
        self.q_store, self.q_slot = q_store, q_slot
        self.c_store, self.c_slot = c_store, c_slot
        self.cls, self.sep, _ = special
        self.max_len, self.len_s1, self.len_cand = max_len, len_s1, len_cand

    @classmethod
    def for_side(cls, side, groups, rows, cols, s1_store, s1_slot, pool_store, pool_slot,
                 special, max_len, len_s1, len_cand):  # fmt: skip
        if side == "pool":
            return cls(groups, rows, pool_store, pool_slot, s1_store, s1_slot, special,
                       max_len, len_s1, len_cand)  # fmt: skip
        return cls(groups, cols, s1_store, s1_slot, pool_store, pool_slot, special, max_len,
                   len_s1, len_cand)  # fmt: skip

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, i: int) -> Encoded:
        g = self.groups[i]
        q = self.q_store.get(int(self.q_slot[g.s1]))
        cands = [self.c_store.get(int(self.c_slot[c])) for c in self.cand[g.idx]]
        return encode_group(g, q, cands, self.cls, self.sep, self.max_len, self.len_s1,
                            self.len_cand)  # fmt: skip

    def approx_len(self, i: int) -> int:
        g = self.groups[i]
        n = int(self.q_store.len[self.q_slot[g.s1]]) + int(
            self.c_store.len[self.c_slot[self.cand[g.idx]]].sum()
        )
        return min(n + len(g.idx) + 2, self.max_len)


def keys_of(side: str, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """The positions groups are keyed on: Source 1 records, or pool records (side "pool")."""
    return cols if side == "pool" else rows


def texts_at(data_dir, split: str, need1: np.ndarray, needp: np.ndarray):
    """("name | address" of the sorted Source 1 positions ``need1``, the same of the sorted
    pool positions ``needp``, number of S1 records, number of pool records), read in chunks
    so memory holds only the needed texts (a full train frame is ~5 GB). Parsing matches
    ``ber.io.read_tsv``; pool positions run over Source 2 then Source 3, as in ber.scale."""
    import csv

    import pandas as pd

    read = {"sep": "\t", "dtype": str, "keep_default_na": False, "na_filter": False,
            "quoting": csv.QUOTE_NONE, "encoding": "utf-8-sig", "chunksize": 500_000,
            "usecols": [schema.NAME, schema.ADDRESS]}  # fmt: skip
    root = Path(data_dir) / split

    def collect(files: list[Path], positions: np.ndarray):
        out: list[str] = []
        offset = 0
        for f in files:
            for chunk in pd.read_csv(f, **read):
                hi = offset + len(chunk)
                sel = positions[(positions >= offset) & (positions < hi)] - offset
                if len(sel):
                    n = chunk[schema.NAME].to_numpy()[sel]
                    a = chunk[schema.ADDRESS].to_numpy()[sel]
                    out += [f"{x.strip()} | {y.strip()}" for x, y in zip(n, a)]
                offset = hi
        return out, offset

    s1_text, n1 = collect([root / schema.source_file(split, 1)], need1)
    pool_text, npool = collect([root / schema.source_file(split, s) for s in (2, 3)], needp)
    return s1_text, pool_text, n1, npool


def _stores(tok, data_dir, split, rows, cols, len_s1, len_cand):
    """Token stores for the Source 1 and pool records these pairs use, and position -> slot maps."""
    t0 = time.time()
    need1, needp = np.unique(rows), np.unique(cols)
    s1_text, pool_text, n1, npool = texts_at(data_dir, split, need1, needp)
    s1_store = TokenStore(tok, s1_text, len_s1)
    pool_store = TokenStore(tok, pool_text, len_cand)
    s1_slot = np.full(n1, -1, np.int64)
    s1_slot[need1] = np.arange(len(need1))
    pool_slot = np.full(npool, -1, np.int64)
    pool_slot[needp] = np.arange(len(needp))
    log.info("tokenised %d Source 1 and %d pool records in %.0fs", len(need1), len(needp),
             time.time() - t0)  # fmt: skip
    return s1_store, s1_slot, pool_store, pool_slot


def _length_batches(enc: _Encoder, batch: int, rng: np.random.Generator | None):
    """Batches of similar length. Training (``rng``): shuffled, bucketed in windows of 50
    batches, batches in random order. Scoring: sorted by length."""
    n = len(enc)
    lens = np.fromiter((enc.approx_len(i) for i in range(n)), np.int64, n)
    if rng is None:
        order = np.argsort(lens, kind="stable")
        return [order[s : s + batch] for s in range(0, n, batch)]
    order = rng.permutation(n)
    window = batch * 50
    out = []
    for s in range(0, n, window):
        w = order[s : s + window]
        w = w[np.argsort(lens[w], kind="stable")]
        out += [w[t : t + batch] for t in range(0, len(w), batch)]
    return [out[i] for i in rng.permutation(len(out))]


def _loader(enc: _Encoder, batches, pad: int, labels, workers: int):
    import torch

    if os.name == "nt":  # spawned workers cannot pickle this local dataset; Linux forks
        workers = 0

    class Batches(torch.utils.data.Dataset):
        def __len__(self):
            return len(batches)

        def __getitem__(self, i):
            return collate([enc[int(j)] for j in batches[i]], pad, labels)

    return torch.utils.data.DataLoader(
        Batches(), batch_size=None, shuffle=False, num_workers=workers,
        persistent_workers=False, prefetch_factor=4 if workers > 0 else None,
    )  # fmt: skip


# ----------------------------------------------------------------------------- train
def train(
    data_dir: str | Path,
    init: str | Path,
    groups_path: str | Path,
    out_dir: str | Path,
    k: int = K,
    max_len: int = MAX_LEN,
    batch: int = 32,
    lr: float = 3e-5,
    head_lr_mult: float = 10.0,
    epochs: float = 1.0,
    group_weight: float = 0.3,
    max_entities: int = 0,
    n_valid: int = 20_000,
    eval_every: int = 1000,
    grad_ckpt: bool = False,
    workers: int = 4,
    seed: int = 42,
    side: str | None = None,
    drop_owner: float | None = None,
) -> Path:
    """Fine-tune on the groups of ``groups_path`` (``lw_groups_train.npz`` from ``ber.scale``,
    or ``scripts/make_lw_groups.py``). Loss: BCE per candidate + ``group_weight`` x BCE of
    the group head on "some candidate in the group matches". Validation log-loss on up to
    ``n_valid`` held-out keys every ``eval_every`` steps.

    ``side`` (default: the groups file's, else "s1"): "s1" reads a Source 1 record with its
    candidates; "pool" (C7) reads a pool record with its Source 1 suitors, and
    ``drop_owner`` (default 0.2 there) of its groups lose their true owner, so the model
    learns to answer "none of these" for test's orphaned copies."""
    import torch
    from transformers import AutoTokenizer

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    z = np.load(groups_path)
    rows, cols, p1 = z["rows"].astype(np.int64), z["cols"].astype(np.int64), z["p1"]
    labels = z["labels"].astype(np.float32)
    side = side or (str(z["side"]) if "side" in z.files else "s1")
    if drop_owner is None:
        drop_owner = 0.2 if side == "pool" else 0.0
    keys = keys_of(side, rows, cols)
    ents = z["train"].astype(bool)
    if max_entities and ents.sum() > max_entities:
        keep = rng.choice(np.flatnonzero(ents), max_entities, replace=False)
        ents = np.zeros_like(ents)
        ents[keep] = True
    valid = np.zeros_like(ents)
    held = np.flatnonzero(z["valid"].astype(bool) & np.isin(np.arange(len(ents)), keys))
    if n_valid and len(held):
        valid[rng.choice(held, min(n_valid, len(held)), replace=False)] = True
    use = (ents | valid)[keys]
    log.info("listwise training (%s side): %d keys, %d pairs (%.1f%% positive); %d validation keys",
             side, int(ents.sum()), int((ents[keys]).sum()), 100 * labels[ents[keys]].mean(),
             int(valid.sum()))  # fmt: skip

    dev = _device()
    tok = AutoTokenizer.from_pretrained(init)
    special = _special(tok)
    stores = _stores(tok, data_dir, "train", rows[use], cols[use], LEN_S1, LEN_CAND)
    vsel = np.flatnonzero(valid[keys] & (p1 >= 0.02))
    v_enc = _Encoder.for_side(
        side, [Group(g.s1, vsel[g.idx], g.primary) for g in build_groups(keys[vsel], p1[vsel], k, np.random.default_rng(0))],
        rows, cols, *stores, special, max_len, LEN_S1, LEN_CAND,
    )  # fmt: skip

    model = _model(init, grad_ckpt=grad_ckpt).to(dev).train()
    enc_params = list(model.encoder.parameters())
    head_params = list(model.pair_head.parameters()) + list(model.group_head.parameters())
    opt = torch.optim.AdamW(
        [{"params": enc_params, "lr": lr}, {"params": head_params, "lr": lr * head_lr_mult}],
        weight_decay=0.01,
    )
    per_epoch = math.ceil(
        sum(1 for _ in training_groups(keys, p1, ents, np.random.default_rng(seed), k)) / batch
    )
    n_steps = max(1, math.ceil(per_epoch * epochs))
    warm = max(1, n_steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warm, max(0.0, (n_steps - s) / max(1, n_steps - warm)))
    )
    amp = _autocast_dtype()
    # float16 (T4, P100: no bfloat16) needs loss scaling or small gradients underflow to zero
    scaler = torch.amp.GradScaler("cuda", enabled=amp == torch.float16)
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def loss_of(b):
        with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
            pair, group = model(b["input_ids"].to(dev), b["attention_mask"].to(dev),
                                b["seg"].to(dev), b["n_cand"])  # fmt: skip
        y, m = b["y"].to(dev), b["valid"].to(dev)
        lp = bce(pair[m], y[m])
        yg = (y * m).amax(1)
        return lp + group_weight * bce(group, yg), lp

    def validate() -> float:
        if not len(v_enc):
            return float("nan")
        model.eval()
        tot, n = 0.0, 0
        with torch.inference_mode():
            for b in _loader(v_enc, _length_batches(v_enc, batch * 2, None), special[2], labels, 0):
                _, lp = loss_of(b)
                c = int(b["valid"].sum())
                tot, n = tot + lp.item() * c, n + c
        model.train()
        return tot / max(n, 1)

    def save(meta: dict) -> None:
        model.encoder.save_pretrained(out)
        tok.save_pretrained(out)
        torch.save({"pair_head": model.pair_head.state_dict(),
                    "group_head": model.group_head.state_dict()}, out / "heads.pt")  # fmt: skip
        (out / "listwise.json").write_text(json.dumps(meta, indent=2))

    meta = {"init": str(init), "side": side, "drop_owner": drop_owner, "k": k, "max_len": max_len,
            "len_s1": LEN_S1, "len_cand": LEN_CAND,
            "batch": batch, "lr": lr, "epochs": epochs, "steps": n_steps,
            "entities": int(ents.sum()), "groups_file": str(groups_path)}  # fmt: skip
    log.info("validation log-loss before training: %.4f", validate())
    t0 = time.time()
    step, epoch, skipped = 0, 0, 0
    while step < n_steps:
        groups = training_groups(keys, p1, ents, np.random.default_rng(seed + epoch), k,
                                 labels=labels, drop_owner=drop_owner)  # fmt: skip
        enc = _Encoder.for_side(side, groups, rows, cols, *stores, special, max_len, LEN_S1,
                                LEN_CAND)  # fmt: skip
        for b in _loader(enc, _length_batches(enc, batch, rng), special[2], labels, workers):
            if step >= n_steps:
                break
            try:
                loss, lp = loss_of(b)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
            except torch.OutOfMemoryError:
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                skipped += 1
                log.warning("out of GPU memory on a batch; skipped (%d so far)", skipped)
                continue
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            if step % 100 == 0 or step == 1:
                log.info("step %d/%d loss %.4f pair %.4f (%.1f steps/s)", step, n_steps,
                         loss.item(), lp.item(), step / (time.time() - t0))  # fmt: skip
            if step % eval_every == 0:
                log.info("step %d validation log-loss %.4f", step, validate())
                save({**meta, "partial_step": step})
        epoch += 1
    meta.update({"valid_logloss": validate(), "skipped_batches": skipped,
                 "seconds": round(time.time() - t0)})  # fmt: skip
    save(meta)
    # written last: ber.scale waits for this file and reads the kind to pick the scorer
    (out / "crossenc.json").write_text(json.dumps({"kind": "listwise", **meta}, indent=2))
    log.info("listwise model saved to %s (%.0fs)", out, time.time() - t0)
    return out


# ----------------------------------------------------------------------------- score
def score(
    model_dir: str | Path,
    data_dir: str | Path,
    split: str,
    rows: np.ndarray,
    cols: np.ndarray,
    p1: np.ndarray,
    batch: int = 64,
    workers: int = 4,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """(P(match) per pair, P(some candidate in the pair's group matches) per pair), as
    float16 in the given order. Pairs are grouped by ``p1`` as in training: per Source 1
    entity, or per pool record for a pool-side model (C7), whose group probability is then
    P(some suitor owns this record)."""
    import torch
    from transformers import AutoTokenizer

    model_dir = Path(model_dir)
    meta = json.loads((model_dir / "listwise.json").read_text())
    k, max_len = int(meta["k"]), int(meta["max_len"])
    len_s1, len_cand = int(meta["len_s1"]), int(meta["len_cand"])
    side = meta.get("side", "s1")
    rows, cols = rows.astype(np.int64), cols.astype(np.int64)
    keys = keys_of(side, rows, cols)
    p1 = np.asarray(p1, dtype=np.float32)
    dev = _device()
    tok = AutoTokenizer.from_pretrained(model_dir)
    special = _special(tok)
    groups = build_groups(keys, p1, k, np.random.default_rng(seed))
    stores = _stores(tok, data_dir, split, rows, cols, len_s1, len_cand)
    enc = _Encoder.for_side(side, groups, rows, cols, *stores, special, max_len, len_s1, len_cand)
    model = _model(model_dir, model_dir / "heads.pt").to(dev).eval()
    amp = _autocast_dtype()
    if amp is not None:
        model = model.to(amp)
    out = np.full(len(rows), np.nan, np.float32)
    first_group = np.full(int(keys.max()) + 1 if len(keys) else 0, np.nan, np.float32)
    s1_of = np.fromiter((g.s1 for g in groups), np.int64, len(groups))
    is_first = np.r_[True, s1_of[1:] != s1_of[:-1]].astype(bool)[: len(groups)]
    t0, done = time.time(), 0

    def run(gi: np.ndarray) -> None:
        b = collate([enc[int(i)] for i in gi], special[2])
        try:
            pair, group = model(b["input_ids"].to(dev), b["attention_mask"].to(dev),
                                b["seg"].to(dev), b["n_cand"])  # fmt: skip
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(gi) <= 2:
                raise
            run(gi[: len(gi) // 2])
            run(gi[len(gi) // 2 :])
            return
        pp = torch.sigmoid(pair).cpu().numpy()
        gp = torch.sigmoid(group).cpu().numpy()
        pos, prim = b["pos"], b["primary"]
        sel = prim & (pos >= 0)
        out[pos[sel]] = pp[sel]
        first_group[s1_of[gi[is_first[gi]]]] = gp[is_first[gi]]

    with torch.inference_mode():
        for gi in _length_batches(enc, batch, None):
            run(gi)
            done += len(gi)
            if (done // batch) % 500 == 0:
                log.info("scored %d/%d groups (%.0f/s)", done, len(groups),
                         done / (time.time() - t0 + 1e-9))  # fmt: skip
    log.info("scored %d pairs in %d groups in %.0fs", len(rows), len(groups), time.time() - t0)
    grp = first_group[keys] if len(keys) else np.empty(0, np.float32)
    return out.astype(np.float16), grp.astype(np.float16)


# ----------------------------------------------------------------------------- evaluate
def _auc_logloss(p: np.ndarray, y: np.ndarray) -> dict[str, float]:
    from ber.crossenc import evaluate as pair_eval

    return pair_eval(p, y)


def _macro(rows, cols, p, labels, n_true, entities, beta: float = 0.5) -> tuple[float, float]:
    """Macro F-beta over ``entities`` (sorted S1 positions) after exclusive assignment, with
    the threshold tuned on even-indexed entities and reported on odd-indexed ones."""
    from ber.decision import macro_from_mask
    from ber.scale import exclusive_adjust

    adj = exclusive_adjust(rows, cols, p.astype(np.float32))
    halves = (entities[::2], entities[1::2])
    best_t, best = 0.5, -1.0
    for t in np.round(np.arange(0.05, 0.96, 0.01), 3):
        sub = halves[0]
        sel = np.isin(rows, sub)
        f = macro_from_mask(np.searchsorted(sub, rows[sel]), labels[sel], adj[sel] >= t,
                            n_true[sub], beta)  # fmt: skip
        if f > best:
            best, best_t = f, float(t)
    sub = halves[1]
    sel = np.isin(rows, sub)
    return best_t, macro_from_mask(np.searchsorted(sub, rows[sel]), labels[sel],
                                   adj[sel] >= best_t, n_true[sub], beta)  # fmt: skip


def evaluate_models(
    groups_path: str | Path,
    data_dir: str | Path,
    lw_models: list[str],
    ce_models: list[str],
    max_entities: int = 60_000,
    min_p1: float = 0.02,
    seed: int = 0,
) -> dict[str, dict]:
    """Compare stage 1, pairwise cross-encoders and listwise models on the same held-out keys
    of a groups file: pair AUC / log-loss, plus the mean-logit blend of every model with
    stage 1. Source 1 side: a macro-F0.5 proxy (exclusive, one tuned threshold, pairs below
    ``min_p1`` never selected). Pool side (C7): owner accuracy, the share of pool records
    whose best suitor above 0.5 is the owner, or that no suitor passes when nobody owns it."""
    from ber.crossenc import score as ce_score

    z = np.load(groups_path)
    rows, cols, p1 = z["rows"].astype(np.int64), z["cols"].astype(np.int64), z["p1"]
    labels, n_true = z["labels"].astype(bool), z["n_true"]
    side = str(z["side"]) if "side" in z.files else "s1"
    keys = keys_of(side, rows, cols)
    held = np.flatnonzero(z["valid"].astype(bool))
    held = held[np.isin(held, keys)]
    rng = np.random.default_rng(seed)
    ents = np.sort(rng.choice(held, min(max_entities, len(held)), replace=False))
    sel = np.isin(keys, ents) & (p1 >= min_p1)
    r, c, q, y = rows[sel], cols[sel], p1[sel].astype(np.float32), labels[sel]
    # groups from scripts/make_lw_groups.py carry a label-derived p1 proxy: not a baseline
    proxy = bool(z["p1_proxy"]) if "p1_proxy" in z.files else False
    scores = {} if proxy else {"stage1_p1": q}
    for m in ce_models:
        scores[f"ce:{Path(m).name}"] = ce_score(m, data_dir, "train", r, c).astype(np.float32)
    for m in lw_models:
        pp, _ = score(m, data_dir, "train", r, c, q)
        scores[f"lw:{Path(m).name}"] = pp.astype(np.float32)
    logit = {k: np.log(np.clip(v, 1e-4, 1 - 1e-4) / (1 - np.clip(v, 1e-4, 1 - 1e-4)))
             for k, v in scores.items()}  # fmt: skip
    anchor = next((k for k in scores if k.startswith("ce:")), None) if proxy else "stage1_p1"
    for k in list(scores):
        if anchor is not None and k != anchor and not (proxy and k.startswith("ce:")):
            # the mean-logit blend: does the model add to what the anchor already knows?
            scores[f"blend:{k}+{anchor}"] = 1 / (1 + np.exp(-(logit[k] + logit[anchor]) / 2))
    report = {}
    for k, v in scores.items():
        if side == "pool":
            report[k] = {**_auc_logloss(v, y), "owner_accuracy": round(_owner_accuracy(c, v, y), 5)}
        else:
            t, f = _macro(r, c, v, y, n_true, ents)
            report[k] = {**_auc_logloss(v, y), "macro_f05_proxy": round(f, 5), "threshold": t}
        log.info("%s: %s", k, report[k])
    return report


def _owner_accuracy(keys: np.ndarray, p: np.ndarray, y: np.ndarray, t: float = 0.5) -> float:
    """Per key (pool record): right when its best pair is above ``t`` and true, or when no
    pair is above ``t`` and none is true (the record is an orphan or a decoy)."""
    order = np.lexsort((-p, keys))
    k, pp, yy = keys[order], p[order], y[order]
    first = np.r_[True, k[1:] != k[:-1]]
    has_true = np.bincount(np.searchsorted(k[first], k), weights=yy.astype(float)) > 0
    top_p, top_y = pp[first], yy[first]
    right = np.where(top_p >= t, top_y, ~has_true)
    return float(right.mean()) if len(right) else float("nan")


# ----------------------------------------------------------------------------- driver
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ber.listwise")
    ap.add_argument("command", choices=["train", "score", "eval"])
    ap.add_argument("--data-dir", default="student_resource/dataset")
    ap.add_argument("--init", default="models/ce-small-v2", help="train: encoder to start from")
    ap.add_argument("--model", default="models/lw-small", help="comma-separated for eval")
    ap.add_argument("--groups", help="train/eval: lw_groups_train.npz from ber.scale")
    ap.add_argument(
        "--ce", default="", help="eval: pairwise cross-encoders to compare, comma-separated"
    )
    ap.add_argument("--split", default="test")
    ap.add_argument("--pairs", help="score: npz with rows, cols, p1 (positions in split order)")
    ap.add_argument("--out", help="score: output .npy of float16 probabilities (+ _grp.npy)")
    ap.add_argument("--k", type=int, default=K)
    ap.add_argument("--max-len", type=int, default=MAX_LEN)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-entities", type=int, default=0)
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname).1s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")  # fmt: skip
    if args.command == "train":
        train(args.data_dir, args.init, args.groups, args.model, k=args.k, max_len=args.max_len,
              batch=args.batch, lr=args.lr, epochs=args.epochs, max_entities=args.max_entities,
              grad_ckpt=args.grad_ckpt, workers=args.workers)  # fmt: skip
    elif args.command == "eval":
        report = evaluate_models(args.groups, args.data_dir,
                                 [m for m in args.model.split(",") if m],
                                 [m for m in args.ce.split(",") if m],
                                 max_entities=args.max_entities or 60_000)  # fmt: skip
        print(json.dumps(report, indent=2))
    else:
        with np.load(args.pairs) as z:
            rows, cols = z["rows"], z["cols"]
            p1 = z["p1"] if "p1" in z.files else np.zeros(len(rows), np.float32)
        # ber.scale passes no --batch: groups per scoring batch come from the environment
        # (out-of-memory batches are split, so a large value is safe)
        batch = int(os.environ.get("BER_LISTWISE_SCORE_BATCH", max(128, args.batch * 4)))
        pp, gp = score(args.model, args.data_dir, args.split, rows, cols, p1, batch=batch,
                       workers=args.workers)  # fmt: skip
        out = Path(args.out)
        np.save(out, pp)
        np.save(out.with_name(f"{out.stem}_grp.npy"), gp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
