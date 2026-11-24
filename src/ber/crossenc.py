"""Cross-encoder: reads a Source 1 record and a candidate together and scores the pair.

A bi-encoder compares two independently built vectors; a cross-encoder attends
across both records, so it can check that "Pioneer Industries, 12 MG Rd" and
"Pioneer Industrial Traders, 12 MG Road" differ in the one word that matters.
It is initialised from the fine-tuned bi-encoder (``ber.embed``, from
multilingual-e5-small, MIT) with a linear head on the mean-pooled output.

Training pairs come from the encoder-share entities only (``ber.embed.
encoder_mask``): every true pair, plus each entity's highest-scoring wrong
candidates from token blocking as hard negatives. ``ber.scale`` scores the
uncertain pairs of every other entity and of test, and feeds the probability
to the stage-2 model, which never trains on encoder-share entities.

Needs PyTorch and transformers (run with the torch interpreter)::

    py -3.13 -m ber.crossenc train --blocking artifacts/cache/train_blocking_<key>.joblib
    py -3.13 -m ber.crossenc score --split test --pairs pairs.npz --out scores.npy
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ber import schema
from ber.embed import encoder_mask, load_frames, truth_pairs
from ber.modeltext import default_fold, model_fold, record_texts

log = logging.getLogger("ber.crossenc")

MAX_LEN = 96


def _text(df: pd.DataFrame, fold: bool = False) -> list[str]:
    """``"name | address"`` per record; ``fold``: Latin accents folded (ber.modeltext)."""
    return record_texts(df[schema.NAME].tolist(), df[schema.ADDRESS].tolist(), fold)


def _text_of(df: pd.DataFrame, idx: np.ndarray, fold: bool = False) -> np.ndarray:
    """The pair text of only the records ``idx`` (None elsewhere): training touches a few
    million of a split's ~12M records, and a full list of Python strings costs gigabytes."""
    out = np.empty(len(df), dtype=object)
    idx = np.unique(idx)
    out[idx] = record_texts(
        df[schema.NAME].to_numpy()[idx], df[schema.ADDRESS].to_numpy()[idx], fold
    )
    return out


def _encode(tok, left: list[str], right: list[str]):
    """Tokenise record pairs. A tokenizer with no separator token (a decoder such as Qwen3)
    gets the two records joined by an explicit marker instead of a bare concatenation."""
    if tok.sep_token is None:
        joined = [f"Record A: {a} || Record B: {b}" for a, b in zip(left, right)]
        return tok(joined, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
    return tok(left, right, padding=True, truncation="longest_first", max_length=MAX_LEN,
               return_tensors="pt")  # fmt: skip


def _infer_dtype(model_dir: str | Path):
    """fp16 for the encoders (their cached scores were made that way); bf16 for decoders."""
    import torch
    from transformers import AutoConfig

    decoder = AutoConfig.from_pretrained(model_dir).model_type.startswith("qwen")
    return torch.bfloat16 if decoder else torch.float16


def _model(init: str | Path, head: Path | None = None):
    """An encoder with a one-logit head. A checkpoint that already is a one-logit sequence
    classifier (a pretrained reranker such as bge-reranker-v2-m3) keeps its own trained
    head; any other encoder gets a new linear head on its mean-pooled output."""
    import torch
    from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification

    conf = AutoConfig.from_pretrained(init)
    native = (
        any(a.endswith("ForSequenceClassification") for a in (conf.architectures or []))
        and conf.num_labels == 1
    )

    class CrossEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            if native:
                self.encoder = AutoModelForSequenceClassification.from_pretrained(init)
                self.head = torch.nn.Identity()
            else:
                self.encoder = AutoModel.from_pretrained(init)
                self.head = torch.nn.Linear(self.encoder.config.hidden_size, 1)

        def forward(self, input_ids, attention_mask, **kw):
            if native:
                out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **kw)
                return out.logits.squeeze(-1)
            h = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **kw)
            h = h.last_hidden_state
            m = attention_mask.unsqueeze(-1).to(h.dtype)
            v = (h * m).sum(1) / m.sum(1).clamp(min=1)
            return self.head(v.to(self.head.weight.dtype)).squeeze(-1)

    model = CrossEncoder()
    if head is not None and not native:
        model.head.load_state_dict(torch.load(head, map_location="cpu"))
    return model


def train(
    data_dir: str | Path,
    init: str | Path,
    blocking: str | Path,
    out_dir: str | Path,
    n_neg: int = 4,
    max_pairs: int = 1_500_000,
    batch: int = 256,
    lr: float = 3e-5,
    seed: int = 42,
    knn: str | Path | None = None,
    n_neg_knn: int = 3,
    pseudo: str | Path | None = None,
    pseudo_split: str = "test",
    micro_batch: int = 0,
    freeze_embeddings: bool = False,
) -> Path:
    """``knn``: the embedding kNN lists of train (``ber.embed``); each entity's best
    ``n_neg_knn`` wrong neighbours join the token-blocking negatives, so the cross-encoder
    also learns the confusions the embedding view brings into the candidate set.

    ``pseudo``: pseudo-labelled pairs of another split (rows, cols, labels, as written by
    scripts/make_pseudo_labels.py) mixed into training, for self-training on a country the
    training data lacks. ``init`` may be an already trained cross-encoder, whose head is
    kept, so the model adapts rather than starts over.

    ``micro_batch``: accumulate each ``batch`` from slices of this size, so a large backbone
    (bge-reranker-v2-m3, 568M) trains with the same effective batch in 16 GB;
    ``freeze_embeddings`` leaves the word embeddings (a 250k-token vocabulary) as they are."""
    import joblib
    import torch
    from transformers import AutoTokenizer

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    s1, pool = load_frames(data_dir, "train")
    a, b = truth_pairs(data_dir, s1, pool)
    owner = np.full(len(pool), -1, dtype=np.int64)
    owner[b] = a
    enc = encoder_mask(s1[schema.ENTITY_ID].to_numpy())

    def hardest(r, col, sc, k):
        """Each encoder-share entity's k highest-scoring wrong candidates."""
        neg = (owner[col] != r) & enc[r]
        r, col, sc = r[neg], col[neg], sc[neg]
        order = np.lexsort((-sc, r))
        r, col = r[order], col[order]
        rank = np.arange(len(r)) - np.searchsorted(r, r)
        return r[rank < k], col[rank < k]

    c = joblib.load(blocking, mmap_mode="r")  # read only the arrays it needs from disk
    neg_r, neg_c = hardest(c["rows"].astype(np.int64), c["cols"].astype(np.int64),
                           c["base"]["score"], n_neg)  # fmt: skip
    del c
    if knn:
        with np.load(knn) as z:
            kr, kc = hardest(z["fwd_rows"].astype(np.int64), z["fwd_cols"].astype(np.int64),
                             z["fwd_sim"].astype(np.float32), n_neg_knn)  # fmt: skip
        key = np.unique(np.concatenate([neg_r * len(pool) + neg_c, kr * len(pool) + kc]))
        neg_r, neg_c = key // len(pool), key % len(pool)
    pos = enc[a]
    pos_r, pos_c = a[pos], b[pos]
    rows = np.concatenate([pos_r, neg_r])
    cols = np.concatenate([pos_c, neg_c])
    y = np.concatenate([np.ones(len(pos_r), np.float32), np.zeros(len(neg_r), np.float32)])
    pick = rng.permutation(len(rows))[:max_pairs]
    rows, cols, y = rows[pick], cols[pick], y[pick]
    fold = default_fold()  # recorded below; scoring reads the text the model trained on
    texts = [(_text_of(s1, rows, fold), _text_of(pool, cols, fold))]
    del s1, pool  # the frames are the bulk of this process's memory
    src = np.zeros(len(y), dtype=np.int8)
    if pseudo:
        with np.load(pseudo) as z:
            pr, pc, py = z["rows"].astype(np.int64), z["cols"].astype(np.int64), z["labels"]
        s1b, poolb = load_frames(data_dir, pseudo_split)
        texts.append((_text_of(s1b, pr, fold), _text_of(poolb, pc, fold)))
        del s1b, poolb
        rows, cols = np.concatenate([rows, pr]), np.concatenate([cols, pc])
        y = np.concatenate([y, py.astype(np.float32)])
        src = np.concatenate([src, np.ones(len(py), dtype=np.int8)])
        mix = rng.permutation(len(y))
        rows, cols, y, src = rows[mix], cols[mix], y[mix], src[mix]
        log.info("with %d pseudo-labelled %s pairs", len(py), pseudo_split)
    log.info("cross-encoder pairs: %d (%.1f%% positive)", len(y), 100 * y.mean())

    dev = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(init)
    head = Path(init) / "head.pt"
    model = _model(init, head if head.exists() else None).to(dev).train()
    if freeze_embeddings:  # a large vocabulary's embeddings: most of the optimiser memory
        for name, p in model.named_parameters():
            if "word_embeddings" in name or "embed_tokens" in name:
                p.requires_grad_(False)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    n_steps = len(y) // batch
    warm = max(1, n_steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warm, max(0.0, (n_steps - s) / max(1, n_steps - warm)))
    )
    lossf = torch.nn.BCEWithLogitsLoss(reduction="sum")
    micro = min(micro_batch or batch, batch)  # gradient accumulation for large backbones
    t0 = time.time()
    for step in range(n_steps):
        opt.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for m0 in range(step * batch, (step + 1) * batch, micro):
            sl = slice(m0, min(m0 + micro, (step + 1) * batch))
            left = [texts[s][0][i] for i, s in zip(rows[sl], src[sl])]
            right = [texts[s][1][i] for i, s in zip(cols[sl], src[sl])]
            x = _encode(tok, left, right).to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(**x)
            part = lossf(logits.float(), torch.as_tensor(y[sl], device=dev)) / batch
            part.backward()
            loss_sum += part.item()
        loss = torch.tensor(loss_sum)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if (step + 1) % 200 == 0 or step == 0:
            log.info("step %d/%d loss %.4f (%.1f steps/s)", step + 1, n_steps, loss.item(),
                     (step + 1) / (time.time() - t0))  # fmt: skip
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(out)
    tok.save_pretrained(out)
    torch.save(model.head.state_dict(), out / "head.pt")
    (out / "crossenc.json").write_text(
        json.dumps(
            {
                "init": str(init),
                "pseudo": str(pseudo) if pseudo else None,
                "pairs": len(y),
                "positive_share": float(y.mean()),
                "n_neg": n_neg,
                "batch": batch,
                "micro_batch": micro_batch,
                "freeze_embeddings": freeze_embeddings,
                "lr": lr,
                "steps": n_steps,
                "text_fold": fold,
                "seconds": round(time.time() - t0),
            },
            indent=2,
        )
    )
    log.info("cross-encoder saved to %s (%.0fs)", out, time.time() - t0)
    return out


def score(
    model_dir: str | Path,
    data_dir: str | Path,
    split: str,
    rows: np.ndarray,
    cols: np.ndarray,
    batch: int | None = None,
) -> np.ndarray:
    """P(match) per (S1 position, pool position) pair, as float16, in the given order.

    ``batch`` defaults by model width: 1024 pairs for small encoders, 128 for large ones.
    A 1024-pair batch of a large encoder on long US/India texts overflows 16 GB; on Windows
    the driver then spills into shared system memory instead of raising out-of-memory (so
    the halving below never triggers), and bge-reranker-v2-m3 ran ~3x slower than at 128."""
    import torch
    from transformers import AutoTokenizer

    s1, pool = load_frames(data_dir, split)
    fold = model_fold(model_dir)
    ta, tb = _text(s1, fold), _text(pool, fold)
    dev = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = _model(model_dir, Path(model_dir) / "head.pt").to(dev, _infer_dtype(model_dir)).eval()
    if batch is None:
        from transformers import AutoConfig

        width = getattr(AutoConfig.from_pretrained(model_dir), "hidden_size", 0)
        batch = 1024 if width <= 512 else 128
    lengths = np.fromiter(
        (len(ta[r]) + len(tb[c]) for r, c in zip(rows, cols)), np.int32, len(rows)
    )
    order = np.argsort(lengths, kind="stable")
    out = np.empty(len(rows), np.float16)
    t0 = time.time()

    def run(idx: np.ndarray) -> None:
        """Score one batch; on running out of GPU memory (another job may share the GPU),
        split it in half and retry rather than fail the whole run."""
        try:
            x = _encode(tok, [ta[i] for i in rows[idx]], [tb[i] for i in cols[idx]]).to(dev)
            out[idx] = torch.sigmoid(model(**x).float()).cpu().numpy()
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(idx) <= 8:
                raise
            half = len(idx) // 2
            run(idx[:half])
            run(idx[half:])

    with torch.inference_mode():
        for s in range(0, len(order), batch):
            idx = order[s : s + batch]
            run(idx)
            if (s // batch) % 500 == 0:
                log.info(
                    "scored %d/%d (%.0f/s)",
                    s + len(idx),
                    len(rows),
                    (s + len(idx)) / (time.time() - t0 + 1e-9),
                )
    log.info("scored %d pairs in %.0fs", len(rows), time.time() - t0)
    return out


def evaluate(p: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """AUC (rank-based), log-loss, and accuracy and F0.5 at the best threshold."""
    p = p.astype(np.float64)
    order = np.argsort(p, kind="stable")
    ranks = np.empty(len(p))
    ranks[order] = np.arange(1, len(p) + 1)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    auc = (ranks[y].sum() - n_pos * (n_pos + 1) / 2) / max(n_pos * n_neg, 1)
    q = np.clip(p, 1e-6, 1 - 1e-6)
    logloss = float(-np.mean(np.where(y, np.log(q), np.log(1 - q))))
    best = 0.0
    for t in np.arange(0.05, 0.96, 0.05):
        pred = p >= t
        tp = float((pred & y).sum())
        prec, rec = tp / max(pred.sum(), 1), tp / max(n_pos, 1)
        best = max(best, 1.25 * prec * rec / max(0.25 * prec + rec, 1e-9))
    return {
        "auc": round(float(auc), 5),
        "logloss": round(logloss, 5),
        "best_pair_f05": round(best, 5),
        "pairs": len(p),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ber.crossenc")
    ap.add_argument("command", choices=["train", "score"])
    ap.add_argument("--data-dir", default="student_resource/dataset")
    ap.add_argument("--init", default="models/e5-small-ft")
    ap.add_argument("--model", default="models/ce-small")
    ap.add_argument("--blocking", help="train: a train blocking cache (joblib) for hard negatives")
    ap.add_argument("--split", default="test")
    ap.add_argument("--pairs", help="score: npz with rows, cols (positions in split order)")
    ap.add_argument("--out", help="score: output .npy of float16 probabilities")
    ap.add_argument("--max-pairs", type=int, default=1_500_000)
    ap.add_argument("--n-neg", type=int, default=4)
    ap.add_argument("--knn", help="train: embedding kNN lists (npz) for extra hard negatives")
    ap.add_argument("--n-neg-knn", type=int, default=3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--pseudo", help="train: pseudo-labelled pairs of another split (npz)")
    ap.add_argument("--pseudo-split", default="test")
    ap.add_argument("--micro-batch", type=int, default=0, help="train: gradient-accumulation slice")
    ap.add_argument("--freeze-embeddings", action="store_true", help="train: keep word embeddings")
    ap.add_argument("--seed", type=int, default=42, help="train: which pairs are sampled, and init")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname).1s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")  # fmt: skip
    if args.command == "train":
        train(args.data_dir, args.init, args.blocking, args.model, args.n_neg, args.max_pairs,
              batch=args.batch, lr=args.lr, knn=args.knn, n_neg_knn=args.n_neg_knn,
              pseudo=args.pseudo, pseudo_split=args.pseudo_split, micro_batch=args.micro_batch,
              freeze_embeddings=args.freeze_embeddings, seed=args.seed)  # fmt: skip
    else:
        with np.load(args.pairs) as z:
            rows, cols = z["rows"], z["cols"]
        np.save(args.out, score(args.model, args.data_dir, args.split, rows, cols))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
