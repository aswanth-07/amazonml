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

log = logging.getLogger("ber.crossenc")

MAX_LEN = 96


def _text(df: pd.DataFrame) -> list[str]:
    return [f"{n} | {a}" for n, a in zip(df[schema.NAME].tolist(), df[schema.ADDRESS].tolist())]


def _model(init: str | Path, head: Path | None = None):
    import torch
    from transformers import AutoModel

    class CrossEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = AutoModel.from_pretrained(init)
            self.head = torch.nn.Linear(self.encoder.config.hidden_size, 1)

        def forward(self, input_ids, attention_mask, **kw):
            h = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **kw)
            h = h.last_hidden_state
            m = attention_mask.unsqueeze(-1).to(h.dtype)
            v = (h * m).sum(1) / m.sum(1).clamp(min=1)
            return self.head(v.float()).squeeze(-1)

    model = CrossEncoder()
    if head is not None:
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
) -> Path:
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

    c = joblib.load(blocking)
    r, col, sc = c["rows"].astype(np.int64), c["cols"].astype(np.int64), c["base"]["score"]
    del c
    neg = (owner[col] != r) & enc[r]
    r, col, sc = r[neg], col[neg], sc[neg]
    order = np.lexsort((-sc, r))
    r, col = r[order], col[order]
    start = np.searchsorted(r, r)
    rank = np.arange(len(r)) - start
    keep = rank < n_neg
    neg_r, neg_c = r[keep], col[keep]
    pos = enc[a]
    pos_r, pos_c = a[pos], b[pos]
    rows = np.concatenate([pos_r, neg_r])
    cols = np.concatenate([pos_c, neg_c])
    y = np.concatenate([np.ones(len(pos_r), np.float32), np.zeros(len(neg_r), np.float32)])
    pick = rng.permutation(len(rows))[:max_pairs]
    rows, cols, y = rows[pick], cols[pick], y[pick]
    log.info("cross-encoder pairs: %d (%.1f%% positive)", len(y), 100 * y.mean())

    ta, tb = _text(s1), _text(pool)
    dev = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(init)
    model = _model(init).to(dev).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    n_steps = len(y) // batch
    warm = max(1, n_steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warm, max(0.0, (n_steps - s) / max(1, n_steps - warm)))
    )
    lossf = torch.nn.BCEWithLogitsLoss()
    t0 = time.time()
    for step in range(n_steps):
        sl = slice(step * batch, (step + 1) * batch)
        x = tok([ta[i] for i in rows[sl]], [tb[i] for i in cols[sl]], padding=True,
                truncation="longest_first", max_length=MAX_LEN, return_tensors="pt").to(dev)  # fmt: skip
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(**x)
        loss = lossf(logits.float(), torch.as_tensor(y[sl], device=dev))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
                "pairs": len(y),
                "positive_share": float(y.mean()),
                "n_neg": n_neg,
                "batch": batch,
                "lr": lr,
                "steps": n_steps,
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
    batch: int = 1024,
) -> np.ndarray:
    """P(match) per (S1 position, pool position) pair, as float16, in the given order."""
    import torch
    from transformers import AutoTokenizer

    s1, pool = load_frames(data_dir, split)
    ta, tb = _text(s1), _text(pool)
    dev = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = _model(model_dir, Path(model_dir) / "head.pt").to(dev).half().eval()
    lengths = np.fromiter(
        (len(ta[r]) + len(tb[c]) for r, c in zip(rows, cols)), np.int32, len(rows)
    )
    order = np.argsort(lengths, kind="stable")
    out = np.empty(len(rows), np.float16)
    t0 = time.time()
    with torch.inference_mode():
        for s in range(0, len(order), batch):
            idx = order[s : s + batch]
            x = tok([ta[i] for i in rows[idx]], [tb[i] for i in cols[idx]], padding=True,
                    truncation="longest_first", max_length=MAX_LEN, return_tensors="pt").to(dev)  # fmt: skip
            out[idx] = torch.sigmoid(model(**x).float()).cpu().numpy()
            if (s // batch) % 500 == 0:
                log.info(
                    "scored %d/%d (%.0f/s)",
                    s + len(idx),
                    len(rows),
                    (s + len(idx)) / (time.time() - t0 + 1e-9),
                )
    log.info("scored %d pairs in %.0fs", len(rows), time.time() - t0)
    return out


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
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname).1s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")  # fmt: skip
    if args.command == "train":
        train(args.data_dir, args.init, args.blocking, args.model, args.n_neg, args.max_pairs)
    else:
        with np.load(args.pairs) as z:
            rows, cols = z["rows"], z["cols"]
        np.save(args.out, score(args.model, args.data_dir, args.split, rows, cols))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
