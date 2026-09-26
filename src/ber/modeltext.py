"""The text the neural models read: ``"name | address"``, with Latin accents folded.

Train names carry accent noise only as acute accents (á é í ó ú). French test pool names
carry ç à è ô â ë ê ï î instead (11% of them; none in 10M train records), which the
encoders never saw and whose SentencePiece vocabulary keeps apart from the plain letter
("Çlub" and "Club" are different tokens). Folding the diacritics of Latin letters before a
model reads the text removes that noise channel for every country; other scripts are left
exactly as they are (``ber.normalize.text.fold_latin``).

A model records whether it was trained on folded text (``text_fold`` in the
``finetune.json`` / ``crossenc.json`` / ``listwise.json`` its training writes). Scoring
follows the model, so a model trained before this existed keeps reading raw text. New
models fold unless ``BER_TEXT_FOLD=0``.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from pathlib import Path

from ber.normalize.text import fold_latin

FOLD_ENV = "BER_TEXT_FOLD"
META_FILES = ("crossenc.json", "listwise.json", "finetune.json")


def default_fold() -> bool:
    """Whether a model trained now reads folded text (``BER_TEXT_FOLD``, default on)."""
    return os.environ.get(FOLD_ENV, "1").strip().lower() not in ("0", "false", "no", "off")


def model_fold(model_dir: str | Path) -> bool:
    """Whether the model in ``model_dir`` was trained on folded text (False when it never
    recorded it: a model from before folding existed, or a base checkpoint)."""
    for name in META_FILES:
        path = Path(model_dir) / name
        if path.exists():
            try:
                info = json.loads(path.read_text())
            except json.JSONDecodeError:
                continue
            if "text_fold" in info:
                return bool(info["text_fold"])
    return False


def record_text(name: str, address: str, fold: bool, prefix: str = "") -> str:
    text = f"{prefix}{name} | {address}"
    return fold_latin(text) if fold else text


def record_texts(names: Iterable[str], addresses: Iterable[str], fold: bool,
                 prefix: str = "") -> list[str]:  # fmt: skip
    return [record_text(n, a, fold, prefix) for n, a in zip(names, addresses)]
