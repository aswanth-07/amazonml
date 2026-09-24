"""Script- and punctuation-level cleaning shared by names and addresses."""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

try:  # ISC-licensed transliteration of non-Latin scripts (Devanagari, etc.)
    from anyascii import anyascii as _anyascii
except ImportError:  # pragma: no cover - optional
    _anyascii = None

_DOTTED_ACRONYM = re.compile(r"\b(?:[a-z]\.){2,}")
_APOSTROPHE = re.compile(r"[’‘'`´]")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_WS = re.compile(r"\s+")


def fold_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


@lru_cache(maxsize=1_000_000)
def clean(text: str, transliterate: bool = True) -> str:
    """Lowercase ASCII, ``&`` spelled out, dotted acronyms joined, other punctuation to spaces.

    ``"L.L.C."`` -> ``"llc"``, ``"McDonald's"`` -> ``"mcdonalds"``,
    ``"Société Générale"`` -> ``"societe generale"``, ``"A&B"`` -> ``"a and b"``.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    if not text.isascii():
        # anyascii folds Latin accents too; dropping combining marks first would
        # strip viramas and vowel signs from Indic scripts before transliteration.
        text = _anyascii(text) if transliterate and _anyascii is not None else fold_accents(text)
    text = text.casefold()
    text = text.replace("&", " and ").replace("+", " and ").replace("@", " at ")
    text = _DOTTED_ACRONYM.sub(lambda m: m.group(0).replace(".", ""), text)
    text = _APOSTROPHE.sub("", text)
    text = _NON_ALNUM.sub(" ", text)
    return _WS.sub(" ", text).strip()


def canonical_tokens(
    tokens: list[str], mapping: dict[str, str], drop: frozenset[str] = frozenset()
) -> list[str]:
    out = []
    for tok in tokens:
        tok = mapping.get(tok, tok)
        if tok and tok not in drop:
            out.append(tok)
    return out


def replace_phrases(text: str, phrases: list[tuple[re.Pattern[str], str]]) -> str:
    for pattern, repl in phrases:
        text = pattern.sub(repl, text)
    return _WS.sub(" ", text).strip()


def compile_phrases(pairs: list[tuple[str, str]]) -> list[tuple[re.Pattern[str], str]]:
    """Whole-word phrase rewrites, longest first so they do not shadow each other."""
    ordered = sorted(pairs, key=lambda p: -len(p[0]))
    return [(re.compile(rf"\b{re.escape(src)}\b"), dst) for src, dst in ordered]


# Transliteration skeleton: collapses spelling variants common in romanised
# Indic names (Lakshmi/Laxmi, Shree/Sri, Venkateshwara/Venkateswara) and in
# English/French doubling. It is a matching key, never shown to anyone.
_SKELETON_RULES = [
    ("ksh", "x"),
    ("chh", "c"),
    ("sh", "s"),
    ("ch", "c"),
    ("ph", "f"),
    ("th", "t"),
    ("dh", "d"),
    ("bh", "b"),
    ("kh", "k"),
    ("gh", "g"),
    ("jh", "j"),
    ("ck", "k"),
    ("q", "k"),
    ("z", "j"),
    ("w", "v"),
    ("ee", "i"),
    ("oo", "u"),
    ("y", "i"),
]
_REPEAT = re.compile(r"(.)\1+")


@lru_cache(maxsize=1_000_000)
def skeleton(token: str) -> str:
    if not token or token.isdigit():
        return token
    out = token
    for src, dst in _SKELETON_RULES:
        out = out.replace(src, dst)
    out = _REPEAT.sub(r"\1", out)
    if len(out) > 3 and out.endswith(("a", "e", "h")):
        out = out[:-1]
    return out
