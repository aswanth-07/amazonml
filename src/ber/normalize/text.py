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


@lru_cache(maxsize=4096)
def _fold_latin_char(ch: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", ch) if not unicodedata.combining(c))


def _latin_with_marks(o: int) -> bool:
    # Latin-1 Supplement, Latin Extended-A/B, Latin Extended Additional
    return 0x80 <= o < 0x250 or 0x1E00 <= o < 0x1F00


def fold_latin(text: str) -> str:
    """Drop the diacritics of Latin letters only: ``"Çlub Spôrtive"`` -> ``"Club Sportive"``.

    Unlike ``fold_accents`` it leaves every other script alone (Devanagari vowel signs and
    viramas are combining marks too), keeps letters without a decomposition (ß, ø, œ) and
    changes nothing else, so it is safe on text a neural model reads as is."""
    if text.isascii():
        return text
    out: list[str] = []
    for ch in text:
        if _latin_with_marks(ord(ch)):
            out.append(_fold_latin_char(ch))
        elif (
            unicodedata.combining(ch) and out and out[-1][-1:].isascii() and out[-1][-1:].isalpha()
        ):
            continue  # a separate mark on a Latin letter ("e" + U+0301)
        else:
            out.append(ch)
    return "".join(out)


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


# Consonant key: a coarser phonetic code that survives romanisation of Indic
# scripts, where vowels are the least stable part. "painiar" and "pioneer" both
# give "pnr", "sophtveyr" and "software" give "sftvr", "imdstrij" and
# "industries" give "indstrs".
_KEY_RULES = [
    ("ph", "f"), ("bh", "b"), ("kh", "k"), ("gh", "g"), ("dh", "d"), ("th", "t"),
    ("sh", "s"), ("ch", "k"), ("ck", "k"), ("c", "k"), ("q", "k"), ("x", "ks"),
    ("z", "j"), ("w", "v"), ("y", "i"),
]  # fmt: skip
# anyascii writes the anusvara as "m" before a consonant ("imdstrij", "marketimg");
# only the velar/palatal/dental-stop cases are rewritten, "limtid" keeps its m.
_NASAL = re.compile(r"m(?=[dgkj])")
_VOWELS = re.compile(r"[aeiou]")


@lru_cache(maxsize=2_000_000)
def consonant_key(token: str) -> str:
    if not token or not token.isalpha():
        return token
    out = token
    for src, dst in _KEY_RULES:
        out = out.replace(src, dst)
    out = _NASAL.sub("n", out)
    out = out[0] + _VOWELS.sub("", out[1:])
    out = _REPEAT.sub(r"\1", out)
    if len(out) > 2 and out.endswith("j"):
        out = out[:-1] + "s"
    return out


_LEET = str.maketrans(
    {"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "6": "g", "7": "t", "8": "b", "9": "g"}
)
_HAS_DIGIT = re.compile(r"\d")
_HAS_ALPHA = re.compile(r"[a-z]")


def deleet(token: str) -> str:
    """Undo digit-for-letter typos inside words ("g1obal", "6lobal"); leave real numbers."""
    if _HAS_DIGIT.search(token) and len(_HAS_ALPHA.findall(token)) >= 2:
        return token.translate(_LEET)
    return token
