"""Business-name normalisation: legal suffixes, trade names, acronyms, spelling skeletons.

Mappings here are general knowledge of how business names are written in the
US, India and France (and neighbouring conventions), plus the romanised forms
anyascii produces for Indic-script legal words. They are not a lookup of any
real entity, so they sit within the challenge's fair-play rules.

Noise handled (observed in the challenge data): legal forms moved to the front
("LLC Sapphire"), honorific prefixes (Mr, Dr, Smt, M/s), former and trade
names ("X fka Y", "X formerly known as Y", "X t/a Y", "X dba Y"), id tags
("#30473", "(ID: 96415)"), whole-name domains and hashtags ("lynx.com",
"#baskervillemeadows"), digit-for-letter typos ("G1obal"), brackets, stray
punctuation, accents, and names written in Devanagari, Gurmukhi, Bengali,
Gujarati, Tamil, Kannada and other scripts.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache

from ber.normalize.text import (
    _anyascii,
    canonical_tokens,
    clean,
    compile_phrases,
    consonant_key,
    deleet,
    fold_accents,
    replace_phrases,
    skeleton,
)

_PHRASES = compile_phrases(
    [
        ("private limited", "pvt ltd"),
        ("pvt limited", "pvt ltd"),
        ("private ltd", "pvt ltd"),
        ("pra li", "pvt ltd"),
        ("public limited company", "plc"),
        ("limited liability company", "llc"),
        ("limited liability partnership", "llp"),
        ("l l c", "llc"),
        ("l l p", "llp"),
        ("p ltd", "pvt ltd"),
        ("societe a responsabilite limitee", "sarl"),
        ("societe par actions simplifiee", "sas"),
        ("societe anonyme", "sa"),
        ("s a r l", "sarl"),
        ("s a s", "sas"),
        ("m s", ""),  # "M/s" (messrs)
    ]
)

_ALIAS_SPLIT = re.compile(
    r"\s*(?:\bf/?k/?a\b|\bformerly known as\b|\bformerly\b|\bt/a\b|\bd/b/a\b|\bdba\b|\ba/k/a\b"
    r"|\baka\b|\btrading as\b|\bdoing business as\b|\balso known as\b|\bnow known as\b"
    r"|\bpreviously\b)\s*:?\s*"
)
_ID_TAG = re.compile(r"\(\s*id\s*:?\s*\d+\s*\)|\bid\s*:\s*\d+|#\s*\d+")
_DOMAIN = re.compile(
    r"^(?:https?://)?(?:www\.)?([a-z0-9][a-z0-9-]*)\.(?:com|net|org|co\.in|in|co|io|biz|info|fr|us|co\.uk)/?$"
)
_HASHTAG = re.compile(r"^#+([a-z][a-z0-9]*)$")

# Canonical short form for legal-form words, including romanised Indic forms
# ("praivet limited" from प्राइवेट लिमिटेड).
LEGAL_FORMS: dict[str, str] = {
    "corporation": "corp", "corp": "corp", "korporesn": "corp", "karporesn": "corp",
    "incorporated": "inc", "inc": "inc",
    "company": "co", "co": "co", "kmpni": "co", "kampni": "co", "kampani": "co",
    "limited": "ltd", "ltd": "ltd", "limtid": "ltd", "limitd": "ltd", "limted": "ltd",
    "limitid": "ltd", "lmtd": "ltd", "limitad": "ltd",
    "private": "pvt", "pvt": "pvt", "praivet": "pvt", "praibhet": "pvt", "privet": "pvt",
    "prayvet": "pvt", "praivhet": "pvt",
    "pte": "pte", "pty": "pty",
    "llc": "llc", "elelsi": "llc", "llp": "llp", "elelpi": "llp", "lp": "lp", "pllc": "pllc",
    "plc": "plc", "pc": "pc",
    "gmbh": "gmbh", "ag": "ag", "bv": "bv", "nv": "nv", "srl": "srl", "spa": "spa",
    "sa": "sa", "sarl": "sarl", "sas": "sas", "sasu": "sasu", "eurl": "eurl", "snc": "snc",
    "sci": "sci", "cie": "cie", "compagnie": "cie", "opc": "opc",
}  # fmt: skip
LEGAL_SET = frozenset(LEGAL_FORMS.values())

# Non-legal words written in several ways.
WORD_FORMS: dict[str, str] = {
    "et": "and", "brothers": "bros", "bros": "bros", "international": "intl", "intl": "intl",
    "technologies": "tech", "technology": "tech", "manufacturing": "mfg", "associates": "assoc",
    "association": "assn", "department": "dept", "center": "ctr", "centre": "ctr",
    "services": "svc", "service": "svc", "saint": "st", "sainte": "st", "mount": "mt",
    "jewelers": "jewellers", "shree": "sri", "shri": "sri", "sree": "sri", "societe": "ste",
}  # fmt: skip
STOP = frozenset({"the", "le", "la", "les", "l", "a", "of"})
HONORIFICS = frozenset({"mr", "mrs", "ms", "dr", "smt", "messrs", "mx", "prof"})
_MAP = {**WORD_FORMS, **LEGAL_FORMS}


@dataclass(frozen=True)
class NormalizedName:
    clean: str  # cleaned, canonical tokens, legal forms kept
    core: str  # legal forms, articles and honorifics removed
    alias: str  # core of the other name in "X fka Y" / "X t/a Y" / "X dba Y"
    legal: str  # sorted legal forms found, space separated
    skeleton: str  # transliteration skeleton of the core
    acronym: str  # initials of core tokens (>= 2 tokens), else ""
    key: str = ""  # consonant key of the core (see text.consonant_key)
    squash: str = ""  # core without spaces, for domain/hashtag names
    script: bool = False  # written in a non-Latin script
    styled: bool = False  # a domain or hashtag rather than a written-out name
    marker: bool = False  # carried a former/trade-name marker


def _is_non_latin(raw: str) -> bool:
    return any(ord(ch) > 0x24F and unicodedata.category(ch).startswith("L") for ch in raw)


def _ascii_lower(raw: str, transliterate: bool) -> str:
    text = unicodedata.normalize("NFKC", raw)
    if not text.isascii():
        text = _anyascii(text) if transliterate and _anyascii is not None else fold_accents(text)
    return text.casefold().strip()


def _core_tokens(text: str) -> tuple[list[str], list[str]]:
    tokens = canonical_tokens([deleet(t) for t in text.split()], _MAP)
    while tokens and tokens[0] in HONORIFICS:
        tokens = tokens[1:]
    core = [t for t in tokens if t not in LEGAL_SET and t not in STOP and t not in HONORIFICS]
    if not core:  # a name that is only a legal form ("The Company"): keep what there is
        core = [t for t in tokens if t not in HONORIFICS] or tokens
    return tokens, core


@lru_cache(maxsize=1_000_000)
def normalize_name(raw: str, transliterate: bool = True) -> NormalizedName:
    lower = _ascii_lower(raw, transliterate)
    lower = _ID_TAG.sub(" ", lower)
    styled = False
    stripped = lower.strip(" -*.<>[](){}|~")
    for pattern in (_DOMAIN, _HASHTAG):
        m = pattern.match(stripped)
        if m:
            lower, styled = m.group(1), True
            break
    parts = [p for p in _ALIAS_SPLIT.split(lower) if p.strip()]
    marker = len(parts) > 1
    primary = parts[0] if parts else lower
    alias_raw = " ".join(parts[1:])

    text = replace_phrases(clean(primary, transliterate), _PHRASES)
    tokens, core = _core_tokens(text)
    alias = []
    if alias_raw:
        _, alias = _core_tokens(replace_phrases(clean(alias_raw, transliterate), _PHRASES))
        tokens = tokens + [
            t for t in canonical_tokens(clean(alias_raw).split(), _MAP) if t in LEGAL_SET
        ]
    legal = sorted({t for t in tokens if t in LEGAL_SET})
    acronym = "".join(t[0] for t in core if not t.isdigit()) if len(core) >= 2 else ""
    return NormalizedName(
        clean=" ".join(tokens),
        core=" ".join(core),
        alias=" ".join(alias),
        legal=" ".join(legal),
        skeleton=" ".join(skeleton(t) for t in core),
        acronym=acronym,
        key=" ".join(consonant_key(t) for t in core),
        squash="".join(core),
        script=_is_non_latin(raw),
        styled=styled,
        marker=marker,
    )
