"""Business-name normalisation: legal suffixes, trade names, acronyms, spelling skeletons.

Mappings here are general knowledge of how business names are written in the
US, India and France (and neighbouring conventions). They are not a lookup of
any real entity, so they sit within the challenge's fair-play rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from ber.normalize.text import canonical_tokens, clean, compile_phrases, replace_phrases, skeleton

_PHRASES = compile_phrases(
    [
        ("private limited", "pvt ltd"),
        ("pvt limited", "pvt ltd"),
        ("private ltd", "pvt ltd"),
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
        ("doing business as", " dba "),
        ("d b a", " dba "),
        ("trading as", " dba "),
        ("also known as", " dba "),
        ("a k a", " dba "),
    ]
)

# Canonical short form for legal-form words.
LEGAL_FORMS: dict[str, str] = {
    "corporation": "corp",
    "corp": "corp",
    "incorporated": "inc",
    "inc": "inc",
    "company": "co",
    "co": "co",
    "limited": "ltd",
    "ltd": "ltd",
    "private": "pvt",
    "pvt": "pvt",
    "pte": "pte",
    "pty": "pty",
    "llc": "llc",
    "llp": "llp",
    "lp": "lp",
    "plc": "plc",
    "gmbh": "gmbh",
    "ag": "ag",
    "bv": "bv",
    "nv": "nv",
    "srl": "srl",
    "spa": "spa",
    "sa": "sa",
    "sarl": "sarl",
    "sas": "sas",
    "sasu": "sasu",
    "eurl": "eurl",
    "snc": "snc",
    "sci": "sci",
    "cie": "cie",
    "compagnie": "cie",
    "opc": "opc",
}
LEGAL_SET = frozenset(LEGAL_FORMS.values())

# Non-legal words written in several ways.
WORD_FORMS: dict[str, str] = {
    "et": "and",
    "brothers": "bros",
    "bros": "bros",
    "international": "intl",
    "intl": "intl",
    "technologies": "tech",
    "technology": "tech",
    "manufacturing": "mfg",
    "associates": "assoc",
    "association": "assn",
    "department": "dept",
    "center": "ctr",
    "centre": "ctr",
    "services": "svc",
    "service": "svc",
    "saint": "st",
    "sainte": "st",
    "mount": "mt",
    "jewelers": "jewellers",
    "shree": "sri",
    "shri": "sri",
    "sree": "sri",
    "societe": "ste",
}
STOP = frozenset({"the", "le", "la", "les", "l"})


@dataclass(frozen=True)
class NormalizedName:
    clean: str  # cleaned, canonical tokens, legal forms kept
    core: str  # legal forms and articles removed
    alias: str  # core of the trade name after "dba", if any
    legal: str  # sorted legal forms found, space separated
    skeleton: str  # transliteration skeleton of the core
    acronym: str  # initials of core tokens (>= 2 tokens), else ""


@lru_cache(maxsize=1_000_000)
def normalize_name(raw: str, transliterate: bool = True) -> NormalizedName:
    text = replace_phrases(clean(raw, transliterate), _PHRASES)
    parts = [p.strip() for p in text.split(" dba ")] if " dba " in f" {text} " else [text]
    primary = parts[0]
    alias_text = " ".join(parts[1:]).strip()

    tokens = canonical_tokens(primary.split(), {**WORD_FORMS, **LEGAL_FORMS})
    legal = sorted({t for t in tokens if t in LEGAL_SET})
    core_tokens = [t for t in tokens if t not in LEGAL_SET and t not in STOP and t != "dba"]
    if not core_tokens:  # a name that is only a legal form ("The Company"): keep what there is
        core_tokens = [t for t in tokens if t != "dba"]
    alias_tokens = [
        t
        for t in canonical_tokens(alias_text.split(), {**WORD_FORMS, **LEGAL_FORMS})
        if t not in LEGAL_SET and t not in STOP and t != "dba"
    ]
    acronym = "".join(t[0] for t in core_tokens if not t.isdigit()) if len(core_tokens) >= 2 else ""
    return NormalizedName(
        clean=" ".join(tokens),
        core=" ".join(core_tokens),
        alias=" ".join(alias_tokens),
        legal=" ".join(legal),
        skeleton=" ".join(skeleton(t) for t in core_tokens),
        acronym=acronym,
    )
