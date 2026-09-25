"""Address normalisation: abbreviations, postal codes, house numbers, landmarks.

Handles the noise the challenge lists (Rd/Road, St/Street, transliterated city
names, missing PIN or state, "Near SBI ATM" style landmarks, reordered
components) for US, Indian and French conventions without assuming the set of
countries is closed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from ber.normalize.text import canonical_tokens, clean, compile_phrases, replace_phrases

_SEGMENT_SPLIT = re.compile(r"[,;\n|]+|\s[-–—]\s")
_PIN_SPACED = re.compile(r"(?<!\d)(\d{3}) (\d{3})(?!\d)")
_POSTAL = re.compile(r"(?<!\d)(\d{5,6})(?!\d)")
_LEADING_NUMBER = re.compile(r"^\W*(?:no\.?|#|number)?\s*(\d{5,6})(?!\d)", re.IGNORECASE)
_DIGITS = re.compile(r"\d+")

_PHRASES = compile_phrases(
    [
        ("next to", "near"),
        ("close to", "near"),
        ("in front of", "opp"),
        ("opposite to", "opp"),
        ("opp to", "opp"),
        ("pres de", "near"),
        ("face a", "opp"),
        ("p o box", "pobox"),
        ("po box", "pobox"),
        ("post box", "pobox"),
        ("boite postale", "pobox"),
        ("h no", ""),
        ("door no", ""),
        ("house no", ""),
        ("s o", ""),  # "S/O" (son of) in Indian addresses
        ("d o", ""),
        ("w o", ""),
    ]
)
_US_PHRASES = compile_phrases(
    [
        ("new york", "ny"),
        ("new jersey", "nj"),
        ("new mexico", "nm"),
        ("new hampshire", "nh"),
        ("north carolina", "nc"),
        ("south carolina", "sc"),
        ("north dakota", "nd"),
        ("south dakota", "sd"),
        ("west virginia", "wv"),
        ("rhode island", "ri"),
        ("district of columbia", "dc"),
    ]
)
# Indian states: English name, postal code, and the romanisation anyascii gives
# for the state written in its own script (महाराष्ट्र -> "mharastr").
_INDIA_PHRASES = compile_phrases(
    [
        ("uttar pradesh", "up"), ("uttr prdes", "up"), ("madhya pradesh", "mp"),
        ("mdhy prdes", "mp"), ("andhra pradesh", "ap"), ("amdhr prdes", "ap"),
        ("himachal pradesh", "hp"), ("arunachal pradesh", "arp"), ("tamil nadu", "tn"),
        ("west bengal", "wb"), ("jammu and kashmir", "jk"), ("jammu kashmir", "jk"),
        ("nct of delhi", "dl"), ("dadra and nagar haveli", "dnh"),
    ]
)  # fmt: skip
INDIA_STATES: dict[str, str] = {
    "maharashtra": "mh", "mharastr": "mh", "mh": "mh", "delhi": "dl", "dilli": "dl",
    "dl": "dl", "up": "up", "karnataka": "ka", "krnatk": "ka", "ka": "ka", "tamilnadu": "tn",
    "tmilnatu": "tn", "tn": "tn", "gujarat": "gj", "gujrat": "gj", "gj": "gj",
    "pscimbng": "wb", "wb": "wb", "telangana": "tg", "telmgan": "tg", "tg": "tg", "ts": "tg",
    "haryana": "hr", "hriyana": "hr", "hr": "hr", "rajasthan": "rj", "rajsthan": "rj",
    "rj": "rj", "kerala": "kl", "kerlm": "kl", "kl": "kl", "bihar": "br", "br": "br",
    "mp": "mp", "amdhrprdes": "ap", "ap": "ap", "punjab": "pb", "pmjab": "pb", "pb": "pb",
    "odisha": "od", "orissa": "od", "odisa": "od", "od": "od", "jharkhand": "jh",
    "jh": "jh", "chhattisgarh": "cg", "cg": "cg", "uttarakhand": "uk", "uttaranchal": "uk",
    "uk": "uk", "hp": "hp", "goa": "ga", "ga": "ga", "jk": "jk", "chandigarh": "chd",
    "puducherry": "py", "pondicherry": "py", "py": "py", "assam": "asm", "tripura": "tr",
    "meghalaya": "ml", "manipur": "mn", "nagaland": "nl", "mizoram": "mz", "sikkim": "sk",
}  # fmt: skip

STREET_FORMS: dict[str, str] = {
    # English
    "street": "st", "str": "st", "st": "st",
    "road": "rd", "rd": "rd", "marg": "rd", "salai": "rd",
    "avenue": "ave", "ave": "ave", "av": "ave", "avn": "ave",
    "boulevard": "blvd", "blvd": "blvd", "bd": "blvd", "boul": "blvd",
    "drive": "dr", "dr": "dr",
    "lane": "ln", "ln": "ln",
    "court": "ct", "ct": "ct",
    "place": "pl", "pl": "pl",
    "square": "sq", "sq": "sq",
    "highway": "hwy", "hwy": "hwy",
    "parkway": "pkwy", "pkwy": "pkwy",
    "expressway": "expy", "expy": "expy",
    "terrace": "ter", "ter": "ter",
    "circle": "cir", "cir": "cir",
    "suite": "ste", "ste": "ste",
    "apartment": "apt", "apt": "apt", "appartement": "apt",
    "floor": "fl", "flr": "fl", "fl": "fl", "etage": "fl",
    "building": "bldg", "bldg": "bldg", "batiment": "bldg", "bat": "bldg",
    "north": "n", "south": "s", "east": "e", "west": "w",
    "northeast": "ne", "northwest": "nw", "southeast": "se", "southwest": "sw",
    "mount": "mt", "mt": "mt", "saint": "st", "sainte": "st", "fort": "ft",
    # India
    "nagar": "ngr", "ngr": "ngr",
    "extension": "extn", "extn": "extn", "ext": "extn",
    "sector": "sec", "sec": "sec",
    "block": "blk", "blk": "blk",
    "phase": "ph", "ph": "ph",
    "stage": "stg", "stg": "stg",
    "cross": "crs", "crs": "crs",
    "layout": "lyt", "lyt": "lyt",
    "colony": "col", "col": "col",
    "opposite": "opp", "opp": "opp", "facing": "opp",
    "nr": "near", "near": "near", "beside": "near", "besides": "near",
    "adjacent": "near", "adj": "near",
    # France
    "rue": "rue", "r": "rue",
    "chemin": "ch", "ch": "ch",
    "allee": "all", "all": "all",
    "impasse": "imp", "imp": "imp",
    "route": "rte", "rte": "rte",
    "faubourg": "fbg", "fbg": "fbg",
    "quai": "quai", "cedex": "cedex",
    # ordinals
    "first": "1st", "second": "2nd", "third": "3rd", "fourth": "4th", "fifth": "5th",
    "sixth": "6th", "seventh": "7th", "eighth": "8th", "ninth": "9th", "tenth": "10th",
}  # fmt: skip

US_STATES: dict[str, str] = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia",
    "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn", "mississippi": "ms",
    "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv", "ohio": "oh",
    "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa", "tennessee": "tn", "texas": "tx",
    "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa", "wisconsin": "wi",
    "wyoming": "wy",
}  # fmt: skip

# Renamed or alternately romanised Indian cities: spelling variants, not a lookup.
CITY_FORMS: dict[str, str] = {
    "bangalore": "bengaluru", "bengalooru": "bengaluru", "bombay": "mumbai",
    "madras": "chennai", "calcutta": "kolkata", "poona": "pune", "gurgaon": "gurugram",
    "mysore": "mysuru", "mangalore": "mangaluru", "trivandrum": "thiruvananthapuram",
    "cochin": "kochi", "baroda": "vadodara", "benares": "varanasi", "banaras": "varanasi",
    "pondicherry": "puducherry", "calicut": "kozhikode", "vizag": "visakhapatnam",
    "simla": "shimla", "belgaum": "belagavi", "hubli": "hubballi", "gauhati": "guwahati",
}  # fmt: skip

LANDMARK_WORDS = frozenset({"near", "opp", "behind", "derriere"})
# "null"/"<NULL>" placeholders, unit markers whose number is what matters, and
# "City"/"Township" suffixes that come and go on the same place name.
DROP = frozenset(
    {"no", "num", "number", "nos", "null", "unit", "pmb", "city", "township", "twp", "none"}
)
_BASE_MAP = {**STREET_FORMS, **CITY_FORMS}
_MAPS = {
    "united states": ({**_BASE_MAP, **US_STATES}, _US_PHRASES),
    "india": ({**_BASE_MAP, **INDIA_STATES}, _INDIA_PHRASES),
}
_DEFAULT_MAP = ({**STREET_FORMS, **US_STATES, **CITY_FORMS}, _US_PHRASES)
_BASE_ONLY = (_BASE_MAP, [])


@dataclass(frozen=True)
class NormalizedAddress:
    clean: str  # every component, canonical tokens
    core: str  # without landmark phrases and postal codes
    landmark: str  # landmark phrases ("near sbi atm"), if any
    postal: str  # postal codes found, sorted, space separated
    numbers: str  # other numbers (house, unit, street), sorted unique, leading zeros stripped
    first: str = ""  # first number in reading order: usually the house number


def extract_postal_codes(raw: str) -> list[str]:
    """5- or 6-digit codes (US ZIP, French CP, Indian PIN incl. "560 001").

    A 5-6 digit number at the very start is a house number, not a code.
    """
    text = _PIN_SPACED.sub(r"\1\2", raw)
    leading = _LEADING_NUMBER.match(text)
    skip_at = leading.start(1) if leading else -1
    return [m.group(1) for m in _POSTAL.finditer(text) if m.start(1) != skip_at]


def _strip_zeros(token: str) -> str:
    return (token.lstrip("0") or "0") if token.isdigit() else token


@lru_cache(maxsize=1_000_000)
def normalize_address(raw: str, transliterate: bool = True, country: str = "") -> NormalizedAddress:
    """Normalise one address. ``country`` is a normalised label; it selects state maps only."""
    if not raw:
        return NormalizedAddress("", "", "", "", "")
    # No label: the historical default (US states). A known label: its own map.
    # Any other label (France, or a country never seen): street forms only.
    token_map, country_phrases = _MAPS.get(country, _DEFAULT_MAP if not country else _BASE_ONLY)
    postal = sorted(set(extract_postal_codes(raw)))
    postal_set = set(postal)

    all_tokens: list[str] = []
    core_tokens: list[str] = []
    landmark_tokens: list[str] = []
    for segment in _SEGMENT_SPLIT.split(_PIN_SPACED.sub(r"\1\2", raw)):
        text = replace_phrases(clean(segment, transliterate), _PHRASES)
        text = replace_phrases(text, country_phrases)
        tokens = canonical_tokens(text.split(), token_map, DROP)
        tokens = [t if t in postal_set else _strip_zeros(t) for t in tokens]
        if not tokens:
            continue
        all_tokens.extend(tokens)
        cut = next((i for i, t in enumerate(tokens) if t in LANDMARK_WORDS), None)
        kept, landmark = (tokens, []) if cut is None else (tokens[:cut], tokens[cut:])
        core_tokens.extend(t for t in kept if t not in postal_set)
        landmark_tokens.extend(landmark)

    core = " ".join(core_tokens)
    in_order = [n for n in _DIGITS.findall(core) if n not in postal_set]
    numbers = sorted(set(in_order), key=lambda n: (len(n), n))
    return NormalizedAddress(
        clean=" ".join(all_tokens),
        core=core,
        landmark=" ".join(landmark_tokens),
        postal=" ".join(postal),
        numbers=" ".join(numbers),
        first=in_order[0] if in_order else "",
    )


_COUNTRY_FORMS: dict[str, str] = {
    "us": "united states", "usa": "united states", "u s": "united states",
    "u s a": "united states", "united states of america": "united states",
    "america": "united states", "in": "india", "ind": "india", "bharat": "india",
    "fr": "france", "fra": "france", "republique francaise": "france",
}  # fmt: skip


@lru_cache(maxsize=10_000)
def normalize_country(raw: str) -> str:
    """Canonical country label. Unknown labels pass through cleaned; the set stays open."""
    text = clean(raw)
    return _COUNTRY_FORMS.get(text, text)
