# ruff: noqa
"""Byte/line-level integrity of every TSV: BOM, CRLF, tabs per line, NULs, bad UTF-8,
control / invisible / odd-space characters, quotes, mojibake, placeholders."""

import os
import re, unicodedata
from collections import Counter
from pathlib import Path

D = Path(os.environ.get("BER_DATA_DIR", "dataset"))
files = sorted(D.glob("*/*.tsv"))
PAT = {
    "ctrl(excl tab)": re.compile(r"[\x00-\x08\x0b-\x1f\x7f]"),
    "zero-width/BOM": re.compile(r"[\u200b-\u200f\u2060\ufeff]"),
    "nbsp/odd space": re.compile(r"[\u00a0\u2000-\u200a\u202f\u205f\u3000]"),
    "soft hyphen": re.compile(r"\u00ad"),
    "replacement char": re.compile(r"\ufffd"),
    "mojibake": re.compile(r"Ã[\u0080-\u00bf]|Â[\u0080-\u00bf ]|â€"),
    "html entity": re.compile(r"&(amp|quot|apos|lt|gt|#\d+|nbsp);", re.I),
    "double quote": re.compile(r'"'),
    "backslash": re.compile(r"\\"),
    "combining mark": re.compile(r"[\u0300-\u036f]"),
    "fullwidth": re.compile(r"[\uff01-\uff5e]"),
    "curly quote/dash": re.compile(r"[\u2018\u2019\u201c\u201d\u2013\u2014]"),
    "lead/trail space": re.compile(r"(^|\t) | (\t|$)"),
    "double space": re.compile(r"  "),
}
PLACE = re.compile(
    r"(^|\t)(null|<null>|none|nan|n/a|na|-|\?|0|unknown|not available)(?=\t|$)", re.I
)
NL, CR, TAB = b"\n", b"\r", "\t"
for f in files:
    crlf = cr = nul = bad = n = nfc = 0
    tabs, place, ex = Counter(), Counter(), {}
    hits = dict.fromkeys(PAT, 0)
    with open(f, "rb") as fh:
        head = fh.readline()
        bom = head.startswith(b"\xef\xbb\xbf")
        header = head.decode("utf-8-sig").rstrip()
        last = b""
        for b in fh:
            last = b
            if b.endswith(CR + NL):
                crlf += 1
            cr += b.count(CR)
            nul += b.count(b"\x00")
            try:
                line = b.decode("utf-8")
            except UnicodeDecodeError:
                bad += 1
                line = b.decode("utf-8", errors="replace")
            line = line.rstrip("\n").rstrip("\r")
            if not line:
                continue
            n += 1
            tabs[line.count(TAB)] += 1
            for k, p in PAT.items():
                if p.search(line):
                    hits[k] += 1
                    ex.setdefault(k, line[:160])
            for m in PLACE.finditer(line):
                place[m.group(2).lower()] += 1
            if not line.isascii() and unicodedata.normalize("NFC", line) != line:
                nfc += 1
    print(f"\n=== {f.parent.name}/{f.name}  header={header!r}")
    print(
        f"  rows={n:,} BOM={bom} CRLF={crlf} CR={cr} NUL={nul} badUTF8lines={bad} endsNL={last.endswith(NL)}"
    )
    print(f"  tabs/line={dict(tabs)}  not-NFC lines={nfc}")
    for k, v in hits.items():
        if v:
            print(f"  {k:18s} {v:>9,}  e.g. {ex[k]!r}")
    if place:
        print(f"  placeholder fields: {dict(place.most_common(12))}")
