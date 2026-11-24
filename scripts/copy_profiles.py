"""Operator profiles of real US copies, for the synthetic-French simulator.

For 40,000 random true (Source 1, copy) pairs of train's US businesses, detect which of the
generator's perturbations the copy shows (name: case, accent noise, wrapper brackets,
legal-form add/drop/change, typo, pool-heavy word, rebrand, word drop/replace/add, swap;
address: removal, upper case, reorder, fewer parts, number missing/changed/formatted).
``scripts/synth_france.py --profiles`` samples whole profiles, so the simulator reproduces
how the operators co-occur on real copies. Only train inputs and train labels are used.

    python scripts/copy_profiles.py --out artifacts/us_copy_profiles.json
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata

import numpy as np

from ber.embed import load_frames, truth_pairs
from ber.normalize.name import LEGAL_SET
from ber.rescue import first_numbers
from ber.vocab import pool_heavy

TOK = re.compile(r"[a-z0-9]+")
WRAP = re.compile(r"<<|\*\*\*|\[|\]|\(|\)|\{|\}")


def fold(s: str) -> str:
    return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()


def toks(s: str) -> list[str]:
    return TOK.findall(fold(s))


def lev1(a: str, b: str) -> bool:
    """One substitution, insertion, deletion or adjacent-letter swap apart."""
    if a == b or abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        diff = sum(x != y for x, y in zip(a, b))
        return diff == 1 or (sorted(a) == sorted(b) and diff == 2)
    s, long_ = (a, b) if len(a) < len(b) else (b, a)
    return any(long_[:i] + long_[i + 1 :] == s for i in range(len(long_)))


def name_ops(n1: str, n2: str, heavy: set[str]) -> dict[str, bool]:
    o = {}
    raw2 = str(n2)
    o["upper"] = raw2.isupper() and not str(n1).isupper()
    o["lower"] = raw2.islower() and not str(n1).islower()
    o["accent"] = any(ord(ch) > 127 for ch in raw2) and not any(ord(ch) > 127 for ch in str(n1))
    o["wrapper"] = bool(WRAP.search(raw2)) and not bool(WRAP.search(str(n1)))
    t1, t2 = toks(n1), toks(n2)
    l1, l2 = [t for t in t1 if t in LEGAL_SET], [t for t in t2 if t in LEGAL_SET]
    o["legal_add"] = bool(l2) and not l1
    o["legal_drop"] = bool(l1) and not l2
    o["legal_change"] = bool(l1) and bool(l2) and set(l1) != set(l2)
    c1, c2 = [t for t in t1 if t not in LEGAL_SET], [t for t in t2 if t not in LEGAL_SET]
    s1, s2 = set(c1), set(c2)
    add, drop = s2 - s1, s1 - s2
    typo = {a for a in drop if any(lev1(a, b) for b in add)}
    typo_b = {b for b in add if any(lev1(a, b) for a in drop)}
    add2, drop2 = add - typo_b, drop - typo
    o["typo"] = bool(typo)
    o["heavy"] = bool(add2 & heavy)
    add3 = add2 - heavy
    o["rebrand"] = bool(s1) and not (s1 & s2) and not typo
    o["drop"] = bool(drop2) and not add3 and not o["rebrand"]
    o["replace"] = bool(drop2) and bool(add3) and not o["rebrand"]
    o["add"] = bool(add3) and not drop2 and not o["rebrand"]
    o["swap"] = c1 != c2 and sorted(c1) == sorted(c2)
    return o


def addr_ops(a1: str, a2: str) -> dict[str, bool]:
    o = {}
    a1, a2 = str(a1 or ""), str(a2 or "")
    o["addr_removed"] = not a2.strip()
    if o["addr_removed"]:
        return o
    o["addr_upper"] = a2.isupper() and not a1.isupper()
    p1 = [c.strip().lower() for c in a1.split(",")]
    p2 = [c.strip().lower() for c in a2.split(",")]
    o["reorder"] = len(p1) > 1 and fold(p1[0])[:4] != fold(p2[0])[:4]
    o["addr_fewer_parts"] = len(p2) < len(p1)
    n1, n2 = first_numbers([a1])[0], first_numbers([a2])[0]
    o["num_missing"] = bool(n1) and not n2
    o["num_changed"] = bool(n1) and bool(n2) and n1 != n2
    o["num_format"] = bool(re.search(r"n°|no\.|#|\(\d|\d+\s*-\s|^0\d", a2.lower())) and not bool(
        re.search(r"n°|no\.|#", a1.lower())
    )
    return o


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="student_resource/dataset")
    ap.add_argument("--n", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    s1, pool = load_frames(args.data_dir, "train")
    c1 = s1.country.str.strip().str.lower().to_numpy()
    cp = pool.country.str.strip().str.lower().to_numpy()
    heavy = pool_heavy(s1.business_name.to_numpy(), c1, pool.business_name.to_numpy(), cp)
    a, b = truth_pairs(args.data_dir, s1, pool)[:2]
    a, b = np.asarray(a), np.asarray(b)
    idx = np.flatnonzero(c1[a] == "us")
    idx = np.random.default_rng(args.seed).choice(idx, args.n, replace=False)
    prof = []
    for i in idx:
        o = name_ops(s1.business_name.iat[a[i]], pool.business_name.iat[b[i]], heavy["us"])
        o.update(addr_ops(s1.business_address.iat[a[i]], pool.business_address.iat[b[i]]))
        prof.append({k: bool(v) for k, v in o.items()})
    with open(args.out, "w") as f:
        json.dump(prof, f)
    print(f"wrote {len(prof):,} profiles to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
