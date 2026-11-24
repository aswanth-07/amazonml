"""Synthetic, correctly labelled French pairs from a replica of the data generator.

France has no labels in train, and pseudo labels inherit the teachers' mistakes. This
script replays the generator's copy operators, at the rates measured on true US copies in
train (name: case, legal form, pool-heavy word, word drop/replace/add/swap, typo, accent
noise, brackets, web-domain/hashtag/handle forms, rebrand; address: removal, missing or
changed house number (a digit dropped or added, as true copies do), number formats,
street-type abbreviation, region -> departement, city variants, component order, case,
street typos), on test's French Source 1 records. Negatives are the generator's decoys:
planted lookalikes (a legal or pool-heavy word *and* a house number moved by 1-9), other
businesses with the same name, and neighbours on the same street.

Writes a data split (``<out>/<split>/<split>_source{1,2,3}.tsv``) and labelled pairs
(``--pairs``, ``--eval-pairs`` for held-out businesses) for ``ber.crossenc train --pseudo``.
Only test's own Source 1 inputs and vocabulary mined from its pool names are used.

    python scripts/synth_france.py --data-dir student_resource/dataset --out artifacts/synth \
        --pairs artifacts/synth_fr_train.npz --eval-pairs artifacts/synth_fr_eval.npz
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from ber.embed import load_frames
from ber.vocab import pool_heavy

# rates measured on 30k true US copies in train (scratch analysis, 26 Sep)
RATE = dict(upper=0.131, lower=0.079, accent=0.076, wrapper=0.070, legal_add=0.075,
            legal_drop=0.191, legal_change=0.02, typo=0.052, heavy=0.094, rebrand=0.079,
            drop=0.089, replace=0.155, add=0.079, swap=0.033, addr_removed=0.047,
            addr_upper=0.436, reorder=0.457,
            fewer_parts=0.079, num_missing=0.072, num_changed=0.109, num_format=0.127,
            abbrev=0.35, dept=0.5, city_variant=0.3, street_typo=0.05)  # fmt: skip
# France's own address marginals (its pool records, 26 Sep): the US profile rates would give
# 43% upper-case addresses and 12% without a house number where France has 14% and 7%
FR_OVERRIDE = dict(addr_upper=0.14, num_missing=0.035, addr_removed=0.03, dept=0.33, abbrev=0.45,
                   lower=0.04)  # fmt: skip
LEGAL = ["SARL", "SAS", "SASU", "EURL", "SA", "SCI", "SNC", "EI", "Sarl", "S.A.S.", "S.A.R.L.",
         "Sàrl", "& Fils", "& Cie", "Et Fils", "& Frères", "Cie", "sarl", "sas"]  # fmt: skip
# a legal form ends at a non-word character or the end, so "S.A.S." is taken with its dot
LEGAL_RE = re.compile(r"\b(?:S\.?A\.?R\.?L\.?|S\.?A\.?S\.?U?\.?|EURL|SCI|SNC|EI|E\.I\.|SA|S[àa]rl|Cie)(?![\w])|&\s*(?:Fils|Cie|Fr[èe]res)|Et Fils", re.I)  # fmt: skip
HEAVY_FORMS = {"associes": "& Associés", "developpement": "Développement", "holding": "Holding",
               "international": "International", "distribution": "Distribution",
               "participations": "Participations"}  # fmt: skip
ABBREV = {"rue": ["R", "R.", "RUE"], "avenue": ["Av", "Av.", "Ave", "AV"], "boulevard": ["Bd", "Bd.", "BD", "Bld"],
          "allée": ["All", "All.", "Allee", "ALLEE"], "impasse": ["Imp", "Imp."], "place": ["Pl", "Pl."],
          "chemin": ["Ch", "Chem."], "route": ["Rte", "Rte."], "square": ["Sq", "SQ"], "cours": ["Crs"]}  # fmt: skip
DEPT = {"lille": "Nord", "tourcoing": "Nord", "roubaix": "Nord", "dunkerque": "Nord", "calais": "Pas-de-Calais",
        "bordeaux": "Gironde", "pessac": "Gironde", "mérignac": "Gironde", "merignac": "Gironde",
        "la teste-de-buch": "Gironde", "lège-cap-ferret": "Gironde", "lege-cap-ferret": "Gironde",
        "nantes": "Loire-Atlantique", "saint-nazaire": "Loire-Atlantique", "pornic": "Loire-Atlantique",
        "la baule-escoublac": "Loire-Atlantique", "saint-herblain": "Loire-Atlantique"}  # fmt: skip
REGIONS = {"hauts-de-france", "nouvelle-aquitaine", "pays de la loire"}
# what real copies add or substitute (US train: "Services", "Center", "Corporation",
# "Federation"; France's pool-heavy words play that part in its pool)
GENERIC = ["Services", "Centre", "Fédération"]  # "Group"/"Co"-like words mark planted lookalikes
REPLACE_WORDS = ["Services", "Centre"]  # what a replaced word becomes (US: "Service(s)", "Center")
LEGAL_LONG = {"sarl": "S.A.R.L.", "sas": "S.A.S.", "sasu": "S.A.S.U.", "eurl": "E.U.R.L.", "sa": "S.A.",
              "inc": "Incorporated", "llc": "L.L.C.", "corp": "Corporation", "co": "Company",
              "ltd": "Limited"}  # fmt: skip
ACCENTS = {"a": "àâ", "e": "éèêë", "i": "îï", "o": "ôö", "u": "ùûü", "A": "ÀÂ", "E": "ÉÈÊ"}


def fold(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()


def typo(w: str, rng: random.Random) -> str:
    if len(w) < 4:
        return w
    i = rng.randrange(1, len(w) - 1)
    op = rng.random()
    if op < 0.35:
        return w[:i] + w[i + 1 :]
    if op < 0.7:
        return w[:i] + rng.choice("abcdefghijklmnoprstuv") + w[i + 1 :]
    if op < 0.85:
        return w[:i] + w[i + 1] + w[i] + w[i + 2 :]
    return w[:i] + w[i] + w[i:]


def scramble(w: str, rng: random.Random) -> str:
    """Real copies' heavier misspellings ("Associates" -> "Aossaistes"): two inner swaps."""
    if len(w) < 5:
        return typo(w, rng)
    c = list(w)
    for _ in range(2):
        i, j = rng.sample(range(1, len(c) - 1), 2)
        c[i], c[j] = c[j], c[i]
    return "".join(c)


def reformat(core: list[str], rng: random.Random) -> str:
    """A rebrand in the generator is mostly the same name as a web domain or tag."""
    joined = re.sub(r"[^\w]", "", "".join(core))
    kind = rng.random()
    if kind < 0.7:
        out = fold(joined) + ".com"
        return out.upper() if rng.random() < 0.3 else out
    if kind < 0.85:
        return "#" + fold(joined)
    return "@" + joined


def accent(s: str, rng: random.Random) -> str:
    idx = [i for i, ch in enumerate(s) if ch in ACCENTS]
    if not idx:
        return s
    i = rng.choice(idx)
    return s[:i] + rng.choice(ACCENTS[s[i]]) + s[i + 1 :]


class Simulator:
    def __init__(
        self,
        heavy: list[str],
        vocab: list[str],
        invented: list[str],
        seed: int,
        profiles: list[dict] | None = None,
    ):
        self.rng = random.Random(seed)
        self.profiles, self.profile = profiles or [], {}
        self.heavy = [HEAVY_FORMS.get(h, h.capitalize()) for h in heavy]
        self.vocab, self.invented = vocab, invented

    def p(self, key: str) -> bool:
        """An operator fires as in the sampled real copy's profile when the profile records
        it (keeps the joint distribution: most real copies change one thing), else at its rate."""
        if key in FR_OVERRIDE:
            return self.rng.random() < FR_OVERRIDE[key]
        if key in self.profile:
            return bool(self.profile[key])
        return self.rng.random() < RATE[key]

    def sample(self) -> None:
        self.profile = self.rng.choice(self.profiles) if self.profiles else {}

    def copy(self, n: str, a: str, lookalike: bool = False) -> tuple[str, str]:
        self.sample()
        return self.name(n, lookalike), self.address(a, lookalike)

    # ---------------------------------------------------------------- name
    def name(self, n: str, lookalike: bool = False) -> str:
        r = self.rng
        legal = [m.group(0) for m in LEGAL_RE.finditer(n)]
        core = LEGAL_RE.sub(" ", n).split()
        if not core:
            core = n.split()
        if lookalike:  # the planted recipe: the name plus a legal or pool-heavy word
            if r.random() < 0.5 and self.heavy:
                core = [*core, r.choice(self.heavy)]
            else:
                legal = [r.choice([x for x in LEGAL if x not in legal])]
        elif self.p("rebrand"):
            if r.random() < 0.78:
                return reformat(core, r)
            core, legal = [r.choice(self.invented)], []
        else:
            generic = GENERIC
            if len(core) > 1 and self.p("drop"):
                core.pop(r.randrange(len(core)))
            if self.p("replace"):
                if r.random() < 0.5 or len(core) < 2:
                    i = r.randrange(len(core))
                    core[i] = scramble(core[i], r)
                else:
                    core.pop(r.randrange(len(core)))
                    core.append(r.choice(REPLACE_WORDS))
            if self.p("add"):
                w = r.choice(generic)
                core = [*core, w] if r.random() < 0.7 else [w, *core]
            if self.p("heavy"):  # how real copies carry the pool-heavy words
                kind = r.random()
                if kind < 0.35:  # a website suffix: "Name LLC | www.namefirst.com"
                    site = fold(re.sub(r"[^\w]", "", "".join(core[:2])))[:10]
                    legal = [*legal, f"| www.{site}.com"]
                elif kind < 0.55 and legal:  # a legal form spelled out or dotted
                    legal = [LEGAL_LONG.get(fold(legal[0]).replace(".", ""), legal[0])]
                elif kind < 0.85:  # an alias: "Invented DBA: Name" / "Invented formerly Name"
                    alias = r.choice(self.invented) if self.invented else "Nexcalo"
                    core = [
                        alias,
                        r.choice(
                            [
                                "DBA:",
                                "dba",
                                "formerly",
                                "formerly:",
                                "Formerly",
                                "formerly known as",
                            ]
                        ),
                        *core,
                    ]
                elif self.heavy:  # the country's own pool-heavy word, added as is
                    core.append(r.choice(self.heavy))
            if len(core) > 1 and self.p("swap"):
                r.shuffle(core)
            if self.p("typo"):
                i = r.randrange(len(core))
                core[i] = typo(core[i], r)
            if legal and self.p("legal_drop"):
                legal = []
            elif (not legal and self.p("legal_add")) or (legal and self.p("legal_change")):
                legal = [r.choice(LEGAL)]
        words = core + legal if r.random() < 0.85 else legal + core
        out = " ".join(words)
        if self.p("accent"):
            out = accent(out, r)
        if self.p("wrapper"):
            out = r.choice(["<< {}", "*** {}", "[{}]", "({})", "{} ***", "{}!"]).format(out)
        if self.p("upper"):
            out = out.upper()
        elif self.p("lower"):
            out = out.lower()
        return re.sub(r"\s+", " ", out).strip()

    # ---------------------------------------------------------------- address
    def number(self, num: str, lookalike: bool) -> str:
        r = self.rng
        if lookalike:
            return str(max(1, int(num) + r.choice([-1, 1]) * r.randint(1, 9)))
        if self.p("num_changed"):  # real copies: 5724 -> 724, 238 -> 23, 4537 -> 4538
            k = r.random()
            if len(num) >= 2 and k < 0.4:
                return num[1:].lstrip("0") or num
            if len(num) >= 2 and k < 0.75:
                return num[:-1]
            i = r.randrange(len(num))
            return num[:i] + str((int(num[i]) + r.randint(1, 8)) % 10) + num[i + 1 :]
        return num

    def address(self, a: str, lookalike: bool = False) -> str:
        r = self.rng
        if not lookalike and self.p("addr_removed"):
            return ""
        parts = [x.strip() for x in a.split(",") if x.strip()]
        out = []
        for part in parts:
            m = re.match(r"^(\d+)(\s*(?:bis|ter|[A-Za-z])?\b)?\s*(.*)$", part)
            if m:
                num, suf, rest = m.group(1), m.group(2) or "", m.group(3)
                words = rest.split()
                if words and self.p("abbrev"):
                    for k, vs in ABBREV.items():
                        if fold(k) == fold(words[0]):
                            words[0] = r.choice(vs)
                            break
                if len(words) > 1 and self.p("street_typo"):
                    i = r.randrange(1, len(words))
                    words[i] = typo(words[i], r)
                if not lookalike and self.p("num_missing"):
                    out.append(" ".join(words))
                    continue
                num = self.number(num, lookalike)
                if self.p("num_format"):
                    num = r.choice(
                        ["N° {}", "No. {}", "#{}", "({})", "{} -", "00{}", "NO. {}"]
                    ).format(num)
                out.append(f"{num}{suf} " + " ".join(words))
            elif fold(part) in REGIONS:
                if self.p("fewer_parts"):
                    continue
                if self.p("dept"):
                    city = next(
                        (fold(x) for x in parts if fold(x) in {fold(k) for k in DEPT}), None
                    )
                    dept = next((v for k, v in DEPT.items() if fold(k) == city), None)
                    out.append(dept or part)
                else:
                    out.append(part)
            else:
                if self.p("city_variant"):
                    part = r.choice([part.capitalize(), part.replace("Saint-", "St-"),
                                     part.replace("Saint-", "St.-"), fold(part).title(),
                                     part.replace("-", " ")])  # fmt: skip
                out.append(part)
        if len(out) > 1 and self.p("reorder"):
            r.shuffle(out)
        s = ", ".join(x for x in out if x)
        return s.upper() if self.p("addr_upper") else s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="student_resource/dataset")
    ap.add_argument("--out", required=True, help="data dir to write the synthetic split into")
    ap.add_argument("--split", default="synth")
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--eval-pairs", required=True)
    ap.add_argument("--copies", type=float, default=1.3, help="mean synthetic copies per business")
    ap.add_argument("--eval-share", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--profiles", default="", help="real copies' operator profiles (json list)")
    args = ap.parse_args()
    s1, pool = load_frames(args.data_dir, "test")
    c1 = s1["country"].fillna("").str.strip().str.lower().to_numpy()
    cp = pool["country"].fillna("").str.strip().str.lower().to_numpy()
    heavy = pool_heavy(s1["business_name"].to_numpy(), c1, pool["business_name"].to_numpy(), cp)
    fr = s1[c1 == "france"].reset_index(drop=True)
    fp = pool[cp == "france"]
    s1_words = {t for n in fr["business_name"] for t in re.findall(r"[a-z]+", fold(n))}
    # invented brand names look like "Veralumzeta": capitalised, not a concatenated domain
    invented = [str(n) for n in fp["business_name"]
                if str(n).isalpha() and len(str(n)) >= 6 and str(n)[0].isupper() and str(n)[1:].islower()
                and not str(n).endswith("com") and fold(str(n)) not in s1_words][:30000]  # fmt: skip
    vocab_counts = pd.Series([w for n in fr["business_name"] for w in str(n).split()
                              if w.isalpha() and len(w) > 2 and not LEGAL_RE.fullmatch(w)]).value_counts()  # fmt: skip
    vocab = vocab_counts.index[:3000].tolist()
    print(
        f"French S1 {len(fr):,}; heavy {sorted(heavy.get('france', []))}; invented names {len(invented):,}"
    )
    profiles = json.loads(Path(args.profiles).read_text()) if args.profiles else None
    sim = Simulator(sorted(heavy.get("france", [])), vocab, invented, args.seed, profiles)
    rng = np.random.default_rng(args.seed)
    names, addrs = (
        fr["business_name"].astype(str).to_numpy(),
        fr["business_address"].fillna("").astype(str).to_numpy(),
    )
    held = rng.random(len(fr)) < args.eval_share

    # neighbours (same street, other name) and namesakes (same name words, other business)
    def street_key(a):
        m = re.search(r"\d+\s*(?:bis|ter)?\s*([^,]+)", fold(a))
        return m.group(1).strip() if m else ""

    by_street, by_name = defaultdict(list), defaultdict(list)
    for i, (n, a) in enumerate(zip(names, addrs)):
        by_street[street_key(a)].append(i)
        by_name[" ".join(sorted(re.findall(r"[a-z]{3,}", fold(LEGAL_RE.sub(" ", n)))))].append(i)
    pool_rows, pairs = [], []  # pairs: (s1 index, pool index, label)

    def add_pool(n, a):
        pool_rows.append((f"S2-9{len(pool_rows):08d}", n, a, "France"))
        return len(pool_rows) - 1

    for i in range(len(fr)):
        k = rng.poisson(args.copies)
        for _ in range(max(1, k) if rng.random() < 0.95 else 0):
            pairs.append((i, add_pool(*sim.copy(names[i], addrs[i])), 1))
        if re.search(r"\d", addrs[i]) and rng.random() < 0.6:  # planted lookalike
            pairs.append((i, add_pool(*sim.copy(names[i], addrs[i], lookalike=True)), 0))
        for group in (by_name, by_street):
            key = (" ".join(sorted(re.findall(r"[a-z]{3,}", fold(LEGAL_RE.sub(" ", names[i]))))) if group is by_name
                   else street_key(addrs[i]))  # fmt: skip
            others = [j for j in group.get(key, []) if j != i]
            if others and rng.random() < 0.5:
                j = others[rng.integers(len(others))]
                pairs.append((i, add_pool(*sim.copy(names[j], addrs[j])), 0))
    out = Path(args.out) / args.split
    out.mkdir(parents=True, exist_ok=True)
    kw = {"sep": "\t", "index": False, "quoting": csv.QUOTE_NONE, "escapechar": "\\"}
    cols = ["entity_id", "business_name", "business_address", "country"]
    fr[cols].to_csv(out / f"{args.split}_source1.tsv", **kw)
    pd.DataFrame(pool_rows, columns=cols).to_csv(out / f"{args.split}_source2.tsv", **kw)
    pd.DataFrame(columns=cols).to_csv(out / f"{args.split}_source3.tsv", **kw)
    p = np.array(pairs, dtype=np.int64)
    ev = held[p[:, 0]]
    for path, sel in ((args.pairs, ~ev), (args.eval_pairs, ev)):
        np.savez(path, rows=p[sel, 0], cols=p[sel, 1], labels=p[sel, 2].astype(np.float32))
    print(f"{len(pool_rows):,} synthetic pool records; train pairs {int((~ev).sum()):,} "
          f"({p[~ev, 2].mean():.1%} positive), eval pairs {int(ev.sum()):,}")  # fmt: skip
    for i in range(12):
        s, q, y = pairs[i * 37]
        print(f"  [{y}] {names[s]} | {addrs[s]}\n       {pool_rows[q][1]} | {pool_rows[q][2]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
