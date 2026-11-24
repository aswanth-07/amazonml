"""One submission from two runs' outputs, chosen per Source 1 country.

Each Source 1 entity takes its matches *and* its candidate pairs from the run assigned to its
country, so candidate_pairs.tsv stays exactly the set that entity's decision was taken on.
Pool records are country-specific, so the two runs never compete for one record; the script
checks that no pool record ends up matched twice.

    python scripts/combine_by_country.py --default output-v9-f --take france=output-v6a-rdb2 \
        --out-dir output-hybrid
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ber.embed import load_frames
from ber.normalize.name import LEGAL_FORMS
from ber.normalize.text import fold_latin
from ber.rescue import first_numbers
from ber.submission import write_outputs
from ber.vocab import name_tokens

_WORD = re.compile(r"[a-z]+")


def read(path: Path) -> dict[str, list[str]]:
    m = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    return {a: [x for x in s.split(",") if x] for a, s in zip(m.iloc[:, 0], m.iloc[:, 1])}


def legal_forms(name: str) -> frozenset[str]:
    return frozenset(LEGAL_FORMS[w] for w in _WORD.findall(fold_latin(name).lower().replace(".", ""))
                     if w in LEGAL_FORMS)  # fmt: skip


def pair_probs(scores: str, pairs: list[tuple[str, str]]) -> np.ndarray:
    """The run's probability of each (Source 1 id, pool id) pair (0 when it was not scored)."""
    z = np.load(scores)
    i1, ip = pd.Index(z["s1_ids"]), pd.Index(z["pool_ids"])
    key = z["rows"].astype(np.int64) * len(ip) + z["cols"]
    order = np.argsort(key)
    key = key[order]
    want = (i1.get_indexer([a for a, _ in pairs]).astype(np.int64) * len(ip)
            + ip.get_indexer([b for _, b in pairs]))  # fmt: skip
    pos = np.minimum(np.searchsorted(key, want), len(key) - 1)
    return np.where(key[pos] == want, z["p"][order[pos]], 0.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--default", required=True, help="output dir for every other country")
    ap.add_argument("--take", action="append", default=[], help="country=output_dir")
    ap.add_argument(
        "--lookalike-veto",
        default="",
        help="output dir of a run whose decision a planted-lookalike-shaped match of the default "
        "run must share (house number moved 1-9, the Source 1 name tokens all kept)",
    )
    ap.add_argument(
        "--legal-veto",
        default="",
        help="country list (e.g. us) for the legal-form lookalike veto; needs --lookalike-veto "
        "and --scores",
    )
    ap.add_argument("--scores", default="", help="the default run's test_scores.npz")
    ap.add_argument("--data-dir", default="student_resource/dataset")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    if args.legal_veto and not (args.lookalike_veto and args.scores):
        ap.error("--legal-veto needs --lookalike-veto and --scores")
    s1, pool = load_frames(args.data_dir, "test")
    country = dict(zip(s1["entity_id"], s1["country"].fillna("").str.strip().str.lower()))
    take = dict(t.split("=", 1) for t in args.take)
    runs = {d: (read(Path(d) / "matching_results.tsv"), read(Path(d) / "candidate_pairs.tsv"))
            for d in {args.default, *take.values()}}  # fmt: skip
    matches, candidates = {}, {}
    for sid in s1["entity_id"]:
        d = take.get(country[sid], args.default)
        m, c = runs[d]
        matches[sid] = m.get(sid, [])
        candidates[sid] = c.get(sid, [])
    if args.lookalike_veto:
        # v9's synthetic-French scorer accepts the generator's planted lookalikes (a legal
        # word added, house number moved 1-9) that train labels say are 2.4% true and v6a
        # rejects: such a match of the default run stands only if the veto run made it too
        veto = read(Path(args.lookalike_veto) / "matching_results.tsv")
        n1 = dict(zip(s1["entity_id"], first_numbers(s1["business_address"].fillna("").to_numpy())))
        t1 = dict(zip(s1["entity_id"], (name_tokens(x) for x in s1["business_name"])))
        pnum = dict(
            zip(pool["entity_id"], first_numbers(pool["business_address"].fillna("").to_numpy()))
        )
        pname = dict(zip(pool["entity_id"], pool["business_name"].fillna("")))
        dropped = 0
        for sid, ms in matches.items():
            if country[sid] in take:
                continue
            keep_v = set(veto.get(sid, []))
            kept = []
            for q in ms:
                a, b = n1[sid], pnum[q]
                look = (
                    a and b and a != b and abs(int(a[:15]) - int(b[:15])) <= 9
                    and t1[sid] and t1[sid] <= name_tokens(pname[q])
                )  # fmt: skip
                if look and q not in keep_v:
                    dropped += 1
                    continue
                kept.append(q)
            matches[sid] = kept
        print(f"lookalike veto: {dropped:,} matches dropped")
    if args.legal_veto:
        # Test's US pool carries 3-4x train's candidates of one more planted shape: the
        # Source 1 name with a legal form added or changed and the house number changed.
        # Train's rate of true pairs there, scaled to test, is ~19% of what the default run
        # accepts: drop them when the number moved 1-9 (unless near certain), or moved
        # further and the veto run rejected them (for d >= 100 only when the name also
        # changed, where true copies with a new number are rarer than decoys).
        lv_countries = set(args.legal_veto.split(","))
        todo = [(sid, q) for sid, ms in matches.items()
                if country[sid] in lv_countries and country[sid] not in take for q in ms]  # fmt: skip
        pr = dict(zip(todo, pair_probs(args.scores, todo)))
        leg1 = {sid: legal_forms(n) for sid, n in zip(s1["entity_id"], s1["business_name"])}
        dropped = 0
        for sid, ms in matches.items():
            if country[sid] not in lv_countries or country[sid] in take:
                continue
            keep_v = set(veto.get(sid, []))
            kept = []
            for q in ms:
                a, b = n1[sid], pnum[q]
                la, lb = leg1[sid], legal_forms(pname[q])
                if not (a and b and a != b and lb and la != lb):
                    kept.append(q)
                    continue
                d = abs(int(a[:15]) - int(b[:15]))
                sub = bool(t1[sid]) and t1[sid] <= name_tokens(pname[q])
                if (d <= 9 and pr[(sid, q)] < 0.995) or (
                    d >= 10 and q not in keep_v and (d <= 99 or not sub)
                ):
                    dropped += 1
                    continue
                kept.append(q)
            matches[sid] = kept
        print(f"legal-form lookalike veto: {dropped:,} matches dropped")
    used: dict[str, str] = {}
    for sid, ms in matches.items():
        for q in ms:
            if q in used:
                raise ValueError(f"pool record {q} matched by {used[q]} and {sid}")
            used[q] = sid
            if q not in set(candidates[sid]):
                raise ValueError(f"{sid} matches {q} outside its candidates")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_outputs(out, matches, candidates, list(s1["entity_id"]))
    n = sum(len(v) for v in matches.values())
    print(f"wrote {out}: {n:,} matched pairs; candidates per S1 "
          f"{sum(len(v) for v in candidates.values()) / len(candidates):.3f}")  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
