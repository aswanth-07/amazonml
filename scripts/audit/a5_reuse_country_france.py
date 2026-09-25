# ruff: noqa
"""Holdout-optimism (name/address reuse train<->test), country-label vs content conflicts,
field swaps, truncation spikes, French elision, identical pool records with different owners."""

import os
import gc
import polars as pl

D = os.environ.get("BER_DATA_DIR", "dataset")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_width_chars(230)
pl.Config.set_fmt_str_lengths(80)


def norm(c):
    return pl.col(c).str.to_lowercase().str.replace_all(r"[^\p{L}\p{N}]+", " ").str.strip_chars()


def raw(split, s):
    return pl.read_csv(
        f"{D}/{split}/{split}_source{s}.tsv", separator="\t", quote_char=None, infer_schema=False
    ).fill_null("")


def keys(split, s):
    return raw(split, s).select(
        "entity_id",
        "country",
        norm("business_name").alias("nk"),
        norm("business_address").alias("ak"),
    )


# ---- 1. name / address reuse: is the holdout easier than test?
tr1, te1 = keys("train", 1), keys("test", 1)
trn = tr1.group_by("country", "nk").len()
print(
    "[reuse] train S1 whose exact name also names ANOTHER train S1:",
    tr1.join(trn, on=["country", "nk"])
    .group_by("country")
    .agg((pl.col("len") > 1).mean().round(4))
    .rows(),
)
print(
    "[reuse] test S1 whose exact name names a TRAIN S1:",
    te1.join(
        trn.select("country", "nk", pl.lit(True).alias("hit")), on=["country", "nk"], how="left"
    )
    .group_by("country")
    .agg(pl.col("hit").is_not_null().mean().round(4))
    .rows(),
)
ten = te1.group_by("country", "nk").len()
print(
    "[reuse] test S1 whose exact name names ANOTHER test S1:",
    te1.join(ten, on=["country", "nk"])
    .group_by("country")
    .agg((pl.col("len") > 1).mean().round(4))
    .rows(),
)
tra = (
    tr1.filter(pl.col("ak") != "")
    .select("country", "ak")
    .unique()
    .with_columns(pl.lit(True).alias("hit"))
)
print(
    "[reuse] test S1 whose exact address is a train S1 address:",
    te1.join(tra, on=["country", "ak"], how="left")
    .group_by("country")
    .agg(pl.col("hit").is_not_null().mean().round(4))
    .rows(),
)
both = te1.join(tr1, on=["country", "nk", "ak"])
print("[reuse] test S1 identical (name+addr) to a train S1:", both.height, both.head(3).rows())
# name-group sizes in test vs train (generic chain names)
for nm, d in (("train", trn), ("test", ten)):
    print(
        f"[names] {nm}: S1 name groups >=2: {d.filter(pl.col('len') > 1).height:,}; records in groups >=10:",
        d.filter(pl.col("len") >= 10)["len"].sum(),
        "; top:",
        d.sort("len", descending=True).head(6).rows(),
    )
print(
    "[names] test S1 exact dup name+addr:",
    te1.group_by("country", "nk", "ak").len().filter(pl.col("len") > 1).height,
)
del tr1, trn, tra, both
gc.collect()

# ---- 2. country label vs address content, field swaps, truncation (test, raw text)
te = pl.concat([raw("test", s).with_columns(pl.lit(s).alias("src")) for s in (1, 2, 3)])
a, n = pl.col("business_address").str.to_lowercase(), pl.col("business_name").str.to_lowercase()
te = te.with_columns(
    a.str.contains(
        r"\b(maharashtra|karnataka|tamil nadu|gujarat|rajasthan|uttar pradesh|kerala|telangana|west bengal|nagar|marg)\b|\b\d{3} ?\d{3}$"
    ).alias("indian"),
    a.str.contains(
        r"\b(rue|avenue de|boulevard de|chemin|allée|allee|impasse|cedex|place de)\b"
    ).alias("french"),
    a.str.contains(
        r"\b(al|ak|az|ar|ca|co|ct|fl|ga|il|ny|nj|tx|wa|or|pa|oh|mi|nc|va)\b \d{5}\b|\b(street|drive|lane|suite)\b"
    ).alias("usish"),
    n.str.contains(r"^\d+ [a-z]+ (st|street|rd|road|ave|avenue|drive|dr|lane|ln|rue)\b").alias(
        "addr_in_name"
    ),
    pl.col("business_name").str.len_chars().alias("nl"),
    pl.col("business_address").str.len_chars().alias("al"),
)
print(
    "\n[country vs content] share of records whose address looks Indian / French / US-ish, by label:"
)
print(
    te.group_by("country")
    .agg(
        pl.col("indian").mean().round(4),
        pl.col("french").mean().round(4),
        pl.col("usish").mean().round(4),
        pl.len(),
    )
    .sort("country")
)
for lab, col in (("France", "indian"), ("US", "french"), ("India", "french"), ("France", "usish")):
    x = (
        te.filter((pl.col("country") == lab) & pl.col(col))
        .select("entity_id", "business_name", "business_address")
        .head(4)
        .rows()
    )
    print(f"   {lab} labelled but {col}:", x)
print(
    "\n[field swap] names that look like a street address, by country/src:",
    te.group_by("country", "src")
    .agg(pl.col("addr_in_name").mean().round(5))
    .sort("country", "src")
    .rows(),
)
print(
    "   e.g.",
    te.filter(pl.col("addr_in_name")).select("business_name", "business_address").head(5).rows(),
)
for col in ("nl", "al"):
    c = te.group_by("country", col).len()
    tot = te.group_by("country").len().rename({"len": "tot"})
    mx = te.group_by("country").agg(
        pl.col(col).max().alias("max"), pl.col(col).quantile(0.999).alias("q999")
    )
    top = (
        c.join(tot, on="country")
        .with_columns((pl.col("len") / pl.col("tot")).alias("share"))
        .sort("len", descending=True)
    )
    print(
        f"[truncation] {col}: max / q99.9 by country:",
        mx.sort("country").rows(),
        "; most common exact lengths:",
        top.group_by("country").head(3).sort("country").select("country", col, "share").rows(),
    )

# ---- 3. French elision / apostrophes / arrondissement / cedex (test)
fr = te.filter(pl.col("country") == "France").with_columns(
    pl.col("business_address").str.to_lowercase().alias("a"),
    pl.col("business_name").str.to_lowercase().alias("n"),
)
pats = {
    "x'word (apostrophe)": r"\b[dlj]['’]\p{L}",
    "x word (elision spaced)": r"\b[dl] [aeiouhéè]\p{L}",
    "de la / du / des": r"\b(de la|du|des)\b",
    "curly apostrophe": r"’",
    "straight apostrophe": r"'",
    "cedex": r"\bcedex\b",
    "bp/cs": r"\b(bp|cs) ?\d+",
    "arrondissement": r"\b\d{1,2}(e|er|eme|ème)\b|arrondissement",
    "bis/ter": r"\b\d+ ?(bis|ter)\b",
    "5-digit CP": r"\b\d{5}\b",
    "st-/ste-": r"\bste?[- ]\p{L}",
    "saint/sainte": r"\bsainte?\b",
    "hyphenated place": r"\p{L}-\p{L}",
    "zi/za/zac": r"\b(zi|za|zac|z\.i\.)\b",
    "lieu-dit": r"lieu[- ]dit|\bld\b",
    "accented": r"[àâäçéèêëîïôöùûüÿœæ]",
}
rows = []
for k, p in pats.items():
    rows.append(
        (
            k,
            *[
                round(fr.filter(pl.col("src") == s)["a"].str.contains(p).mean(), 4)
                for s in (1, 2, 3)
            ],
            *[
                round(fr.filter(pl.col("src") == s)["n"].str.contains(p).mean(), 4)
                for s in (1, 2, 3)
            ],
        )
    )
print("\n[France] pattern share  (address S1,S2,S3 | name S1,S2,S3):")
for r in rows:
    print(
        f"   {r[0]:26s} addr {r[1]:.4f} {r[2]:.4f} {r[3]:.4f} | name {r[4]:.4f} {r[5]:.4f} {r[6]:.4f}"
    )
print(
    "   spaced-elision examples:",
    fr.filter(pl.col("a").str.contains(pats["x word (elision spaced)"]))
    .select("business_address")
    .head(6)
    .to_series()
    .to_list(),
)
print(
    "   apostrophe examples:",
    fr.filter(pl.col("a").str.contains(pats["x'word (apostrophe)"]))
    .select("business_address")
    .head(4)
    .to_series()
    .to_list(),
)
del te, fr
gc.collect()

# ---- 4. identical pool records owned by different S1 (train): what are they?
gt = pl.read_csv(
    f"{D}/train/train_ground_truth.tsv", separator="\t", quote_char=None, infer_schema=False
).fill_null("")
own = (
    gt.with_columns(pl.col("matched_entity_ids").str.split(","))
    .explode("matched_entity_ids")
    .filter(pl.col("matched_entity_ids") != "")
    .select(
        pl.col("matched_entity_ids").alias("entity_id"), pl.col("source1_entity_id").alias("s1")
    )
)
del gt
pool = pl.concat([keys("train", s) for s in (2, 3)]).join(own, on="entity_id", how="left")
del own
g = pool.group_by("country", "nk", "ak").agg(
    pl.len().alias("n"), pl.col("s1").drop_nulls().n_unique().alias("owners")
)
amb = g.filter(pl.col("owners") > 1)
print(
    f"\n[ties] identical pool records with different owners: {amb.height:,} groups, {amb['n'].sum():,} records; "
    f"empty address in {(amb['ak'] == '').mean():.3f} of groups"
)
print(
    "   non-empty-address examples:",
    amb.filter(pl.col("ak") != "").head(8).select("nk", "ak", "n", "owners").rows(),
)
