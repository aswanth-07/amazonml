# ruff: noqa
"""Duplicates, exact-equal decoys, label-noise candidates, empty and swapped fields (train)."""

import os
import sys
import polars as pl

D = os.environ.get("BER_DATA_DIR", "dataset")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_width_chars(230)
pl.Config.set_fmt_str_lengths(70)
SPLIT = sys.argv[1] if len(sys.argv) > 1 else "train"


def src(s):
    return pl.read_csv(
        f"{D}/{SPLIT}/{SPLIT}_source{s}.tsv", separator="\t", quote_char=None, infer_schema=False
    ).fill_null("")


def norm(c):  # light key: casefold, punctuation to space, collapse
    return pl.col(c).str.to_lowercase().str.replace_all(r"[^\p{L}\p{N}]+", " ").str.strip_chars()


KEEP = ["entity_id", "country", "nk", "ak", "ea", "en"]
flags = [
    (pl.col("business_address") == "").alias("ea"),
    (pl.col("business_name") == "").alias("en"),
]
s1 = (
    src(1)
    .with_columns(norm("business_name").alias("nk"), norm("business_address").alias("ak"), *flags)
    .select(KEEP)
)
pool = pl.concat(
    [
        src(s)
        .with_columns(
            norm("business_name").alias("nk"), norm("business_address").alias("ak"), *flags
        )
        .select(KEEP)
        .with_columns(pl.lit(s).alias("src"))
        for s in (2, 3)
    ]
)
if SPLIT == "train":
    gt = pl.read_csv(
        f"{D}/train/train_ground_truth.tsv", separator="\t", quote_char=None, infer_schema=False
    ).fill_null("")
    ex = (
        gt.with_columns(pl.col("matched_entity_ids").str.split(","))
        .explode("matched_entity_ids")
        .filter(pl.col("matched_entity_ids") != "")
        .rename({"source1_entity_id": "s1", "matched_entity_ids": "entity_id"})
    )
    pool = pool.join(ex, on="entity_id", how="left").with_columns(pl.col("s1").fill_null("-"))
    s1 = s1.join(
        ex.group_by("s1").len().rename({"s1": "entity_id", "len": "k"}), on="entity_id", how="left"
    ).with_columns(pl.col("k").fill_null(0))

print(f"##### {SPLIT}")
print(
    "empty name S1:",
    s1["en"].sum(),
    " empty addr S1:",
    s1["ea"].sum(),
    " pool empty name:",
    pool["en"].sum(),
    " pool empty addr:",
    pool["ea"].sum(),
    " pool both empty:",
    (pool["en"] & pool["ea"]).sum(),
)
print(
    "name key empty after cleaning (only punctuation) S1/pool:",
    (s1["nk"] == "").sum(),
    (pool["nk"] == "").sum(),
)
print(
    "very short names (<=2 chars) S1/pool:",
    (s1["nk"].str.len_chars() <= 2).sum(),
    (pool["nk"].str.len_chars() <= 2).sum(),
)

# 1. exact duplicates inside S1 (the "deduplicated" source)
d1 = s1.group_by("country", "nk", "ak").agg(
    pl.len().alias("n"), pl.col("entity_id"), *([pl.col("k")] if SPLIT == "train" else [])
)
dd = d1.filter(pl.col("n") > 1)
print(
    f"\n[S1 exact dup name+addr] groups {dd.height:,}, records {dd['n'].sum() if dd.height else 0:,}"
)
print(dd.head(8))
dn = s1.filter(pl.col("nk") != "").group_by("country", "nk").len().filter(pl.col("len") > 1)
print(
    f"[S1 same name, any addr] groups {dn.height:,} records {dn['len'].sum():,}; biggest:",
    dn.sort("len", descending=True).head(8).rows(),
)

# 2. exact duplicates inside the pool
g = pool.group_by("country", "nk", "ak").agg(
    pl.len().alias("n"),
    *(
        [pl.col("s1").n_unique().alias("owners"), (pl.col("s1") == "-").sum().alias("unm")]
        if SPLIT == "train"
        else []
    ),
)
m = g.filter(pl.col("n") > 1)
print(
    f"\n[pool exact dup name+addr] groups {m.height:,}, records {m['n'].sum():,} of {pool.height:,}"
)
if SPLIT == "train":
    cat = m.with_columns(
        pl.when((pl.col("owners") == 1) & (pl.col("unm") == 0))
        .then(pl.lit("same owner"))
        .when(pl.col("unm") == pl.col("n"))
        .then(pl.lit("all unmatched"))
        .when(pl.col("unm") > 0)
        .then(pl.lit("mix matched+unmatched"))
        .otherwise(pl.lit("different owners"))
        .alias("cat")
    )
    print(cat.group_by("cat").agg(pl.len(), pl.col("n").sum()).sort("len", descending=True))

# 3. S1 == pool exactly (name+addr), and whether that is a labelled match
j = (
    s1.select("entity_id", "country", "nk", "ak")
    .rename({"entity_id": "sid"})
    .join(
        pool.select("entity_id", "country", "nk", "ak", *(["s1"] if SPLIT == "train" else [])),
        on=["country", "nk", "ak"],
    )
)
print(f"\n[S1 == pool exact name+addr] pairs {j.height:,}")
if SPLIT == "train":
    j = j.with_columns(
        (pl.col("sid") == pl.col("s1")).alias("true"), (pl.col("s1") == "-").alias("pool_unmatched")
    )
    print(j.group_by("true", "pool_unmatched").len().sort("len", descending=True))
    print("examples of exact-equal but NOT labelled:")
    bad = j.filter(~pl.col("true")).head(6)
    for r in bad.iter_rows(named=True):
        print(
            "   ",
            r["sid"],
            "|",
            r["nk"][:40],
            "|",
            r["ak"][:60],
            "-> pool",
            r["entity_id"],
            "owner",
            r["s1"],
        )
    jn = (
        s1.select("entity_id", "country", "nk")
        .rename({"entity_id": "sid"})
        .filter(pl.col("nk").str.len_chars() > 8)
        .join(pool.select("entity_id", "country", "nk", "s1"), on=["country", "nk"])
    )
    jn = jn.with_columns((pl.col("sid") == pl.col("s1")).alias("true"))
    print(
        f"[S1 == pool exact NAME only, len>8] pairs {jn.height:,}; true share {jn['true'].mean():.4f}"
    )

    # 4. label-noise candidates: true pairs with no shared name token and no shared address token
    t = (
        pool.filter(pl.col("s1") != "-")
        .select("entity_id", "s1", "nk", "ak", "src")
        .join(
            s1.select(
                pl.col("entity_id").alias("s1"),
                pl.col("nk").alias("nk1"),
                pl.col("ak").alias("ak1"),
            ),
            on="s1",
        )
    )
    tok = lambda c: (
        pl.col(c).str.split(" ").list.eval(pl.element().filter(pl.element().str.len_chars() >= 2))
    )
    t = t.with_columns(
        tok("nk").list.set_intersection(tok("nk1")).list.len().alias("nshare"),
        tok("ak").list.set_intersection(tok("ak1")).list.len().alias("ashare"),
        tok("nk").list.len().alias("nt"),
        tok("ak").list.len().alias("at"),
    )
    print(
        f"\n[true pairs] {t.height:,}; no shared name token: {(t['nshare'] == 0).mean():.4f}; "
        f"no shared addr token (both non-empty): {((t['ashare'] == 0) & (t['at'] > 0)).mean():.4f}; "
        f"neither: {((t['nshare'] == 0) & (t['ashare'] == 0)).mean():.5f} ({((t['nshare'] == 0) & (t['ashare'] == 0)).sum():,})"
    )
    nei = t.filter((pl.col("nshare") == 0) & (pl.col("ashare") == 0)).sample(
        min(25, t.height), seed=1
    )
    for r in nei.iter_rows(named=True):
        print(
            f"   S1 {r['nk1'][:38]:38s} | {r['ak1'][:55]:55s}\n      S{r['src']} {r['nk'][:38]:38s} | {r['ak'][:55]}"
        )
