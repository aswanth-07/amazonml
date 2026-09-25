# ruff: noqa
"""Train vs test shift: field profiles, name-key reachability (matched-share estimate),
orphan-cluster vs lone-decoy structure, and token vocabularies that only one side uses."""

import os
import gc
import polars as pl

D = os.environ.get("BER_DATA_DIR", "dataset")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_width_chars(230)
pl.Config.set_fmt_str_lengths(60)


def norm(c):
    return pl.col(c).str.to_lowercase().str.replace_all(r"[^\p{L}\p{N}]+", " ").str.strip_chars()


def load(split, s):
    df = pl.read_csv(
        f"{D}/{split}/{split}_source{s}.tsv", separator="\t", quote_char=None, infer_schema=False
    ).fill_null("")
    a, n = pl.col("business_address"), pl.col("business_name")
    return df.select(
        "entity_id",
        "country",
        norm("business_name").alias("nk"),
        norm("business_address").alias("ak"),
        (a == "").alias("ea"),
        n.str.len_chars().alias("nl"),
        a.str.len_chars().alias("al"),
        (~(n + a).str.contains(r"^[\x00-\x7f]*$")).alias("nonascii"),
        a.str.contains(r"\b\d{5}\b").alias("pc5"),
        a.str.contains(r"\b\d{6}\b|\b\d{3} \d{3}\b").alias("pc6"),
        a.str.to_lowercase()
        .str.contains(r"\b(near|opp|opposite|behind|beside|nr)\b")
        .alias("landmark"),
        n.str.contains(r"\d").alias("digit_name"),
        a.str.count_matches(",").alias("commas"),
        pl.lit(s).alias("src"),
    )


prof = []
tok_tables = {}
for split in ("train", "test"):
    s1 = load(split, 1)
    pool = pl.concat([load(split, 2), load(split, 3)])
    both = pl.concat([s1, pool])
    prof.append(
        both.group_by("country", "src")
        .agg(
            pl.len().alias("n"),
            pl.col("ea").mean().round(4),
            pl.col("nl").median(),
            pl.col("al").median(),
            pl.col("nonascii").mean().round(4),
            pl.col("pc5").mean().round(3),
            pl.col("pc6").mean().round(3),
            pl.col("landmark").mean().round(3),
            pl.col("digit_name").mean().round(3),
            pl.col("commas").mean().round(2),
        )
        .with_columns(pl.lit(split).alias("split"))
    )
    # name-key reachability and sibling structure
    s1k = s1.select("country", "nk").unique().with_columns(pl.lit(True).alias("in_s1"))
    sib = pool.group_by("country", "nk").len().rename({"len": "sib"})
    p = (
        pool.join(s1k, on=["country", "nk"], how="left")
        .with_columns(pl.col("in_s1").fill_null(False))
        .join(sib, on=["country", "nk"])
    )
    if split == "train":
        gt = pl.read_csv(
            f"{D}/train/train_ground_truth.tsv", separator="\t", quote_char=None, infer_schema=False
        ).fill_null("")
        own = (
            gt.with_columns(pl.col("matched_entity_ids").str.split(","))
            .explode("matched_entity_ids")
            .filter(pl.col("matched_entity_ids") != "")
            .select(pl.col("matched_entity_ids").alias("entity_id"), pl.lit(True).alias("m"))
        )
        p = p.join(own, on="entity_id", how="left").with_columns(pl.col("m").fill_null(False))
        print("\n[train] exact name key found in S1 (same country), by matched/unmatched:")
        print(
            p.group_by("country", "m")
            .agg(
                pl.col("in_s1").mean().round(4),
                (pl.col("sib") >= 2).mean().round(4).alias("has_sibling"),
                pl.len(),
            )
            .sort("country", "m")
        )
        print(
            "[train] unmatched records NOT name-found in S1: share with a same-name sibling:",
            p.filter(~pl.col("m") & ~pl.col("in_s1"))
            .group_by("country")
            .agg((pl.col("sib") >= 2).mean().round(4))
            .rows(),
        )
        print(
            "[train] matched records NOT name-found in S1: share with a same-name sibling:",
            p.filter(pl.col("m") & ~pl.col("in_s1"))
            .group_by("country")
            .agg((pl.col("sib") >= 2).mean().round(4))
            .rows(),
        )
    else:
        print("\n[test] exact name key found in S1 (same country):")
        print(
            p.group_by("country")
            .agg(
                pl.col("in_s1").mean().round(4),
                (pl.col("sib") >= 2).mean().round(4).alias("has_sibling"),
                pl.len(),
            )
            .sort("country")
        )
        print(
            "[test] records NOT name-found in S1: share with a same-name sibling:",
            p.filter(~pl.col("in_s1"))
            .group_by("country")
            .agg((pl.col("sib") >= 2).mean().round(4))
            .rows(),
        )
    # token vocabularies (address + name) per country and side
    for field in ("ak", "nk"):
        for side, df in (
            ("s1", s1.sample(fraction=0.4, seed=0)),
            ("pool", pool.sample(fraction=0.2, seed=0)),
        ):
            t = (
                df.select("country", pl.col(field).str.split(" ").alias("t"))
                .explode("t")
                .filter(pl.col("t").str.contains(r"^\p{L}{2,}$"))
                .group_by("country", "t")
                .len()
            )
            tok_tables[(split, field, side)] = t
    del s1, pool, both, p
    gc.collect()

print("\n### field profiles")
print(pl.concat(prof).sort("country", "src", "split"))


def diff(a, b, label, country, min_a=80, top=40):
    x = (
        a.filter(pl.col("country") == country)
        .join(b.filter(pl.col("country") == country), on=["country", "t"], how="left", suffix="_b")
        .fill_null(0)
    )
    na, nb = x["len"].sum(), max(x["len_b"].sum(), 1)
    x = x.with_columns(((pl.col("len") / na) / ((pl.col("len_b") + 1) / nb)).alias("ratio")).filter(
        pl.col("len") >= min_a
    )
    print(f"\n[{label} | {country}] tokens over-represented (ratio = share here / share there):")
    print(
        ", ".join(
            f"{t}({n},{r:.0f}x)"
            for t, n, r in x.sort("ratio", descending=True)
            .head(top)
            .select("t", "len", "ratio")
            .rows()
        )
    )


for field in ("ak", "nk"):
    diff(
        tok_tables[("test", field, "pool")],
        tok_tables[("test", field, "s1")],
        f"{field}: test pool vs test S1",
        "France",
    )
    diff(
        tok_tables[("test", field, "s1")],
        tok_tables[("test", field, "pool")],
        f"{field}: test S1 vs test pool",
        "France",
    )
    for c in ("US", "India"):
        diff(
            tok_tables[("test", field, "pool")],
            tok_tables[("train", field, "pool")],
            f"{field}: TEST pool vs TRAIN pool",
            c,
        )
        diff(
            tok_tables[("test", field, "s1")],
            tok_tables[("train", field, "s1")],
            f"{field}: TEST S1 vs TRAIN S1",
            c,
        )
