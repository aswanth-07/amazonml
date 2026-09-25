# ruff: noqa
"""What are the extra test pool records?  Three independent estimators of the matched share,
decoy-type markers (branch words, added corporate words), and accent-noise alphabets."""

import os
import gc
import polars as pl

D = os.environ.get("BER_DATA_DIR", "dataset")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_width_chars(250)
pl.Config.set_fmt_str_lengths(70)
pl.Config.set_tbl_cols(-1)
BRANCH = r"\b(downtown|uptown|midtown|eastgate|westgate|northside|southside|lakeside|riverside|greater|central|east|west|north|south)\b"
CORP = r"\b(holding|holdings|participations|developpement|développement|distribution|gestion|invest|investissement|immobilier|services|group|groupe|enterprises|ventures|capital|partners|international|global|solutions)\b"
DIA = r"[À-ÖØ-öø-ÿĀ-ſ]"  # Latin letters with diacritics


def load(split, s):
    df = pl.read_csv(
        f"{D}/{split}/{split}_source{s}.tsv", separator="\t", quote_char=None, infer_schema=False
    ).fill_null("")
    n = pl.col("business_name").str.to_lowercase()
    return df.select(
        "entity_id",
        "country",
        pl.lit(s).alias("src"),
        (pl.col("business_address") == "").alias("ea"),
        n.str.contains(BRANCH).alias("branch"),
        n.str.contains(CORP).alias("corp"),
        pl.col("business_name").str.contains(DIA).alias("dia_name"),
        pl.col("business_name").str.extract_all(DIA).alias("dchars"),
        n.str.contains(r"\b(fka|formerly|dba|doing business as|t/a|aka)\b").alias("alias"),
    )


gt = pl.read_csv(
    f"{D}/train/train_ground_truth.tsv", separator="\t", quote_char=None, infer_schema=False
).fill_null("")
own = (
    gt.with_columns(pl.col("matched_entity_ids").str.split(","))
    .explode("matched_entity_ids")
    .filter(pl.col("matched_entity_ids") != "")
    .select(pl.col("matched_entity_ids").alias("entity_id"), pl.lit(True).alias("m"))
)
del gt
res = []
for split in ("train", "test"):
    for s in (1, 2, 3):
        df = load(split, s)
        if split == "train" and s > 1:
            df = df.join(own, on="entity_id", how="left").with_columns(pl.col("m").fill_null(False))
        else:
            df = df.with_columns(pl.lit(None, dtype=pl.Boolean).alias("m"))
        agg = df.group_by("country", "m").agg(
            pl.len().alias("n"),
            pl.col("ea").mean().round(4),
            pl.col("branch").mean().round(4),
            pl.col("corp").mean().round(4),
            pl.col("alias").mean().round(4),
            pl.col("dia_name").mean().round(4),
        )
        res.append(agg.with_columns(pl.lit(split).alias("split"), pl.lit(s).alias("src")))
        dc = (
            df.select("country", "dchars")
            .explode("dchars")
            .drop_nulls()
            .group_by("country", "dchars")
            .len()
            .sort("len", descending=True)
        )
        for c in dc["country"].unique().sort():
            top = dc.filter(pl.col("country") == c).head(14)
            print(
                f"[diacritics in NAMES] {split} S{s} {c}: "
                + " ".join(f"{a}:{b}" for a, b in top.select("dchars", "len").rows())
            )
        del df
        gc.collect()
tab = pl.concat(res).sort("country", "src", "split", "m")
print("\n", tab)

# matched-share estimates for the test pool from each marker: a*x_matched + (1-a)*x_unmatched = x_test
print(
    "\nimplied matched share of the TEST pool (train pool = 0.74), per marker and country/source:"
)
for c in ("US", "India"):
    for s in (2, 3):
        t = tab.filter((pl.col("country") == c) & (pl.col("src") == s))
        mt = t.filter((pl.col("split") == "train") & pl.col("m"))
        un = t.filter((pl.col("split") == "train") & ~pl.col("m"))
        te = t.filter(pl.col("split") == "test")
        out = []
        for col in ("ea", "branch", "corp", "alias", "dia_name"):
            xm, xu, xt = mt[col].item(), un[col].item(), te[col].item()
            out.append(f"{col}={((xt - xu) / (xm - xu)) if xm != xu else float('nan'):.3f}")
        print(
            f"   {c} S{s}: "
            + "  ".join(out)
            + f"   (if matches/S1 unchanged: {3.46 / (5.76 if c == 'US' else 5.82):.3f})"
        )
