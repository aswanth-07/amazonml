# ruff: noqa
"""IDs and ground truth: formats, overlaps, list hygiene, exclusivity, cardinality, countries."""

import os
import polars as pl

D = os.environ.get("BER_DATA_DIR", "dataset")
pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_width_chars(200)
pl.Config.set_fmt_str_lengths(80)


def src(split, s, cols=("entity_id", "country")):
    return pl.read_csv(
        f"{D}/{split}/{split}_source{s}.tsv",
        separator="\t",
        quote_char=None,
        infer_schema=False,
        missing_utf8_is_empty_string=True,
        columns=list(cols),
    )


def ids(split):
    return pl.concat([src(split, s).with_columns(pl.lit(s).alias("src")) for s in (1, 2, 3)])


tr, te = ids("train"), ids("test")
for name, df in (("train", tr), ("test", te)):
    print(f"\n### {name}: rows per source", df.group_by("src").len().sort("src").rows())
    f = df.with_columns(
        pl.col("entity_id").str.extract(r"^(S\d)-").alias("pfx"),
        pl.col("entity_id").str.extract(r"^S\d-(.*)$").alias("num"),
    )
    print("  prefix vs file:", f.group_by("src", "pfx").len().sort("src").rows())
    print(
        "  id body is digits:",
        f.select(pl.col("num").str.contains(r"^\d+$").mean()).item(),
        " body lengths:",
        f.group_by("src", pl.col("num").str.len_chars().alias("L"))
        .len()
        .sort("src", "L")
        .rows()[:12],
    )
    print("  dup ids within split:", df.height - df["entity_id"].n_unique())
    print("  country x source:", f.group_by("country", "src").len().sort("country", "src").rows())
    raw_c = df["country"].unique().to_list()
    print("  distinct raw country labels:", sorted(raw_c)[:30])
both = tr.join(te, on="entity_id", how="inner")
print("\n### ids present in BOTH train and test:", both.height, both.head(5).rows())

gt = pl.read_csv(
    f"{D}/train/train_ground_truth.tsv",
    separator="\t",
    quote_char=None,
    infer_schema=False,
    missing_utf8_is_empty_string=True,
)
print("\n### ground truth", gt.shape, gt.columns)
s1 = tr.filter(pl.col("src") == 1)
print("  gt rows dup s1:", gt.height - gt["source1_entity_id"].n_unique())
print(
    "  gt s1 not in S1 file:",
    gt.join(s1, left_on="source1_entity_id", right_on="entity_id", how="anti").height,
)
print(
    "  S1 without gt row:",
    s1.join(gt, left_on="entity_id", right_on="source1_entity_id", how="anti").height,
)
lst = gt["matched_entity_ids"]
print(
    "  lists with spaces:",
    lst.str.contains(" ").sum(),
    " with ',,' or edge comma:",
    lst.str.contains(r",,|^,|,$").sum(),
    " with quotes:",
    lst.str.contains('"').sum(),
)
ex = (
    gt.with_columns(pl.col("matched_entity_ids").str.split(","))
    .explode("matched_entity_ids")
    .filter(pl.col("matched_entity_ids") != "")
    .rename({"source1_entity_id": "s1", "matched_entity_ids": "pid"})
)
print("  pairs:", ex.height, " dup pairs:", ex.height - ex.unique().height)
pool = tr.filter(pl.col("src") != 1)
print(
    "  matched ids not in pool:",
    ex.join(pool, left_on="pid", right_on="entity_id", how="anti").height,
    " matched ids that are S1:",
    ex.filter(pl.col("pid").str.starts_with("S1-")).height,
)
own = ex.group_by("pid").agg(pl.len().alias("owners"))
print("  pool ids with >1 owner:", own.filter(pl.col("owners") > 1).height)
card = ex.group_by("s1").agg(
    pl.len().alias("k"),
    pl.col("pid").str.starts_with("S2-").sum().alias("k2"),
    pl.col("pid").str.starts_with("S3-").sum().alias("k3"),
)
card = s1.join(card, left_on="entity_id", right_on="s1", how="left").fill_null(0)
print("  cardinality by country:")
print(card.group_by("country", "k").len().sort("country", "k"))
print(
    "  (k2,k3) combos:",
    card.group_by("k2", "k3").len().sort("len", descending=True).head(15).rows(),
)
x = ex.join(s1.rename({"entity_id": "s1", "country": "c1"}), on="s1").join(
    pool.rename({"entity_id": "pid", "country": "cp"}), on="pid"
)
print(
    "  cross-country true pairs:",
    x.filter(pl.col("c1") != pl.col("cp")).height,
    x.filter(pl.col("c1") != pl.col("cp")).head(5).rows(),
)
pm = pool.join(own, left_on="entity_id", right_on="pid", how="left").with_columns(
    pl.col("owners").is_not_null().alias("m")
)
print(
    "  pool matched share by country/source:",
    pm.group_by("country", "src").agg(pl.col("m").mean(), pl.len()).sort("country", "src").rows(),
)
for name, df in (("train", tr), ("test", te)):
    c = df.group_by("country", "src").len().pivot(on="src", index="country", values="len")
    print(
        f"  {name} pool per S1:",
        c.with_columns(((pl.col("2") + pl.col("3")) / pl.col("1")).alias("pool/S1")).rows(),
    )
# id number vs label: is there ordering signal left in ids (e.g. matched pool ids adjacent)?
xs = x.with_columns(
    pl.col("s1").str.extract(r"(\d+)$").cast(pl.Int64).alias("n1"),
    pl.col("pid").str.extract(r"(\d+)$").cast(pl.Int64).alias("np"),
)
print("  corr(id numbers) in true pairs:", xs.select(pl.corr("n1", "np")).item())
sib = (
    xs.filter(pl.col("pid").str.starts_with("S2-"))
    .sort("s1", "np")
    .with_columns(pl.col("np").diff().over("s1").alias("gap"))
)
print(
    "  gap between sibling S2 ids of the same S1 (quantiles):",
    sib["gap"].drop_nulls().quantile(0.01),
    sib["gap"].drop_nulls().median(),
    " random expectation ~",
    pool.height // 2,
)
