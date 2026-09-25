# ruff: noqa
import os
import polars as pl

D = os.environ.get("BER_DATA_DIR", "dataset")
for split in ("train", "test"):
    for s in (1, 2, 3):
        df = pl.read_csv(
            f"{D}/{split}/{split}_source{s}.tsv",
            separator="\t",
            quote_char=None,
            infer_schema=False,
        ).fill_null("")
        r = (
            df.group_by("country")
            .agg(
                pl.len().alias("n"),
                pl.col("business_address").str.contains("°").sum().alias("deg_sign"),
                pl.col("business_name")
                .str.contains(r"[çÇàÀâÂèÈêÊëËîÎïÏôÔùÙûÛüÜ]")
                .sum()
                .alias("non_acute_name"),
                pl.col("business_name").str.contains(r"[áÁíÍóÓúÚ]").sum().alias("acute_aiou_name"),
            )
            .sort("country")
        )
        print(split, s, r.rows())
