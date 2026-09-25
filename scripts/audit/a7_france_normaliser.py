# ruff: noqa
"""France: street-type stopwords, Ste/Sainte, added corporate words, accent noise, and how the
pipeline's own normaliser treats them (run from the repo's venv so ber is importable)."""

import os
import sys
import polars as pl

from ber.normalize.address import normalize_address  # noqa: E402
from ber.normalize.name import normalize_name  # noqa: E402

D = os.environ.get("BER_DATA_DIR", "dataset")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_width_chars(230)
pl.Config.set_fmt_str_lengths(90)


def fr(s):
    return (
        pl.read_csv(
            f"{D}/test/test_source{s}.tsv", separator="\t", quote_char=None, infer_schema=False
        )
        .fill_null("")
        .filter(pl.col("country") == "France")
        .with_columns(pl.lit(s).alias("src"))
    )


df = pl.concat([fr(1), fr(2), fr(3)])
a = pl.col("business_address").str.to_lowercase()
street = r"\b(rue|avenue|av|bd|boulevard|place|pl|allee|allée|impasse|imp|chemin|ch|route|rte|quai|cours|crs|square|passage) (de|du|des|d'|de la|de l')"
print(
    "street-type followed by de/du/des, by source:",
    df.group_by("src").agg(a.str.contains(street).mean().round(4)).sort("src").rows(),
)
print(
    "street-type immediately followed by a capitalised word (no article), by source:",
    df.group_by("src")
    .agg(
        pl.col("business_address")
        .str.contains(
            r"(?i:\b(rue|avenue|boulevard|place|allée|impasse|chemin|route|quai|cours)) [A-Z]"
        )
        .mean()
        .round(4)
    )
    .sort("src")
    .rows(),
)
print(
    "'Ste ' / 'Ste-' (Sainte abbreviated) by source:",
    df.group_by("src")
    .agg(pl.col("business_address").str.contains(r"\b(STE|Ste)[- ]").mean().round(4))
    .sort("src")
    .rows(),
)
print(
    "'Sainte' by source:",
    df.group_by("src").agg(a.str.contains(r"\bsainte\b").mean().round(4)).sort("src").rows(),
)

# how ber normalises a few French forms
for raw in [
    "12 Rue Sainte-Catherine, Bordeaux, Nouvelle-Aquitaine",
    "12 RUE STE CATHERINE, BORDEAUX, Gironde",
    "4 Cours de l'Intendance, Bordeaux",
    "4 CRS INTENDANCE, BORDEAUX",
    "8 Passage du Nord, Lille",
    "8 PASS DU NORD, LILLE",
    "3 Rue de la Paix, Nantes",
    "3 R PAIX, NANTES",
    "Appartement 12, 5 Rue X",
    "APP 12, 5 RUE X",
    "N° 39 Rue Violette",
    "Nº 39 R. Violette",
]:
    print(f"   addr {raw!r:55s} -> {normalize_address(raw, True, 'france').clean!r}")
for raw in [
    "Dupont SARL",
    "Dupont Développement",
    "Dupont Participations",
    "DUPONT HOLDING",
    "Çlub Spôrtive de Lille",
    "Club Sportive de Lille",
    '"""ehpad Club SAS"',
    "Ehpad Club SAS",
    "Société Générale",
    "Ste Generale",
    "Ets Martin",
    "Établissements Martin",
]:
    n = normalize_name(raw, True)
    print(f"   name {raw!r:30s} -> {n}")

# names: pool-only corporate words; do they sit on an S1 name root at the same city?
n = pl.col("business_name").str.to_lowercase()
corp = r"\b(d[ée]veloppement|participations|holding|distribution)\b"
print(
    "\nshare of names with développement/participations/holding/distribution by source:",
    df.group_by("src").agg(n.str.contains(corp).mean().round(4)).sort("src").rows(),
)
ex = df.filter(n.str.contains(corp) & (pl.col("src") > 1)).sample(12, seed=3)
s1 = df.filter(pl.col("src") == 1).with_columns(
    n.str.replace_all(r"[^\p{L} ]", "").str.split(" ").list.first().alias("root")
)
for r in ex.iter_rows(named=True):
    root = r["business_name"].lower().split()[0]
    hits = (
        s1.filter(pl.col("root") == root).head(3).select("business_name", "business_address").rows()
    )
    print(f"  S{r['src']} {r['business_name'][:40]:40s} | {r['business_address'][:60]}")
    for h in hits:
        print(f"        S1 {h[0][:40]:40s} | {h[1][:60]}")
