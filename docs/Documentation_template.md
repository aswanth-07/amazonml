# Business Entity Resolution: Methodology

> **Draft.** Written before the dataset arrived. When the organisers'
> `Documentation_template.md` arrives, move this content into their headings
> and fill every `TODO(data)` with numbers from `runs/<run>/metrics.json`.
> This file is what `ber package` puts at the root of the submission zip.

## 1. Overview

The pipeline has five stages, each a separate module that can be tested and
measured on its own:

```
raw TSVs ─▶ normalise ─▶ candidate generation ─▶ pair features ─▶ LightGBM matcher ─▶ per-entity decision ─▶ outputs
            (names,      (union of blockers,      (76 features)     (GroupKFold OOF)     (exclusive, tuned
             addresses)   capped per S1)                                                  for macro F0.5)
```

The design follows from the metric. Macro F0.5 averages over every Source 1
entity, singletons included. A false merge on a singleton costs a full point
for that entity, and precision counts about twice as much as recall. So the
final decision is made **per Source 1 entity**, and its operating point is
**tuned on out-of-fold predictions against the exact leaderboard formula**.

Results (5-fold, grouped by Source 1 entity, on train):

| Quantity | Value |
|---|---|
| Blocking pair recall | TODO(data) |
| Blocking oracle F0.5 (perfect matcher on our candidates) | TODO(data) |
| Candidates per S1 entity / reduction ratio | TODO(data) |
| Out-of-fold macro F0.5 | TODO(data) |
| Out-of-fold F0.5 by country (US / India) | TODO(data) |
| Singleton accuracy | TODO(data) |
| Public leaderboard | TODO(data) |

## 2. Normalisation

Normalisation uses rules only, is symmetric across sources, and does no
external lookup (`src/ber/normalize/`).

- **Script and punctuation.** NFKC, transliteration of non-Latin scripts to
  ASCII (anyascii) with Latin accents folded, casefolding, `&` → "and",
  dotted acronyms joined (L.L.C. → llc), apostrophes dropped, other
  punctuation turned into spaces.
- **Names.** Multi-word legal forms are canonicalised (private limited → pvt
  ltd, société à responsabilité limitée → sarl), and single tokens map to a
  canonical short form (corporation → corp, limited → ltd). Legal forms are
  split off into their own field, so "Acme Corp" and "Acme Corporation" share
  a *core* name while the suffix remains evidence. Trade names after
  DBA/"trading as" become an alias field. The acronym of the core tokens is
  kept, and so is a **transliteration skeleton** (ksh→x, sh→s, ee→i,
  doubled letters collapsed). With the skeleton, Lakshmi/Laxmi,
  Shree/Sri/Shri and Venkateswara/Venkateshwara compare as equal.
- **Addresses.** The address is split into components on commas. Street types
  map to one canonical form: Road/Rd/Marg → rd, avenue/av → ave,
  boulevard/bd → blvd, French rue, chemin, allée, impasse. US state names
  map to their codes, and romanised Indian city variants (Bangalore →
  Bengaluru, Bombay → Mumbai) to one spelling. **Landmark phrases** ("near",
  "opp", "behind" and the rest of that component) go into a separate field so
  they do not dilute the street match. **Postal codes** are extracted as
  5-6 digit codes; a spaced PIN such as "560 001" is joined, and a leading
  5-digit house number is not taken for a ZIP. House and unit numbers go
  into their own set.
- **Country.** Aliases map to one label (US/USA/United States) and unknown
  labels pass through unchanged. The pipeline never enumerates countries.

## 3. Candidate generation (blocking)

`src/ber/blocking/`. Several complementary blockers run, and their union is
capped:

| Blocker | Signal | Why |
|---|---|---|
| `name_char` | char 2-4-gram TF-IDF on the core name, top-30 cosine | typos, transliteration, suffix drift |
| `addr_char` | char 2-4-gram TF-IDF on the core address, top-30 | same business, renamed or with an alias |
| `full_word` | word 1-2-gram TF-IDF on name + address, top-20 | word-order swaps, partial overlap |
| `postal_name` | same postal code and same first two letters of the name | cheap, high precision |
| `acronym` | acronym of one name equals the other name | IBM ↔ International Business Machines |
| `emb` (optional) | multilingual sentence embeddings, top-20 | semantic variants; off by default |

- TF-IDF is fit without labels on the records being matched. Top-k search is
  exact, over sparse vectors in memory-bounded dense chunks.
- Search is **partitioned by country**: an S1 record sees pool records with the
  same country label or no label. TODO(data): the audit's same-country rate
  for true pairs confirms this.
- The union is ranked by `0.6 · name cosine + 0.4 · address cosine +
  0.05 · number of blockers that proposed the pair`, then capped at
  **50 candidates per S1 entity**. The capped set is exactly what the matcher
  scores and what `candidate_pairs.tsv` reports.
- Every run reports per-blocker recall, recall before and after the cap, and
  the **oracle F0.5**, the score a perfect matcher would reach on these
  candidates. That number is the ceiling blocking imposes.

## 4. Features

`src/ber/features.py` (76 features). Each one compares two records or
places a pair among its competitors. **No feature encodes which country or
source a record comes from**, so the model transfers to France.

- **Name similarity** on the core name: Levenshtein ratio, partial ratio,
  token-sort and token-set ratios, Jaro-Winkler, normalised Levenshtein;
  token-set on the full cleaned name; skeleton similarity; token Jaccard and
  IDF-weighted Jaccard (rare shared tokens count more); first-token equality;
  acronym match; length ratio; best similarity through DBA aliases;
  legal-form match or conflict; conflict between numbers in names.
- **Address similarity**: the same string measures on the core address plus
  token-set on the full address; postal code match, conflict or missing;
  house/unit-number Jaccard, first-number equality and number conflict;
  landmark presence and landmark similarity; missing-address flag.
- **Whole record**: token-set on name + address; same country (NaN when
  either is unknown); whether the candidate is from S3.
- **Blocking evidence**: each blocker's cosine, hit flags, number of blockers,
  cap score and rank.
- **Competition features**, computed on both sides of the pair for cap score,
  name, address and whole-record similarity:
  - within the S1 entity's candidates: rank, gap to the best, margin over the
    runner-up
  - within the S2/S3 record's suitors: the same three

  These encode whether this S1 entity is the best explanation for the pool
  record, which suits a deduplicated reference source.

## 5. Matcher

`src/ber/matcher.py`. A LightGBM binary classifier (MIT licence, well under
the 8B-parameter cap) scores every candidate pair.

- **Grouped cross-fitting**: 5 folds, with all candidates of a Source 1 entity
  in one fold. Every pair gets an out-of-fold probability from a model that
  never saw that entity's labels. Early stopping runs per fold.
- The final model is refit on all training pairs for the mean best iteration
  × 1.1.
- TODO(data): hyper-parameters and feature importance, from
  `runs/<run>/feature_importance.tsv`.

## 6. Decision layer

`src/ber/decision.py`.

1. **Exclusive assignment.** Source 1 is deduplicated, so each S2/S3 record
   is kept only by its highest-scoring S1 suitor. TODO(data): the audit
   confirms that no S2/S3 id appears under two S1 entities in train.
2. **Operating point.** Either a probability threshold τ, or a per-entity
   expected-F rule. The expected-F rule keeps the top-j candidates that
   maximise estimated expected F0.5 and keeps none when P(no match) is
   higher; this protects singletons.
3. The single scalar is chosen by grid search on out-of-fold probabilities,
   scored with the **exact leaderboard formula**. That scoring includes true
   matches that blocking missed, so the tuning target is the real score and
   not a candidate-only proxy.

## 7. Validation protocol

- Out-of-fold scoring over all train S1 entities mirrors the test setting: one
  model, every S1 entity, the full S2/S3 pool.
- Every run writes `runs/<timestamp>-<tag>/` with the config, metrics (overall,
  per country, per true-match cardinality, blocking), the decision curve,
  feature importance, and side-by-side false-positive and false-negative
  tables. False negatives are labelled *missed by blocking* or *rejected by
  matcher*.
- A synthetic generator (`ber synth`) reproduces the file format and the noise
  patterns, with France only in test. It exercises the whole pipeline in the
  test suite. Synthetic scores only show that the pipeline works; they do not
  predict leaderboard results.

## 8. Rule compliance

- **No external data.** No API, registry, geocoder or web source is used.
  `tests/test_fair_play.py` holds the pipeline to an import allowlist: the
  standard library without its networking modules, plus the pinned
  dependencies. Anything else fails the build.
- **Model licence and size.** LightGBM is MIT. Optional encoders must be in
  `ber.pretrained.REGISTRY` (MIT or Apache-2.0, ≤ 8B parameters) and are
  loaded offline from local files.
- **Open country set.** There is no country one-hot or filter. France is
  normalised, blocked and matched like any other country.
- **Reproducibility.** Requirements are pinned, seeds fixed, LightGBM runs in
  deterministic mode, and one command regenerates both outputs
  (`code/business_entity_resolution/README.md`).

## 9. What we tried

TODO(data): ablations (each blocker, feature group, exclusive on/off,
threshold versus expected-F, cap size) with out-of-fold F0.5, from the
`runs/` records.
