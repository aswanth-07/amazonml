# Problem statement (condensed)

Condensed working notes on the Amazon ML Challenge "Business Entity Resolution"
brief. The organisers' PDF is authoritative; this page records what the code
depends on and where each rule is enforced.

## Task

Three independent sources describe businesses with noisy, partial fields and
no shared identifier. **Source 1 is the deduplicated reference.** For every
Source 1 entity, find all Source 2 and Source 3 records that describe the same
real-world business. An entity can match zero, one or many records.

## Data

All files are **tab-separated** (addresses and id lists contain commas).

| File | Columns |
|---|---|
| `dataset/{train,test}/{split}_source{1,2,3}.tsv` | `entity_id`, `business_name`, `business_address`, `country` |
| `dataset/train/train_ground_truth.tsv` | `source1_entity_id`, `matched_entity_ids` (comma-joined, empty when none) |

- `entity_id` prefixes `S1-`, `S2-` and `S3-` identify the source.
- `country` is an **open set**. Train has US and India; **test adds France**,
  which is absent from train. Do not hard-code, filter or one-hot to
  {US, India}. Every test entity, France included, must be in the submission.
- Name noise: abbreviations (Corp/Corporation, Pvt/Private, Ltd/Limited),
  legal-suffix drift, DBA/trade names, `&` versus "and", word-order swaps,
  typos, transliterations.
- Address noise: Rd/Road and St/Street, transliteration variants, missing
  components (no PIN, no state), landmark references ("Near SBI ATM"),
  municipal numbering formats, reordered components.
- The test set has no ground truth. Validate on a held-out part of train.

## Outputs (both in `output/`, both TSV)

| File | Columns | Scored? |
|---|---|---|
| `matching_results.tsv` | `source1_entity_id`, `matched_entity_ids` | yes, the only scored file |
| `candidate_pairs.tsv` | `source1_entity_id`, `candidate_entity_ids` | no; audited for recall ceiling and reduction ratio |

Rules for both files. The local validator (`validate_outputs` in
`src/ber/submission.py`, run on every output the pipeline writes) enforces each one:

- exactly one row per test Source 1 entity, with no missing or duplicate rows
- an empty list for no matches or no candidates
- only S2 and S3 ids that exist in the test set; no S1 ids, no self-matches
- no duplicate ids within one list; ids joined by commas, with no quoting
- matches must be a subset of candidates (the organisers' validator warns otherwise)

`candidate_pairs.tsv` must be the **last** candidate set, exactly what the
matcher scores. The pipeline writes the capped union that feeds the model, so
the subset property holds by construction (`src/ber/scale.py`, `scripts/rule_variant.py`: the
candidate set is the pairs the decision is taken on).

The organisers ship `utils/validate_submission.py` (stdlib only); run it on the final files.

## Metric

**Macro F-beta with beta = 0.5**, computed per Source 1 entity and averaged over *all*
Source 1 entities, singletons included:

- F0.5 = 1.25 · P · R / (0.25 · P + R)
- no true matches and an empty prediction scores 1.0; no true matches and any prediction scores 0.0
- worked example: predicted {S2-00047, S2-00193, S3-00812}, truth {S2-00047, S3-00812} gives P = 2/3, R = 1, F0.5 = 0.714

The metric weighs precision about twice as heavily as recall: a false merge
costs more than a missed link. Implemented in `src/ber/metrics.py` and tested
against the worked example.

The public leaderboard uses a subset of test, the private one uses the rest,
and final ranking follows the private leaderboard. Predictions are always made
for the full test set.

## Final submission package

```
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/                 # all source code
│       ├── README.md            # exact end-to-end run instructions
│       └── requirements.txt     # pinned environment
└── Documentation_template.md    # filled-in methodology (.md or .pdf)
```

The code folder must regenerate both output files from the train and test data
alone. This repository is that code folder, plus its tests.

The methodology document must cover: methodology, the candidate
generation/blocking strategy, model architecture and feature engineering, and
anything else relevant. There is no page limit, and clarity and depth matter
more than brevity.

## Constraints and fair play

1. Outputs must match the format exactly. A submission that fails validation is not scored.
2. The final model must be **MIT or Apache-2.0 licensed, up to 8B parameters**.
   Here the matcher is LightGBM (MIT). The optional encoders are gated by
   `src/ber/pretrained.py`.
3. **External data lookup is strictly prohibited** and leads to disqualification:
   no entity-resolution APIs, government registries, geocoding, or internet data
   augmentation. `tests/test_fair_play.py` restricts imports to the standard
   library (without networking) and the pinned dependencies.
4. Top teams' packages are reviewed in detail before final ranking.

## Organisers' tips, and where the repo follows them

| Tip | Where |
|---|---|
| Strong blocking bounds recall | multi-blocker union, `oracle_fbeta` reported in every run |
| String similarity features (Jaccard, Levenshtein, TF-IDF cosine) | `src/ber/pairfeat.py` |
| Country-specific address patterns | `src/ber/normalize/address.py` |
| Precision-recall trade-off for F0.5 | `src/ber/tuning.py` rules tuned on the exact metric (`src/ber/decision.py`) |
| Do not neglect singletons | singletons scored in tuning; exclusive assignment; expected-F option |
| Validate the output format | `src/ber/submission.py` validates every file the pipeline writes |
