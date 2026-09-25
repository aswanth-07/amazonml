# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** PerryDaPlatypus
**Team Members:** A Aswanth Raj (team leader), V Nikhil, Nambi Rajan
**Submission Date:** TODO(date)

---

## 1. Executive Summary

A three-stage entity-resolution pipeline built for scale (2.2M Source 1 × 10.3M
Source 2/3 records per split):

1. Multi-view inverted-index blocking on a laptop CPU, with prefix filtering
   and full-cosine re-scoring.
2. A LightGBM matcher over 70+ string, token, number and group-context
   features.
3. A stacked second LightGBM that reads each first-stage probability in the
   context of the entity's other candidates and the record's other suitors.

Exclusive assignment enforces "each Source 2/3 record belongs to at most one
Source 1 entity". One threshold is tuned for the exact leaderboard macro F0.5,
including singletons.

---

## 2. Methodology

### 2.1 Problem Analysis

Findings from the training data:

- **Scale.** Train has 2,206,821 Source 1 and 10,320,219 Source 2/3 records;
  test has 1,732,544 and 9,969,589.
- **Structure.**
  - Source 1 is deduplicated. In the training truth no Source 2/3 id belongs
    to more than one Source 1 entity, so the mapping is exclusive.
  - Mean 3.46 true matches per entity. 5.6% of entities are singletons; 2.4%
    have 7 or more matches.
  - About 26% of Source 2/3 records match no Source 1 entity. These are the
    natural false-merge traps.
  - True pairs always share the country label.
- **Name noise.**
  - Legal forms dropped, abbreviated or moved to the front ("LLC Sapphire").
  - Honorifics ("Mr", "Dr", "Smt", "M/s") and junk prefixes ("--", "<<", "***").
  - Id tags ("#30473", "(ID: 96415)"), brackets, hyphenation, spacing.
  - Digit-for-letter typos ("G1obal", "6lobal") and ordinary typos.
  - Word-order swaps.
  - Former and trade names ("X fka Y", "X formerly known as Y", "X t/a Y",
    "X dba Y").
  - Whole-name domains and hashtags ("lynx.com", "#baskervillemeadows").
  - Accents injected into English names.
  - About 20% of Indian Source 2 names written in Indic scripts (Devanagari,
    Gurmukhi, Bengali, Gujarati, Tamil, Kannada, Telugu).
- **Address noise.**
  - Components reordered ("BELLEVILLE, IL, 57 WINDSOR DR").
  - Street-type abbreviations.
  - State as full name, postal code, or native script ("தமிழ்நாடு", "महाराष्ट्र").
  - `null` and `<NULL>` placeholders, zero-padded house numbers ("004",
    "002662"), unit markers ("Unit", "#", "PMB", "H.no", "Door No").
  - "City" and "Township" suffixes, renamed cities (Bangalore/Bengaluru).
  - House-number typos, and empty addresses.
- **Test.** Adds France (Bordeaux area), which is absent from train.

### 2.2 Solution Strategy

**Approach Type:** Blocking + GBDT classifier + stacked GBDT, with an
exclusive, metric-tuned decision.

**Core innovation:** candidate generation that stays exact while scaling, and
context-aware stacking.

- Each query searches name and address inverted indexes on its **rarest
  tokens up to a document-frequency budget** (prefix filtering). One pair of
  sparse products yields top-m by name, by address and by their weighted sum.
- Every generated pair is then **re-scored with the full (unfiltered)
  cosines** before selection. Pruned tokens therefore limit only which pairs
  are generated, never how they are ranked.
- A reverse search from each Source 2/3 record adds its best Source 1 suitor.
- Stage 2 reads the stage-1 probability in the context of its group. It asks
  whether this is the entity's best candidate and the record's best suitor,
  and by what margin. That is the question exclusivity asks.

---

## 3. Candidate Generation (Blocking)

- **Views.** Tokens are hashed, and IDF is fitted without labels on both sides
  of the split.
  - *Name:* core-name words, consonant-key codes, the squashed name, and
    character 4-grams of the squashed name.
  - *Address:* canonical address words and adjacent-word bigrams.

  The consonant key is a phonetic code robust to romanisation: "painiar" and
  "pioneer" both give `pnr`, and "sophtveyr" and "software" both give `sftvr`.
  The squashed name catches names like "dentalcare" and "pinktraders". The
  4-grams catch typos. Bigrams are rare where both of their words are common.
- **Search.** Exact inverted-index top-m through sparse matrix products,
  within each country (records with no label see every country). For each
  query, the rarest tokens are searched until their summed document frequency
  reaches 8,000, and tokens with pool df > 5,000 are never searched on. The
  name index gives top-50, the address index top-50, and their weighted sum
  (0.36 name + 0.64 address) top-100. The reverse direction (each Source 2/3
  record against the Source 1 index) gives top-5.
- **Re-score and select.** Every generated pair gets its full name, address
  and combined cosines. Kept per Source 1 entity: the top-20 by combined, top-8
  by name and top-8 by address; plus each Source 2/3 record's single best
  Source 1 suitor. The union is exactly what the matcher scores and what
  `candidate_pairs.tsv` lists.
- **Candidate pairs generated:** TODO(run) on test; TODO(run) per Source 1 entity.
- **How true matches were not lost:**
  - Name-only, address-only and combined searches, so a record with an empty
    address or a native-script name is still found through its other view.
  - Full re-scoring, so tokens excluded from the search still count in the
    ranking.
  - The reverse search.
  - Measured on train: generation recall TODO(run), candidate-set pair recall
    TODO(run), and the macro F0.5 a perfect matcher would reach on these
    candidates (oracle) TODO(run). With the pruned-only ranking we started
    from, recall@100 was 0.85.

---

## 4. Matching Model

**Features used (stage 1, 70+).**

- *Name:*
  - Levenshtein ratio, token-set, token-sort and partial ratios, Jaro-Winkler.
  - Consonant-key ratio and token-set ratio.
  - Squashed-name ratio and partial ratio.
  - Best similarity through a former or trade name.
  - Legal-form match, conflict or missing (bitmask).
  - Name lengths; native-script, domain/hashtag and name-marker flags.
- *Address:*
  - Ratio, token-set, token-sort and partial ratios; landmark similarity.
  - House-number equality; number-set Jaccard, shared count and conflict.
  - Postal-code equality; address lengths.
- *Vector:* TF-IDF cosine of the name view, the address view and their
  combination.
- *Group context:* for each of those cosines, the pair's rank, gap to the best
  and margin over the runner-up among the entity's candidates and among the
  record's suitors, plus both group sizes.
- *Other:* whether the candidate comes from Source 3. No feature encodes a
  country, so the model applies unchanged to France.

**Model type:**
- Stage 1: LightGBM (MIT licence) on the candidates of 250,000 randomly
  sampled training entities, with 4 folds grouped by entity. Every sampled pair
  gets an out-of-fold probability. Every other training pair is scored by the
  fold average, from models that never saw its entity.
- Stage 2 (stacking): LightGBM over the stage-1 probability and its
  entity-side and record-side rank, gap, margin, remaining mass and group size,
  plus ten stage-1 similarities. Folds are the same.
- No pretrained language model is used. The whole solution runs on CPU.

**Threshold selection method:**
- Exclusive assignment first: a Source 2/3 record is kept only by its
  highest-probability Source 1 suitor.
- Then one probability threshold, chosen by grid search to maximise the
  **exact leaderboard macro F0.5** on the sampled entities' out-of-fold
  predictions. Singletons are included, and true matches that blocking missed
  count against recall.
- The threshold is then reported on the roughly 1.95M held-out training
  entities, whose predictions come from models that never saw them.

---

## 5. Results & Error Analysis

- **F_0.5 score (macro):**
  - held-out training entities: TODO(run)
  - sampled entities (out-of-fold): TODO(run)
  - public leaderboard: TODO(lb)
- **By country (train):** TODO(run).
- **Common false positives (wrong merges):** TODO(run), from
  `runs/<run>/errors_false_positive.tsv`.
- **Common false negatives (missed matches):** TODO(run), from
  `runs/<run>/errors_false_negative.tsv`.

---

## 6. Conclusion

TODO(run)

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/`:

- `src/main.py`: entry point. `python src/main.py run --data-dir <dataset> --out-dir output`
  regenerates both output files from the raw TSVs.
- `src/ber/table.py`, `src/ber/normalize/`: parallel normalisation into
  compact tables.
- `src/ber/vectorize.py`, `src/ber/blocking/multiview.py`: hashed TF-IDF views,
  prefix filtering, per-country inverted-index search.
- `src/ber/scale.py`: candidate generation, re-scoring and selection, stage 1
  and stage 2 training, tuning, prediction.
- `src/ber/pairfeat.py`: vectorised pair and group-context features.
- `src/ber/decision.py`, `src/ber/metrics.py`: exclusive decision and the
  exact metric.
- `src/ber/submission.py`: writer and validator.
- `src/ber/configs/default.yaml`: every setting. The run used
  `configs/full.yaml` (reproduced in the README).

### B. Additional Results

Blocking recall at different generation budgets and selection sizes (20,000
sampled training entities, forward search only):

| Query budget (Σdf) | Generated recall | k 10/5/5 | k 15/6/6 | k 20/8/8 | k 30/10/10 |
|---|---|---|---|---|---|
| 3,000 | 0.963 | 0.941 (12/entity) | 0.948 (18) | 0.952 (24) | 0.955 (34) |
| 8,000 | 0.968 | 0.944 (12) | 0.951 (18) | 0.956 (24) | 0.959 (34) |

Earlier designs on the same sample:
- rare-token ranking only: recall@100 of 0.77 (df ≤ 2,000) and 0.85 (df ≤ 5,000)
- combined-view generation with re-scoring: recall@30 of 0.878 (df ≤ 5,000)
  and 0.934 (df ≤ 20,000, four times slower)
