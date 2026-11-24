# Methodology: Business Entity Resolution (Amazon ML Challenge 2026)

The methodology document submitted with the final package. File paths in the appendix are
this repository's; in the submission zip the scripts sat under `src/scripts/`.

**Team Name:** PerryDaPlatypus  
**Team Members:** A Aswanth Raj (team leader), V Nikhil, Nambi Rajan  
**Submission Date:** 28 September 2026

---

## 1. Executive Summary

We match 10.0M noisy Source 2/3 records to 1.7M Source 1 businesses with a blocking cascade
whose last stages are learned models: TF-IDF inverted-index search and a fine-tuned
multilingual bi-encoder propose about 39 candidates per record, and a LightGBM filter and two
fine-tuned cross-encoders (multilingual-e5-small, bge-reranker-v2-m3) cut them to 3.84 while
keeping 99.25% of true pairs on held-out train entities. A stacked LightGBM scores each
candidate in the context of the business's other candidates and the record's other suitors,
and an exclusive decision rule, tuned for the exact macro F0.5 at test's decoy density,
picks the matches. For France, absent from train, we self-train the cross-encoders on French
test pairs our pipeline decided confidently (each round gated on labelled US/India pairs)
and add four French decision rules measured on train patterns; the submitted file scores
**0.990376** on the public leaderboard and 0.9912 held-out macro F0.5 on US/India.

---

## 2. Methodology

### 2.1 Problem Analysis

Findings from the data, all measured on the provided files:

- **Scale.**
  - Train: 2,206,821 Source 1 and 10,320,219 Source 2/3 records.
  - Test: 1,732,544 and 9,969,589.
- **Structure.**
  - Source 1 is deduplicated. No Source 2/3 record belongs to two Source 1 entities, so the
    mapping is exclusive.
  - Mean 3.46 true matches per entity (Source 2: 0-5 copies, Source 3: 0-6).
  - 5.6% of entities are singletons.
  - About 26% of Source 2/3 records match nothing. These are hard decoys:
    - the same name at a neighbouring house number;
    - one token changed at the same address;
    - gibberish rebrands with or without "fka <old name>";
    - branch and corporate words added ("Downtown", "Holdings", "Services").
- **Name noise.**
  - Legal forms dropped, abbreviated or moved.
  - Honorifics and junk prefixes; id tags; brackets; spacing.
  - Digit-for-letter typos; word-order swaps.
  - DBA, "t/a" and "fka" aliases; domains and hashtags.
  - Accents injected; about 20% of Indian Source 2 names written in Indic scripts.
- **Address noise.**
  - Components reordered; street-type abbreviations.
  - State as a code, a full name or native script.
  - Placeholders, zero-padded numbers and unit markers.
  - Renamed cities; house-number typos.
  - Empty addresses: 4.4% of true copies, but only 0.3% of decoys.
- **Test differs from train in three ways.**
  1. **France.** About 15% of test Source 1 is French.
     - French Source 2/3 names carry accent noise never seen in train ("Çlub", "Frânce").
     - French decoys use a French corporate vocabulary ("Développement", "Participations",
       "Holding", "& Associés") that never occurs in train.
     - A third of French Source 2/3 addresses name the département where Source 1 names
       the région.
  2. **About twice as many decoys per entity** (5.5-5.8 pool records per Source 1 entity
     against 4.68 in train). In every country they are the train kinds of decoy, not
     orphaned copies.
  3. **Otherwise the same generator.** US and Indian test data behave like train.
- **What loses score (measured on held-out train).**
  - Address-less copies with namesakes: businesses whose best candidate has no address
    score 0.882, against 0.991 overall.
  - Records that go to a rival suitor: 44% of v3's missed pairs.
  - Single-copy entities: 0.968.

### 2.2 Solution Strategy

**Approach Type:** Blocking + Classifier (hybrid): multi-stage blocking, then transformer
cross-encoders, then a stacked GBDT with an exclusive, metric-tuned decision, plus
unsupervised adaptation to the unseen country.  
**Core Innovation:** a blocking cascade whose last stages are learned models (3.84
candidates per record on test; 99.25% of true pairs kept on held-out train entities), group
context for every score, and self-training on the country absent from train, gated on
labelled US/India pairs.

**High-level approach:**

1. **Candidate generation as a cascade** (section 3). Per-country inverted-index TF-IDF
   search and the bi-encoder's nearest neighbours give about 39 pairs per Source 1 record.
   A LightGBM filter over cheap features keeps pairs with p1 ≥ 0.02 (4.58 per record on
   test), and two cross-encoder cuts leave 3.84 (6,650,585 pairs; the final matches are 3.37
   per record). For France, the same-address copies the French decision rules recover stay
   candidates. These are exactly the pairs the decision is taken on: `scripts/rule_variant.py`
   sets every other pair's stage-2 probability to 0 before exclusivity and the decision rule,
   and `ber.scale.predict_split` evaluates stage 2 only on them. `candidate_pairs.tsv` lists
   them.
2. **Two cross-encoders, cascaded** (section 4). multilingual-e5-small (MIT) scores every
   pair the filter keeps; bge-reranker-v2-m3 (Apache-2.0) scores the ~20% the small one is
   unsure of. Their scores make the last two blocking cuts and are the matching model's main
   features: in stage 2 the reranker's columns carry ~78% of the gain.
3. **Group context everywhere.** Every model score is also given as its rank, gap and margin
   among the entity's candidates and among the pool record's suitors. Exclusivity asks which
   of these businesses owns the record, and these columns answer it.
4. **Matching and decision** (section 4). A stacked LightGBM (stage 2) reads the scores in
   that context. Each Source 2/3 record goes to at most one entity, and a decision rule tuned
   for the exact macro F0.5 at test's decoy density picks the matches.
5. **Self-training for the unseen country.** No French labels exist. Pairs that two
   independent versions of our pipeline both decided confidently become pseudo labels, and
   the cross-encoders are retrained on them mixed with (at least half) original labelled
   pairs; the small one also learns from simulated French copies. A gate on held-out US/India
   pairs keeps an adapted model only if its log-loss stays within tolerance. This raised the
   public score from 0.984 to 0.987405, about 70% of that gain from France. Four French
   decision rules, from patterns whose truth is measured on train, adjust the French
   probabilities (section 4).
6. **Each country from the run that decides it best** (section 4). US and India come from
   run v11, France from run v6a, and two vetoes remove US/India matches shaped like the
   generator's planted lookalikes.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:** hashed TF-IDF tokens of the name (core-name words, consonant
  keys, the squashed name, character 4-grams) and of the address (canonical words and word
  bigrams), searched through per-country inverted indexes; a fine-tuned multilingual-e5-small
  embedding (per-country kNN); then a LightGBM candidate filter and two cross-encoders as
  learned pruning stages. Search and kNN run per country.
- **Candidate pairs generated:** 6,650,585 on test (3.84 per Source 1 record; about 39 per
  record after search, 4.58 after the filter), in `output/candidate_pairs.tsv`.
- **How you ensured true matches were not lost:** three independent generators (name,
  address, embedding) with reverse searches, full re-scoring before any cut, and cuts tuned
  on held-out entities. The filter's 0.02 threshold drops 0.34% of true pairs, pairs the
  matcher would almost never have accepted; the whole cascade keeps 99.25% of true pairs,
  and a perfect matcher on the search stage's union would reach macro F0.5 0.9988.

**Stage (i): multi-view inverted-index TF-IDF search** (`ber.blocking`, `ber.vectorize`).

- **Views.** Hashed tokens, with IDF fitted without labels on both sides of the split.
  - *Name view:* core-name words, consonant-key codes, the squashed name, and character
    4-grams.
  - *Address view:* canonical address words and adjacent-word bigrams.
- **Search.** Exact top-m through sparse matrix products within each country.
  - Each query searches its rarest tokens up to a summed document frequency of 8,000.
  - It returns name top-50, address top-50 and weighted-sum top-100.
  - A reverse search from each Source 2/3 record adds its best Source 1 suitor.
- **Re-score.** Every generated pair is re-scored with the full, unfiltered cosines before
  selection: top-20 combined, top-8 by name and top-8 by address per entity.

**Stage (ii): bi-encoder nearest neighbours** (`ber.embed`).

- multilingual-e5-small (MIT) is fine-tuned contrastively on `"name | address"` texts of
  the *encoder share* (40% of training entities, chosen by a hash of the id).
- The encoder share is never used to train or judge the matcher, so embedding similarity
  is never optimistic on the entities the matcher learns from.
- Batches are hard: sorted by country and area, with near-duplicate names adjacent.
- Both splits are encoded. An exact per-country GPU kNN adds each Source 1 record's 20
  nearest Source 2/3 records, plus each Source 2/3 record's nearest Source 1 record.
- Held-out recall of the embedding view alone: @10 0.9885, @20 0.9936, @50 0.9963.

**Stage (iii): LightGBM candidate filter** (stage 1 of the matcher, `ber.scale`).

- **Features.** String, token, number and embedding features plus group context (section
  4) on the union of (i) and (ii), about 39 pairs per entity.
- **Training.** Candidates of 500,000 sampled entities, 4 folds grouped by entity.
- **Output.** Pairs with p1 ≥ 0.02 go on (`scale.candidate_min_p1`). The threshold is the
  same one that gates the cross-encoders.

**Stage (iv): small cross-encoder pruning** (`scale.candidate_min_ce`, `ber.scale.candidate_mask`).

- The small cross-encoder (multilingual-e5-small, section 4) already scores every pair
  stage (iii) keeps. Pairs it scores below 0.001 are dropped before the large cross-encoder
  and stage 2 run.
- Test has about twice train's lookalike records per entity (section 2.1). This cut drops
  test candidates from 4.58 to 3.89 per entity (France 4.81 to 3.62) and leaves the matching
  output byte-identical. On the holdout, 3.98 to 3.93 per entity at
  the same macro F0.5 (0.99066) and pair recall (99.26%).
- Tighter settings trade score for size (holdout): 0.01 gives 3.89 (−0.00001), 0.03 gives
  3.80 (−0.00006), and a stricter stage-1 cut p1 ≥ 0.05 gives 3.70 (−0.00011). A fixed
  top-k per entity is much worse (top 5: 3.63 candidates, −0.0056), because entities have
  up to 11 true copies.

**Stage (v): large cross-encoder pruning** (`scale.candidate_min_ce2`).

- bge-reranker-v2-m3 (section 4) scores the pairs the small cross-encoder is unsure of.
  Pairs it scores below 0.001 are dropped too. Pairs outside that band keep the small
  one's verdict.
- Test candidates fall from 3.89 to 3.84 per entity; on v6a-fr, 17 of the 5.85M matched pairs
  changed. Holdout: 3.93 to 3.90 per entity at the same macro F0.5.

**Stage (vi): France's rule add-back** (`ber.rescue.rescue_mask` and `replace_mask`, applied in
`ber.scale._candidate_set` and `scripts/rule_variant.py`).

- A French same-address copy that only adds the country's pool-heavy words, or swaps one
  name word for France's generic words (section 4, France's decision rules), stays a
  candidate whenever stage (iii)
  keeps it, whatever the cross-encoders score: the France-adapted cross-encoders were
  taught these copies as non-matches.
- On test this keeps 14,203 French pairs the cross-encoder cuts would drop (918,290 to
  932,493 French candidates, +0.055 per French record). 14,102 of them are matched, through
  the rule's probability floor rather than the model's score. US and India have no add-back.

**Candidate counts:**

| Stage | Pairs per Source 1 entity (train) | True pairs covered |
|---|---|---|
| (i) + (ii) union | 39.5 | 99.59% |
| after (iii) | 4.07 | 99.25% |
| after (iv) and (v), i.e. `candidate_pairs.tsv` | 3.90 (holdout) | 99.25% |
| test, after (iii) → (iv) → (v) | 4.58 → 3.89 → **3.84** | n/a (no test labels) |
| test, submitted version | **3.84** (6,650,585 pairs: 3,205,064 S2 + 3,445,521 S3; US 3.96, India 3.82, France 3.59 incl. 14,203 rule add-backs) | n/a |

---

## 4. Matching Model

**Features used** (stage 1, about 80; `ber.pairfeat`, assembled by
`ber.scale.feature_matrix`):

- *Name features:*
  - Levenshtein, token-set, token-sort and partial ratios, and Jaro-Winkler.
  - Consonant-key and squashed-name ratios.
  - Best similarity through a former or trade name.
  - Legal-form match, conflict or missing; name lengths and flags.
  - Token differences: extra or missing short tokens on either side.
- *Address features:*
  - Ratios and landmark similarity.
  - House-number equality, log-difference, prefix and edit distance.
  - Number-set Jaccard and conflict; postal-code equality.
- *Other:*
  - *Vectors:* TF-IDF cosines of the name, address and combined views, and the embedding
    cosine.
  - *Group context:* for each cosine, rank, gap and margin among the entity's candidates
    and among the record's suitors.
  - *Collective:* how many Source 1 businesses sit at the pool record's address; how often
    the record's name and address recur in the pool; namesakes; alias links.
  - *Name clusters for address-less records:* does the entity's address agree with the
    addresses of the record's same-name siblings.
- No feature encodes a country, so the model applies unchanged to France.

**Model type:** LightGBM candidate filter (stage 1), two fine-tuned transformer
cross-encoders (multilingual-e5-small, bge-reranker-v2-m3) and a stacked LightGBM (stage 2)
over their scores in group context.  
**Threshold selection method:** exact leaderboard macro F0.5 (singletons included) on
out-of-fold predictions, over three rule families; the chosen two-threshold rule is
re-tuned at test's decoy density and reported on held-out entities (see Decision below).

**Cross-encoders (`ber.crossenc`):**

- **Input.** `"name | address"` of both records, up to 96 tokens, Latin accents folded.
- **Training data.** Encoder-share pairs:
  - every true pair;
  - each entity's hardest blocking negatives and embedding-kNN negatives (4 + 3).
- **ce-small.** multilingual-e5-small, first fine-tuned as the bi-encoder, with a mean-pool
  linear head. 4M pairs, lr 3e-5.
- **ce-bge.** bge-reranker-v2-m3 with its own classification head. Micro-batches of 64,
  word embeddings frozen, lr 1.5e-5.
- **France adaptation.** Each model continues from its US/India checkpoint.
  - Mix: 1.0-1.2M original pairs plus 0.6-1.2M French pseudo-labelled test pairs.
  - Round 1: pairs where v3 and v4 agree at p ≥ 0.97 (matches) or p ≤ 0.03 (hard
    non-matches).
  - Round 2: v5's matches that both adapted cross-encoders are sure of and v3 did not
    reject, leaving out pool names that add a France-only pool-heavy word (decoys).
  - A gate (`scripts/ce_gate.py`) keeps an adapted model only if its held-out US/India
    log-loss is within the round's tolerance of the model it replaces (round 1: 5% small,
    10% bge; rounds 2-3: 5% small, 3% bge; round 4: no worse).
  - Gate results:
    - e5-small: log-loss 0.0139 → 0.0133; French pairs left undecided (score between 0.1
      and 0.9) 16.2% → 3.7%.
    - bge: log-loss 0.0123 (e5-base reference 0.0119); French pairs undecided 2.3%
      (e5-base 15.2%).
- **Scoring cascade.** ce-small scores every candidate. bge scores only pairs whose ce-small
  score lies in (0.001, 0.999), about 19% of them. Elsewhere its column repeats ce-small's
  confident score. The same rule applies in train and test.

**Stage 2 (stacking):**

- **Inputs.** A LightGBM over:
  - p1;
  - both cross-encoder scores;
  - each one's rank, gap and margin among the entity's candidates and among the record's
    suitors;
  - p1's remaining mass on both sides;
  - embedding coherence with the entity's best other candidate;
  - kept stage-1 columns.
- **Training.** Same entity folds as stage 1; three seed-bagged models per fold in the
  final run.
- **Feature importance (v5).** bge score 45%, bge pool-side margin 28%, small
  cross-encoder 12%, p1 7%.

**Decision:**

- **Exclusive assignment.** A Source 2/3 record is kept only by its highest-probability
  Source 1 suitor.
- **Rule choice.** Three rule families are tuned by grid search for the exact leaderboard
  macro F0.5 on the sample's out-of-fold predictions (singletons included; blocking misses
  count against recall):
  - one threshold;
  - two thresholds (the entity's best candidate against the rest);
  - a per-entity expected-F rule.

  The best on the sample is used and reported on the held-out entities. v5 chose two
  thresholds, 0.60 (top candidate) and 0.78 (others).
- **Tuned at test's decoy density.** Test has about twice train's planted lookalikes per
  entity (section 2). The submitted rule is re-tuned with each train lookalike counted
  twice: 0.54 (top candidate) and 0.82 (others), against 0.54 and 0.74 at train density.
  - On held-out entities re-weighted the same way: +0.00012.
  - On the public leaderboard: +0.000066 (0.988329 → 0.988395).
- **France's decision rules** (`ber.rescue`), from patterns whose truth is measured on train:
  - a log-odds shift of −1.0;
  - a probability floor for same-address copies that only add the country's pool-heavy
    words, or swap one name word for France's generic words ("& Associés", "Groupe");
  - a shift of −3.0 for address-less copies of names several businesses share;
  - rejection of same-address records that swap one name word for another common French
    name word ("Forme Ecole SA" → "Forme Amicale SA"). This is the generator's same-address
    decoy: 10% (US) and 1% (India) true on train, yet the self-trained French scorers
    accepted a third of them. Removing them gained +0.0009 on the public leaderboard.
- **The submitted combination** (`scripts/combine_by_country.py`):
  - US and India are decided by v11. It is v9 (whose small cross-encoder learned French
    from a simulator of the generator, and whose stage 2 adds source fingerprints) with both
    cross-encoders trained one more round: on more labelled US/India pairs, plus French
    labels taken from the rule-corrected French decisions (replace copies positive,
    same-address word swaps negative). Each was kept only after its held-out US/India
    log-loss improved (small 0.01313 → 0.01176, large 0.01137 → 0.01093). On 330k
    held-out US/India businesses: v11 0.99118, v9 0.99108, v6a 0.99082.
  - France is decided by v6a, whose French decisions the public leaderboard confirmed rule
    by rule.
  - A lookalike veto: a v11 match shaped like the generator's planted lookalikes (house
    number moved 1-9, Source 1 name kept) stands only if v6a made it too. Test carries
    about 17 times train's rate of such disagreements, i.e. its lookalike surplus.
  - A legal-form lookalike veto on US matches: a legal form added or changed together with a
    changed house number. Test's US pool has 3-4 times train's rate of candidates of this
    shape. Such a match is dropped when the number moved 1-9 (unless its probability is
    0.995 or more), or moved further and v6a rejected it. On v9's decisions that removed
    2,386 pairs and gained +0.00009 on the public leaderboard.
- **Holdout.** Held-out entities (about 825k) are never in any model's training sample, and
  the encoder share never enters the matcher. Candidate-set recall losses count in the
  score.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** 0.9912 on held-out training entities for the submitted version's
  US/India decisions (v11); **0.990376** on the public leaderboard (`output/matching_results.tsv`).

**Scores by version:**

| Version | What changed | Held-out F0.5 (train) | Public F0.5 |
|---|---|---|---|
| v1 | token blocking + LightGBM + stacking | 0.9645 | 0.947 |
| v3 | + bi-encoder kNN, detail and collective features, e5-small cross-encoder in stage 2, two-threshold rule | 0.9906 | 0.984 |
| v4 | + orphan simulation | 0.9900* | 0.984 |
| v5 | + France self-training, bge reranker, pool-side cross-encoder context, name clusters, k_emb 20 | 0.9907* | **0.987405** |
| v6a-fr | + round-2 France self-training, decoy-word filter, bagged stage 2, candidate cascade, France log-odds, rescue and namesake rules | 0.9908 | 0.987583 |
| v6a-rep | + France replace-copy floor | 0.9908 | 0.988329 |
| v6a-rep-d | + decision rule tuned at test's decoy density | 0.9908 | 0.988395 |
| v6a-rdb | + France same-address word-swap decoys rejected | 0.9908 | 0.989316 |
| hybrid-L | US/India from v9 (synthetic-French small cross-encoder, fingerprints, stage 2 on 60% more entities) with the lookalike veto; France from v6a with the swap-decoy rule at any position | 0.9911 (US/India) | 0.990171 |
| hybrid-L1 | + US legal-form lookalike veto (legal form added or changed, house number changed) | 0.9910 (US/India) | 0.99026 |
| **final (v11-hL1)** | US/India from v11 (both cross-encoders trained a fourth round on more US/India pairs and rule-corrected French labels) | 0.9912 (US/India) | **0.990376** |

\* Measured on the harder orphan-simulated holdout.

**v5 detail (held-out and out-of-fold, 1.33M eligible entities):**

| | Macro F0.5 |
|---|---|
| **By country** | |
| India | 0.9916 |
| US | 0.9900 |
| **By number of true matches** | |
| 0 (singletons) | 0.9938 |
| 1 | 0.9683 |
| 2 | 0.9890 |
| 3+ | 0.9925 |
| **Pair level** | |
| Pair precision | 0.9984 |
| Pair recall | 0.9745 |

**Holdout to public.**
- Test errors run about 1.7× holdout errors on US/India, largely because test has twice the
  decoys: emulating that density on holdout costs 0.0006.
- The public jump from v3 to v5 (+0.0034) is about three times what the US/India holdout
  gain predicts. The rest is France.

- **Common false positives (wrong merges):**
  - a same-name record at a neighbouring or truncated house number
    ("201 kettle moraine dr" vs "21 kettle moraine dr");
  - a gibberish rebrand at the entity's exact address;
  - an address-less copy of a namesake.
- **Common false negatives (missed matches):**
  - address-less copies whose name dropped or reordered a word
    ("dick regional armada" → "dick regional");
  - transliterated or rebranded names with a partly matching address;
  - house-number typos ("4170 greer rd" → "417 greer rd").

---

## 6. Conclusion

Two choices carried most of the score: every model sees the group context of its decision
(rank, gap and margin among competitors on both sides), and we treated France as
unsupervised domain adaptation, whose gated self-training gave the largest single public
gain (0.984 to 0.987405). The learned cascade keeps the matching set at 3.84 candidates per
entity on test, close to the 3.37 matches per entity the decision keeps, at 99.25% pair
recall on the holdout. The main lesson was the train-to-test shift: test carries about
twice train's decoys, so rules tuned at train density over-merged, and re-tuning at test's
density and vetoing lookalike-shaped matches both gained on the public leaderboard.

---

## Appendix

### A. Code Artefacts

This repository (the [README](../README.md) gives the full reproduction):

- `scripts/reproduce.sh`: the whole chain from raw data to both output files (the only
  entry point needed; `setup_envs.sh` builds the two environments first).
- `src/main.py` (or `python -m ber.cli`): `run` trains and predicts one pipeline version;
  `fetch-model` downloads a pretrained encoder.
- `src/ber/scale.py`: blocking orchestration and candidate filter, features, stages 1-2,
  cross-encoder cascade, tuning, prediction, outputs.
- `src/ber/blocking/`, `src/ber/vectorize.py`: inverted-index TF-IDF search.
- `src/ber/embed.py`: bi-encoder fine-tuning, encoding, GPU kNN.
- `src/ber/crossenc.py`: cross-encoder training and scoring.
- `src/ber/pairfeat.py`: pair, detail, collective and name-cluster features.
- `src/ber/tuning.py`, `src/ber/decision.py`, `src/ber/metrics.py`: decision rules, the
  exact metric.
- `src/ber/cli.py`, `config.py`, `io.py`, `schema.py`, `table.py`, `normalize/`, `vocab.py`,
  `modeltext.py`, `matcher.py`, `pretrained.py`, `submission.py`, `utils.py`: command line,
  configuration, data contract and loading, normalisation, entity folds, the licence-checked
  encoder registry, output writer and validator.
- `scripts/rule_variant.py`: each run's decision (candidate cascade, rule tuned at test's
  decoy density, France's decision rules); `src/ber/rescue.py`: France's decision rules.
- `scripts/combine_by_country.py`: the submitted combination (US/India from v11, France
  from v6a) and the two lookalike vetoes.
- `scripts/make_pseudo_labels.py`, `make_pseudo_labels_r2.py`, `make_pseudo_labels_r4.py`,
  `mix_pseudo_labels.py`: French pseudo labels; `synth_france.py`, `copy_profiles.py`: the
  French copy simulator.
- `scripts/ce_gate.py`, `make_ce_eval_pairs.py`, `make_france_sample.py`: the held-out gate.
- `src/ber/listwise.py`, `scripts/make_lw_groups.py`: listwise rerankers (reproduce.sh
  steps 11-12; used in run v7, whose decisions teach the round-3 pseudo labels).
- `scripts/setup_envs.sh`: both Python environments.

### B. Additional Results

**Measured, and not shipped:**

| Idea | Holdout change |
|---|---|
| Averaging v3 and v4 probabilities | +0.00004 |
| Per-country or per-group decision rules | -0.00002 |
| Forcing each address-less record onto its likeliest suitor | -0.00047 |
| French address rules | the forms they fix are under-represented among uncertain French pairs (0.5-0.9× their rate among confident pairs) |

**Fair play.**
- No external data, lookup or API is used. The only network use is downloading the three
  pretrained encoders once (and pip packages).
- Pseudo labels come from the test inputs, our own predictions and simulated French copies
  of test's French Source 1 records only. No test labels exist or were used.
- We used the public leaderboard as a development signal: our leaderboard submissions
  chose France's log-odds shift (-1.0), accepted or rejected each
  French decision rule and the two US/India vetoes, and picked which run decides each
  country. Round-4 French pseudo labels come from those leaderboard-checked decisions.
- All models are MIT or Apache-2.0 and at most 568M parameters.
