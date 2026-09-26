# Data audit: what in the data can cost score (26 Sep 2026)

Every file in `dataset/` was checked on CPU with `scripts/audit/a1-a8` (polars; no model was
run). Reproduce with `pip install polars`, then
`BER_DATA_DIR=<path>/dataset PYTHONPATH=src python scripts/audit/a6_decoy_share.py` (etc.).
Four findings can cost score; everything else checked out clean (table at the end).

| # | Finding | Who it hits | Cost to fix |
| --- | --- | --- | --- |
| 1 | Test has **~2x the decoys per business**, of the ordinary train kinds. They are **not** orphaned copies. | all countries | CPU minutes (re-tune under emulated density) |
| 2 | French pool names carry an **accent noise alphabet never seen in train** (ç à è ô â ë ê ï î …); the encoders read raw text | France | re-train encoders on folded text (run_all does it from scratch) |
| 3 | French decoys use a **French corporate vocabulary** (Développement, Participations, Holding, Distribution) that never occurs in train | France | check on existing predictions (minutes); one feature or pseudo-label filter |
| 4 | **French address normaliser gaps**: dropped articles, glued elisions, `N°` → `ndeg`, APP/cours/passage | France | small, France-gated like the région fix |

## What the branch does about it (`nikhil_exp`, on by default in `scripts/run_all.py`)

| # | Fix | Where | Check it on an existing run (CPU, minutes) |
| --- | --- | --- | --- |
| 1 | Decision rules and the stage-2 search are tuned **at test decoy density**: every train decoy is cloned once (`decision.decoy_boost=2.0`), and the orphan simulation is off (`--drop 0`). `metrics.json` reports both densities under `decision.decoy_boost`. | `ber.tuning.emulate_decoys` / `tune_at_density`, `scripts/stage2_search.py --decoy-boost` | `python scripts/simulate_decoys.py --run runs/<v5> --share 0 --clone 2.0` prints the v5 rule's holdout at test density and a re-tuned rule's |
| 2 | The bi-encoder, cross-encoders and listwise models read **accent-folded text** (Latin letters only; other scripts untouched). Each model records `text_fold`, and scoring follows the model, so older models keep reading raw text. `BER_TEXT_FOLD=0` turns it off. | `ber.modeltext`, `ber.normalize.text.fold_latin` | needs retraining (run_all does it from scratch) |
| 3 | French pseudo-positives whose pool name **adds a France-only pool word** are dropped. The report counts such merges in each submission. Measured on test, the France-only words are associés, développement, distribution, holding, international, participations. | `ber.vocab`, `scripts/make_pseudo_labels.py --drop-decoy-words`, `scripts/check_decoy_words.py` | `python scripts/check_decoy_words.py --matching output/matching_results.tsv --data-dir <dataset>` (about a minute) |
| 4 | **French address rules**: articles dropped, elisions dropped, `N°`→`no`, APP→apt, cours→crs, passage→pass, Ste→Saint. Only for France and labels never seen; US/India output is unchanged. The table caches are keyed on `NORMALIZE_VERSION`. | `ber.normalize.address` | — |

**Why no veto on decoy words.** Words pool-heavy in every country (com, www, dba,
formerly, incorporated) are copy noise: 9-13% of train's true copies add one. So only
country-specific words count, and they are used where a mistake is cheap (pseudo-labels,
a diagnostic), not as a hard rule. If the check finds thousands of French merges adding
one, a France veto is the next step.

## 1. The extra test pool records are decoys, not orphans

Test has 5.53-5.82 pool records per S1 against 4.68 in train. Two explanations were on the
table: orphaned copies (S1 businesses removed, their copies kept; what `drop_s1_share=0.186`
simulates), or simply more decoys. They leave different fingerprints, because in train true
copies and decoys differ sharply on several surface markers:

| marker (pool names/addresses) | train true copies | train decoys | test (all) | implied share of test pool that is true copies |
| --- | --- | --- | --- | --- |
| empty address, US S2 / S3 | 4.90% / 4.58% | 0.32% / 0.32% | 2.94% / 2.84% | 0.572 / 0.592 |
| branch word (Downtown, Eastgate, Greater …), US S2 / S3 | 1.13% / 1.14% | 18.57% / 18.55% | 8.83% / 8.54% | 0.558 / 0.575 |
| corporate word (Holdings, Services, Group …), US S2 / S3 | 8.98% / 9.16% | 29.60% / 29.69% | 17.57% / 17.43% | 0.583 / 0.597 |
| fka/dba alias, US S3 | 3.00% | 0.00% | 1.82% | 0.607 |
| accent-noised name, US S2 / S3 | 7.46% / 7.45% | 4.62% / 4.88% | 6.25% / 6.39% | 0.574 / 0.588 |
| empty address, India S2 / S3 | 3.83% / 4.02% | 0.23% / 0.28% | 2.28% / 2.46% | 0.569 / 0.583 |
| corporate word, India S2 / S3 | 9.88% / 11.54% | 24.29% / 26.50% | 15.99% / 17.64% | 0.576 / 0.592 |
| exact name also in S1 (a4), US / India | 30.4% / 17.4% | 5.8% / 4.2% | 19.3% / 12.0% | 0.551 / 0.590 |

- **Prediction if the extras are ordinary decoys:** 3.46 copies per business out of 5.76
  (US) or 5.82 (India) gives a copy share of 0.601 / 0.595. The markers agree, all within
  0.55-0.61.
- **Prediction if the extras are orphans:** orphans are copies, so the markers would stay
  near the train copy profile. For example, the US S2 empty-address rate would be ~3.9%,
  but 2.9% is observed.
- **Same-name siblings agree too.** Among test pool records whose name is not in S1, the
  share with a same-name sibling is 17.3% (US) and 26.9% (India). The ordinary-decoy mix
  predicts 17.8% and 26.3%. Orphan clusters would push both up.

**What it means**

- Test has about **2.4 decoys per business against 1.22 in train** (2x), and slightly fewer
  copies per business (~3.3 against 3.46).
- US/India test is otherwise the same generator as train: no token is over-represented
  more than 2x at 80+ occurrences, and the share of S1 names that recur is the
  same (US 35.85% vs 35.81%, India 44.41% vs 44.40%).
- **The orphan simulation models the wrong shift.** That fits v4 (orphan simulation)
  scoring exactly v3 on public (0.984). `run_all.py` passes `--drop 0.186` by default; it is
  harmless but not the fix.
- With twice as many decoy suitors per business, false merges roughly double at a fixed
  threshold, so a rule tuned at train density is too loose on test.

**Fix, cheapest first**

1. **Emulate the density on the holdout (CPU, minutes, on saved stage-2 inputs or OOF
   scores).**
   - Clone every decoy pool record (a candidate column with no true owner) once, copying
     its candidate pairs and probabilities.
   - Rerun `exclusive_adjust` and `tune` on the enlarged arrays.
   - The emulated holdout score predicts public US/India. Re-tuning on it gives the
     test-density rule. It fits naturally as a `--decoy-boost 2.0` option in
     `scripts/stage2_search.py` and the tuner.
2. **Train stage 2 at test density.** Weight pairs whose pool record is a decoy by 2, so
   calibration matches the test prior.
3. **Heavier: synthesise extra train decoys** with the train recipes (namesake at a
   neighbouring number, one token apart at the same address, branch or corporate word
   added), then retrain.

## 2. France: an accent-noise alphabet the models never saw

| names containing … | train pool (US+India, 10.3M) | test France S1 | test France S2 / S3 |
| --- | --- | --- | --- |
| ç à è ô â ë ê ï î û ü (and capitals) | **0** | 4.0% (real words: Frères, Lycée) | **11.6% / 11.2%** |
| á í ó ú (and capitals) | 3.6-5.1% | 0 | **0** |

- **What the test generator does.** It noises French pool names with French accents:
  "Çlub", "àmicale", "Spôrtive", "Frânce", "Fïls", "Prîmaire", "Çomite". Train noise is
  acute-only (á é í ó ú) and never touches these letters.
- **The feature side is safe.** The normaliser folds accents with anyascii, so the
  LightGBM features see clean text.
- **The encoders are exposed.** The bi-encoder, cross-encoders and listwise model read raw
  `"name | address"` (`embed.record_texts`, `crossenc._text`, `listwise.texts`).
  multilingual-e5's SentencePiece keeps accents, so "Çlub" and "Club" become different
  tokens. The embedding carries ~79% of stage-1 gain, and the models only learned
  invariance to acute noise. About one French pool name in 13 carries noise of a kind they
  never trained on.
- **Fix: fold Latin diacritics in those three text functions**, identically for train and
  test.
  - How: NFD, then drop combining marks whose base letter is ASCII. Indic vowel signs and
    viramas sit on non-ASCII bases, so they are kept.
  - Then re-train the encoders. `run_all.py` trains them from scratch anyway.
  - Alternative: add French-style accent noise as training augmentation.
  - v5's France-adapted cross-encoder sees this noise through pseudo-labels, which covers
    it only partly.

## 3. France: decoys speak a corporate vocabulary train never had

- **The rates.** Names containing développement / participations / holding / distribution
  are **0.35% of French S1 but 10.7% / 10.2% of S2 / S3**, roughly a quarter of French
  decoys.
- **The train analogue.** In US/India the English words (Holdings, Services, Group …) are a
  decoy marker: 29.6% of decoys against 9% of copies.
- **Examples.** Each is S1 → pool record:

| S1 | pool record | what changed |
| --- | --- | --- |
| YMM Sport SARL, 24 Rue Mazarin, Bordeaux | YMM Sport Participations, 25 Rue Mamarin, Bordeaux | neighbouring number |
| PNZ Club SASU, 33 Rue Maryse Bastie | PNZ Club Développement SASU, N° 42 Rue Maryse Bastie | different number |
| Dionysienne Ecole, 13 Avenue Victor Hugo, Mérignac | Dionysienne Développement, 13 Avenue Victor Hugo | same address, one name token changed |
| Lan Societe, 1 Avenue de Broceliande, Nantes | Lan Societe Distribution, 14 Avenue De Broceliande | different number |

**Why it matters**

- None of these words occurs in train, so no model has learned them as decoy signals, and
  false merges are the expensive error under F0.5.
- Worse, v5's French pseudo-labels come from v3's confident French decisions. If v3 merged
  these, the France cross-encoder is taught to merge them.

**Check first (minutes, on existing outputs).**

- In v3/v4/v5 `matching_results.tsv`, count the French matched pairs whose pool name has
  one of these words and whose S1 name does not.
- Do the same count in the positive pseudo-labels.
- Thousands means a real false-merge source.

**Fixes**

1. Drop such pairs from the positive pseudo-labels.
2. Add a language-agnostic stage-2 feature: the largest pool/S1 frequency ratio among the
   pool name's added tokens. Compute it per country on each split's own inputs, which is
   unsupervised. In US it recovers Holdings/Services; in France, Développement /
   Participations.
3. If the counts are large, a France veto on those pairs.

## 4. France: address-normaliser gaps

All four only affect normalised-token features (not the encoders). Each can be France-gated
like the région fix, so US/India caches stay valid.

| gap | evidence | today | fix |
| --- | --- | --- | --- |
| article after the street type dropped by the pool ("3 R PAIX" vs "3 Rue de la Paix") | street type + de/du/des: S1 48.5%, pool 30.4-31.2% (~37% of articles dropped) | `rue de la paix` vs `rue paix` | drop de, du, des, la, le, les, l, d, au, aux in French addresses |
| elision glued ("de l'Intendance") | ~5% of French addresses | `de lintendance` vs `crs intendance` | strip `[ldjmnst]'` / `’` before apostrophe removal |
| `N°` becomes a junk token | 4.9% of French pool addresses (70k), 0 in S1 and train | `ndeg 39 rue violette` | map `N°` to `no` (already dropped) |
| abbreviations missing | pool-only tokens: APP (125), CRS (888), PASS (183) | `app`, `cours`≠`crs`, `passage`≠`pass` | `app`/`appt`→`apt`, `cours`→`crs`, `passage`→`pass` (France only) |

Negligible: "Ste" for Sainte (0.03%), CEDEX/BP (≤0.1%).

## Checked and clean

| check | result |
| --- | --- |
| file integrity (a1) | no BOM, no CRLF, no NUL, no bad UTF-8, all NFC, exactly 4 fields on every line of every file |
| IDs (a2) | unique within each split; prefix matches file; **0 IDs shared by train and test**; ID numbers uncorrelated with labels (corr 0.0001, sibling gaps random) |
| ground truth (a2) | one row per S1; no unknown or duplicate IDs; no spaces/quotes/empty tokens; **no pool record with two owners** (exclusive holds); **0 cross-country pairs** |
| label noise (a3) | true pairs sharing no name token and no address token: 0.015% (1,113), nearly all transliteration or heavy typos, so real matches |
| exact-copy decoys (a3) | 99,983 S1 = pool (exact name + address) pairs, **all true**; no exact-copy decoys |
| S1 duplicates (a3, a5) | 0 exact duplicates in train S1, 1 in test S1 |
| train/test leakage (a5) | 0 identical records; 5.5% (US) / 4.9% (India) of test S1 addresses reuse a train S1 address (the generator reuses addresses); France shares nothing with train |
| country labels (a5) | no record labelled one country with another country's address (the "French" hits in US/India are real: "461 Rue De Paris, Vineyard, UT") |
| field swaps, truncation (a5) | swaps ≤0.4% (and legitimate: "776 Broadway Drive Apartments"); no length spikes |
| US/India vocabulary shift (a4) | none (all token ratios ≤2x at ≥80 occurrences) |
| minor artifacts, equal in train and test (a1) | mojibake `Â\x80\x93` (~0.02%), `\x1a` (~0.003%), a few HTML entities, ZWNJ inside Indic names, double spaces (~12% of pool rows); all folded by the normaliser |
| test-only artifact (a1) | ~800 French names wrapped as `"""ehpad Club SAS"`; harmless after cleaning |
| pipeline I/O | reads TSVs with `QUOTE_NONE` and no NA inference, writes LF, one row per S1; metric matches the README (singletons 1/0, empty prediction for a non-singleton = 0) |

**Irreducible:** 4,389 groups (9,752 records) of identical pool records belong to different
S1 businesses in train. All but one are address-less, so identical text with different
owners means at least one per group is always wrong or missed (~0.1% of the pool).
