# Team handbook: PerryDaPlatypus, Amazon ML Challenge 2026

Read this first, whether you are a teammate or an agent working for one. It says
what the task is, what has been built, what it scores, how to run it, what was
learned, and which open directions are free for a branch to take.

Last updated 2026-09-26 02:00 IST.

---

## TODO

### Nikhil: listwise reranker (v6 candidate), due 26 Sep 14:00 IST

The listwise group reranker (idea C1: one S1 record read together with all its
candidates, pair head plus group head) is the strongest idea we have not measured yet.
It exists only on your local branch `listwise-reranker`. v6 is built on the 26th from
whatever beats v5 on holdout, so it has to be trained and measured by early afternoon.

1. **Rebase onto current `main` and push the branch.** It branched at `a70d538`; since
   then `main` changed `src/ber/scale.py`:
   - pool-side cross-encoder context (`scale.ce_pool_context`): the same idea as your
     E1, so keep one implementation, main's;
   - `scale.ce_band`: cross-encoders after the first score only the pairs the first is
     unsure of;
   - `scale.ce_train_all`;
   - orphan simulation (`scale.drop_s1_share`);
   - name-cluster features (`FEATURE_VERSION` 4) and `scale.feature_donor_run`.

   Resolve conflicts in favour of main's behaviour and keep your scorer dispatch
   (`"kind": "listwise"`). Then run `pytest -q` and `ruff check src tests scripts`; both
   must be clean.
2. **Train C1 on your GPU tonight** (`scripts/gpu_runbook.sh lw`). Train only on the
   encoder share or the listwise share, never on holdout entities.
3. **Measure it against v5**, on the same holdout definition, with
   `scale.drop_s1_share=0.186`: v4 scored 0.99001 there, and v5 will be added to
   section 2 in the morning. Either add the listwise model to `scale.ce_model` after
   v5's two cross-encoders, or run your stage-2 search (F1) over the saved stage-2 inputs.
4. **Deliver:** the pushed branch, the exact training command, and the holdout macro
   F0.5 with and without the listwise columns. The lead folds it into v6 if it wins.

**Stretch: C7, pool-side listwise** (one pool record against its S1 suitors: which one
owns it, or none). It targets our weakest group: businesses whose best candidate has no
address score 0.882 on holdout, against 0.991 overall.

Rules as always: models MIT or Apache-2.0 and at most 8B parameters, no external data,
report holdout macro F0.5, and do not upload to the portal without telling the lead. If
you need v5's `train_scores.npz`, the blocking caches or the embeddings, ask the lead.

**Status (26 Sep, ~02:00 IST, Nikhil):** step 1 done. Branch **`nikhil_exp`** is rebased onto
`cb483ba` (main's behaviour kept wherever the two overlapped) and pushed; 212 tests and
ruff are clean. It also holds C7 (the stretch), French région/département normalisation,
and training groups built from the raw data on the encoder share, so the holdout stays
v5's. Steps 2-4 (training and measuring) moved to Nambi below: Nikhil's laptop cannot run
them. Plan and evidence: [docs/nikhil_exp_plan.md](docs/nikhil_exp_plan.md).

### Nambi Rajan: train and measure C1 + C7 on `nikhil_exp`, due 26 Sep 14:00 IST

Full steps and commands: [docs/nikhil_exp_plan.md](docs/nikhil_exp_plan.md), section 3.

1. `git switch nikhil_exp`; get `models/ce-small-v2` from the lead as the init.
2. Groups from the raw data (CPU): `bash scripts/gpu_runbook.sh groups` and `pgroups`.
3. Train and gate C1: `LW_INIT=models/ce-small-v2 bash scripts/gpu_runbook.sh lw`, then `lweval`.
4. Train and gate C7: `LW_INIT=models/ce-small-v2 bash scripts/gpu_runbook.sh lwp`, then `lwpeval`.
5. With the lead (v5's caches): v5 + `scale.ce_model=<v5 CEs>,models/lw-small,models/lwp-small`
   + `scale.stage1_model=<v5 run>` + `scale.save_stage2=true` + `scale.drop_s1_share=0.186`;
   `scripts/stage2_search.py` with and without the `lw*`/`lwp*` families.
6. Report the holdout macro F0.5 with and without each model in section 2. The French
   normalisation is a public-LB A/B only (delete `artifacts/cache/test_*` first); upload
   only after telling the lead.

---

## 1. The challenge in five lines

- **Task.** Source 1 (S1) is a deduplicated list of businesses. For every S1 record,
  find all records in Source 2 and Source 3 (S2/S3, "the pool") that describe the same
  business. An S1 record has 0 to 11 matches.
- **Metric.** Macro F0.5 per S1 entity, averaged over all S1 entities, singletons
  included: an empty prediction for a true singleton scores 1.0, and any prediction for
  it scores 0. F0.5 weights precision twice as much as recall.
- **Test** has France, which is **absent from train** (~15% of test S1). Train is US +
  India.
- **Rules that disqualify.** No external data or lookups (APIs, registries, geocoding,
  web). Any model must be **MIT or Apache-2.0 and at most 8B parameters**.
  `tests/test_fair_play.py` guards imports.
- **Deadline.** About **2026-09-28 00:00 IST** (portal countdown). The final ranking
  uses the private leaderboard (the rest of test).

Full brief: [docs/problem_statement.md](docs/problem_statement.md).

## 2. Scores so far

| Version | What it is | Holdout macro F0.5 (train) | Public LB |
|---|---|---|---|
| **v1 baseline** (tag `v1-baseline`, commit 4ddca44) | token blocking + LightGBM + stacked LightGBM | **0.9645** | **0.947** (25 Sep, 16:48) |
| v2 (commit 241b5af) | v1 + fine-tuned embedding blocking + embedding cosine + detail features, 500k training entities | **stage 1: 0.9803** (OOF 0.9800); stage 2 crashed on a cross-encoder dtype bug, fixed in 71533e9; no test output | — |
| **v3** (run 20260925-161528-v3, code a63476f) | v2 + collective features + stage 2 with cross-encoder ce-small-v2 + coherence + two-threshold decision | **0.9906** (stage 1 0.9817) | **0.984** (25 Sep, 20:01) |
| v4 (run 20260925-183824-v4) | v3 + orphan simulation (18.6% of train S1 removed) | 0.99001 (on the harder orphan-simulated holdout) | **0.984** (25 Sep, 21:48): no change over v3 |
| v5 (running since 26 Sep 00:27, `runs/v5_overnight.sh 0.186`) | v4 + France-adapted e5-small cross-encoder + France-adapted **bge-reranker-v2-m3** (568M, Apache-2.0) scored where the small one is unsure (`ce_band`) + pool-side cross-encoder context + name-cluster features + k_emb 20 + 700k training entities; each adapted cross-encoder is gated on held-out US/India log-loss | ETA ~07:00-08:00 | — |

Leaderboard at 00:00 on 2026-09-26: 1st 0.988419, 2nd 0.987818, 3rd 0.987288, 4th 0.987. Our best is 0.984 (v3, v4).

**Holdout → public.** Test errors run at about 1.5-1.7x holdout errors (v1: 3.55% → 5.3%; v3: 0.94% → 1.6%). The gap is proportional, so holdout gains transfer. Public 0.987 needs a holdout of ~0.9923; 0.990 needs ~0.9941.

**Public score is 1.75 points below the train holdout (0.947 vs 0.9645). Leading
explanation: test has more decoys.** Test has 5.53-5.82 pool records per S1 business
(France 5.53, India 5.82, US 5.76) against 4.68 in train, in every country. With ~3.46
true copies per business, about 40% of the test pool matches nothing, against 26% in
train. One likely generator detail: businesses dropped from test Source 1 whose copies
stay in the pool (orphans, the hardest decoys). France is not singled out: its predicted
singleton rate (5.4%) and matches per business (3.46) match train. **Anyone tuning a
threshold or decision rule on train must account for this.** The lead is testing it by
dropping ~19% of train businesses from the candidate set (keeping their copies) to
reproduce the test ratio.

**How the holdout is measured.** Training entities are split. The matcher trains on a
random sample with 4-fold cross-validation grouped by entity; every other entity is a
holdout it never sees. Scoring uses the exact leaderboard metric
(`src/ber/metrics.py`, cross-checked against an independent literal implementation in
`scripts/score_submission.py`, see `tests/test_independent_scorer.py`). Recall counts
true matches that blocking never found, so the number is not inflated. For v1, the
out-of-fold sample (0.9634), the holdout (0.9645) and all 2.2M entities scored from the
written id lists (0.96437) agree.

v1 details (from `runs/<ts>-full-v1/metrics.json` on the lead's machine):

- **Blocking.** 24.2 candidates per S1; 96.7% of true pairs are in the candidate set;
  the oracle F0.5 (a perfect matcher on these candidates) is 0.9886.
- **Stage 1.** OOF 0.9613, holdout 0.9622. **Stage 2.** OOF 0.9634, holdout 0.9645
  (threshold 0.65).
- **By country.** US 0.970, India 0.955.
- **By true number of matches.** 0 → 0.965, **1 → 0.866**, 2 → 0.955, 3+ → 0.974.
- **Test output.** 6.0% predicted singletons, 3.59 matches per matched entity.

v2 blocking, already measured: pair recall **0.9936**, oracle F0.5 **0.9982**.
Blocking is solved; **the matcher is the whole gap.**

## 3. The data (measured on train)

- 2,206,821 S1; 10,320,219 pool records. Test: 1,732,544 S1 (259k France); 9,969,589
  pool records.
- **Exclusive.** No pool record belongs to two S1 entities, and true pairs always share
  a country.
- **Copies per entity.** S2 has 0-5 and S3 has 0-6, roughly independent. Mean 3.46
  matches. 5.6% of S1 are singletons.
- **26% of pool records match nothing.** They are the traps:
  - the same name at a neighbouring house number ("Howard, 450 Mass Ave" vs "471 Mass
    Ave");
  - a name one short token apart at an identical address ("WPQ Realtors" vs "RPQ
    Realtors");
  - gibberish rebrands ("Lumnylazeta"), with or without "fka <old name>".
- **Noise on true copies.**
  - Names: abbreviations, legal forms moved or dropped, honorifics, id tags
    ("#30473"), digit-for-letter typos ("G1obal"), accents, **Indic scripts (~20% of
    Indian S2 names)**, transliteration ("Skti Imphotek" for "Shakti Infotech"),
    "fka/dba/t/a" aliases, domain and hashtag names.
  - Addresses: reordered components, abbreviations, state as name, code or native
    script, "null", zero-padded numbers, truncation to a city, empty addresses.
- **House numbers.** 15% of true pairs have a differing first house number. With the
  same name and a differing number, P(match) is 0.05 overall and 0.54 for the entity's
  top candidate: the hardest confusion.
- **Measured signals now in v3's collective features (EV-0017):**
  - Same street and number, different names: P(match) is **0.92** when one S1 business
    has that address, 0.31 for two, **0.08** for six or more (building vs rebrand).
  - A rebranded candidate whose brand appears elsewhere with "formerly X", where X is
    ≥80% similar to the S1 name: P(match) **0.71** against 0.11.
  - A differing house number that recurs in other pool records marks a real value
    (0.95 as top candidate) rather than a one-off (0.58).
- **France.** The normaliser already handles French legal forms (SARL, SAS, SASU, EURL,
  SNC, SA) and addresses (R./rue, AV, BD, #/No).

## 3b. What the models taught us (read before designing v4)

1. **The embedding similarity drives the matcher.** v2 stage-1 gain shares:
   `cos_emb_rank_pool` 24.1%, `cos_emb` 22.7%, `cos_emb_margin_pool` 20.1%,
   `cos_emb_gap_pool` 12.1%, so **79% comes from the fine-tuned embedding**, mostly
   **pool-side competition** (is this S1 the pool record's best suitor?). Next come
   house-number conflict/Jaccard/log-difference (7%) and address token set (1.8%). The
   embedding model is the main quality lever.
2. **Blocking is solved.** The union of token blocking and embedding kNN (k=10, reverse
   1) holds 99.36% of true pairs; oracle F0.5 is 0.9982.
3. **The cross-encoder is very strong pairwise.** On 600k held-out top-10 candidate
   pairs, `ce-small-v2` scores AUC 0.99985, log-loss 0.0139, best pair F0.5 0.9925. More
   data and embedding-neighbour negatives cut log-loss 36% against `ce-small`.
4. **Train → test shift (explains 0.9645 → 0.947 public).** Test has **5.5-5.8 pool
   records per S1** against **4.68 in train**, in every country, so ~40% of the test
   pool matches nothing, against 26% in train. The likely cause is orphaned copies
   (businesses left out of test Source 1 whose copies remain in the pool). Because the
   model relies on pool-side competition, an orphan's next-best S1 suitor looks like
   its owner: exactly the signal that dominates (point 1). **Remedy under test:**
   simulate it in train by removing ~19% of S1 businesses from the candidate set
   (keeping their copies), then train and tune the decision under that condition.
5. **France looks normal in the test predictions.** 5.4% predicted singletons and 3.46
   matches per business, against 5.6% and 3.46 in train truth. India is the weaker
   country (US 0.970, India 0.955 on the v1 holdout).
6. **Error types (v1).** Wrong merges are neighbours and namesakes (same name,
   neighbouring house number; one-letter name difference at the same address). Misses
   are heavy transliteration, addresses cut to a city or empty, and renames without an
   alias.
7. **The weakest group is entities with exactly one true match** (F0.5 0.866 in v1).

## 3c. Where v3 still loses (read before v5+ work)

- **France is most of the public gap.** v3 is 4-5x less certain on France (unseen in
  train) than on US/India: 0.77% of French businesses have an uncertain best candidate
  against 0.16-0.18%, which implies ~4.3% French error against ~0.9% elsewhere. v5
  answers this with self-training: `scripts/make_pseudo_labels.py` turns v3's confident
  French test decisions into pseudo-labelled pairs (827k matches, 1.04M hard
  non-matches), and `ber.crossenc train --pseudo` fine-tunes the e5-base cross-encoder on
  them mixed with original pairs. Only the provided test inputs are used; no labels, no
  external data.
- **75% of v3's holdout errors are address-less candidates** (a same-name record with no
  address). Where the name's other pool records all sit at the S1 address, P(match) is
  0.97 (v5's name-cluster features). Where namesakes exist elsewhere, P(match) is 0.13
  and the case is largely irreducible for everyone.
- **44% of v3's missed true pairs lose their pool record to another suitor**, typically
  a namesake. v5 gives stage 2 each cross-encoder's pool-side rank, gap and margin.
- **Blocking still misses 0.64% of true pairs.** A third of them sit at embedding ranks
  11-20; v5 raises k_emb to 20.
- **Measured and dropped:** orphan decoys with fixed probabilities (no harm), EM prior
  adjustment (+0.00001), abstaining on near-tied suitors (+0.00001), coherence features
  (~0% of stage-2 gain).

## 4. Architecture

```
raw TSVs → normalise (parallel, cached tables)
  → blocking: hashed TF-IDF name/address views, per-country inverted-index search
       + fine-tuned multilingual-e5-small kNN (v2)                      → candidate_pairs.tsv
  → stage-1 features: string, number, detail (v2), collective (v3), cosines + group context,
       embedding cosine (v2)
  → stage 1: LightGBM, 4 grouped folds on sampled entities
  → stage 2: LightGBM on stage-1 probability in context + cross-encoder score on uncertain
       pairs (v2) + coherence with the entity's best other candidate (v2)
  → exclusive assignment (each pool record to its best S1 suitor)
  → decision rule tuned on OOF: threshold / two-threshold / expected-F (v2) → matching_results.tsv
```

Key files:

| Path | Role |
|---|---|
| `src/ber/scale.py` | the whole large-scale pipeline (`run`, `block_split`, `feature_matrix`, stages, prediction) |
| `src/ber/pairfeat.py` | vectorised pair features (string, number, detail, collective, group context) |
| `src/ber/embed.py` | encoder fine-tuning, encoding and GPU kNN (**torch environment**) |
| `src/ber/crossenc.py` | cross-encoder training and scoring (**torch environment**) |
| `src/ber/tuning.py` | decision rules and their tuning |
| `src/ber/normalize/`, `table.py`, `vectorize.py`, `blocking/multiview.py` | normalisation and token blocking |
| `src/ber/configs/full.yaml` | the settings of the real runs |
| `scripts/learning_curve.py` | offline learning curve and leave-one-country-out on cached features |
| `scripts/score_submission.py` | independent stdlib scorer |
| `scripts/prepare_split.py` | fill a split's caches ahead of a run |

**Leakage rule for learned similarities (important).** The encoder and cross-encoder are
fine-tuned on a fixed 40% of train entities (`ber.embed.encoder_mask`, crc32 of the S1
id). `ber.scale` never trains, tunes or evaluates the matcher on those entities. If
you add any model trained on train pairs whose output becomes a matcher feature, train
it only on encoder-share entities too, or the holdout lies.

## 5. How to run

### Environments

- **Pipeline:** Python 3.10 venv. `pip install -r requirements.txt` (dev:
  `requirements-dev.txt`), then `pip install -e . --no-deps`.
- **GPU steps (embed, crossenc):** any Python with CUDA PyTorch plus
  `requirements-gpu.txt`. The lead uses Python 3.13 with torch 2.11 cu130, launched as
  `py -3.13`. The pipeline calls the cross-encoder through `scale.ce_python`
  (default `py -3.13`), so set that to your torch interpreter.
- **Data** is not in git. Put the challenge's `dataset/` somewhere and pass
  `--data-dir`. The lead uses `student_resource/dataset`.
- **Models** are not in git. Download the base encoder (MIT), licence-checked:
  `python -m ber.cli fetch-model intfloat/multilingual-e5-small --dest models/multilingual-e5-small`.
  Fine-tuned models are made by the steps below.

### v1 baseline (CPU only)

```bash
python -m ber.cli run --config src/ber/configs/full.yaml --data-dir DATA --out-dir output --tag v1
```

This takes about 3 hours cold on a 24-thread laptop. It normalises, blocks, trains,
predicts test, writes `output/matching_results.tsv` + `candidate_pairs.tsv` and
validates them. Metrics go to `runs/<ts>-v1/metrics.json`.

### v2 / v3 (GPU steps first)

```bash
# 1. fine-tune the bi-encoder on encoder-share entities, encode train+test, kNN lists (~1.5 h)
PYTHONPATH=src py -3.13 -m ber.embed all --data-dir DATA --base models/multilingual-e5-small \
    --model models/e5-small-ft --emb-dir artifacts/emb --epochs 0.5 --batch 512 --k 50 --k-reverse 3
# 2. cross-encoder on encoder-share entities (needs the train blocking cache from a v1 run, ~15 min)
PYTHONPATH=src py -3.13 -m ber.crossenc train --init models/e5-small-ft \
    --blocking artifacts/cache/train_blocking_KEY.joblib --model models/ce-small
# 3. the run
python -m ber.cli run --config src/ber/configs/full.yaml --data-dir DATA --out-dir output-v2 --tag v2 \
    --set scale.emb_dir=artifacts/emb --set scale.ce_model=models/ce-small --set scale.train_sample=500000
```

### Iterating quickly

- **Compare cross-encoders** on the same held-out pairs (entities no cross-encoder
  trained on) before paying for a full run:
  `PYTHONPATH=src py -3.13 -m ber.crossenc eval --model models/a,models/b --split train --pairs artifacts/ce_eval_pairs.npz`.
  Build the pairs file once with `python scripts/make_ce_eval_pairs.py --blocking artifacts/cache/train_blocking_KEY.joblib`
  (the top-10 token candidates of 60k non-encoder entities, with labels).
  Current best: `ce-small-v2` (4M pairs, token and embedding-neighbour negatives),
  AUC 0.99985, log-loss 0.0139; `ce-small` (1.5M pairs) scored 0.99970 / 0.0218.
- **Reuse stage 1** when only stage 2, the cross-encoder or the decision changes:
  `--set scale.stage1_model=runs/<previous>/scale_model.joblib`. This saves ~30 minutes;
  it is refused if the features or folds differ.
- **Offline experiments** on a finished run: `scripts/learning_curve.py --run RUNDIR --features FEATURES.npy`
  (add `--loco` for leave-one-country-out).

### Caches (why reruns are fast)

`artifacts/cache/` holds, per split:

- normalised tables (`*.table`);
- blocking output (`{split}_blocking_KEY.joblib`, keyed by the tables and every
  blocking setting);
- feature matrices (`{split}_features_KEY_vN.npy` + `.json`, memory-mapped).

A new feature version copies every column an earlier version already has and computes
only new feature groups (`FEATURE_GROUPS` in `scale.py`). If you change what an
existing column means, **give it a new name**; do not just bump `FEATURE_VERSION`.

### Resources and pitfalls

- 31 GB RAM machine. **Do not run two heavy jobs at once** (a run plus GPU encoding
  reached ~30 GB and the machine went down). Keep the laptop awake and plugged in.
- About 7 GB of the 16 GB GPU was taken by other apps; batch sizes assume ~9 GB free.
- Do not edit `ber/normalize`, `table.py` or `vectorize.py` while a run is going:
  joblib workers re-import them mid-run.
- On Windows, PowerShell mangles JSON and quotes passed to native programs; use bash
  for those.

### Validate and package a submission

```bash
python student_resource/utils/validate_submission.py --matching OUT/matching_results.tsv --candidate OUT/candidate_pairs.tsv --test-dir DATA/test --check-ids
python -m ber.cli package --team PerryDaPlatypus --repo-root WORKTREE_AT_THAT_COMMIT --out-dir OUT --doc docs/Documentation_template.md --dest submissions --data-dir DATA
```

Package from a `git worktree` at the commit that produced the outputs, so the zip's
code reproduces them. The portal takes **two uploads**: `matching_results.tsv` and
`PerryDaPlatypus_submission.zip`.

## 6. Open directions: take one per branch

Status as of this writing. Put your name next to a direction when you start, and add
your holdout number when done.

| # | Direction | Why it may pay | Status / owner |
|---|---|---|---|
| A | **Cross-encoder improvements** (more pairs, harder negatives, larger backbone) | attacks neighbour and rename confusions | lead: ce-small-v2 done (best so far); e5-base see J |
| B | **Training-set size and seed bagging** | v2 trains on 500k of 1.32M eligible entities | lead: `scripts/learning_curve.py` after v2 |
| C | **France / unseen country**: leave-one-country-out as a proxy (train US, score India); drop features that do not transfer; self-training on confident French pairs | 15% of test, unmeasurable locally | open |
| D | **Count-aware decision**: predict each entity's number of copies, then choose the set maximising expected F0.5 | entities with one true match score 0.866 | open (offline on `train_scores.npz`) |
| E | **Set-level assignment**: near-duplicate pool records (siblings) must go to the same S1 entity or to none | stops partial merges of distractor copies | open (needs a pool-to-pool kNN) |
| F | **Consensus features**: majority house number, street and name across a candidate's siblings, compared with S1 | recovers typo'd copies, rejects neighbours | open |
| G | **Regime experts**: separate stage-2 models for same-name/different-number, different-name/same-address, and the rest | two unrelated confusions share one threshold today | open |
| H | **Better bi-encoder**: second round with hard negatives mined from its own kNN (ANCE), and/or the larger multilingual-e5-base | the embedding drives 79% of the matcher's gain (section 3b) | **high priority**, open |
| I | **Orphan simulation**: drop ~19% of train S1 businesses from the candidate set (keep their copies), then train and tune under that condition | the train→test decoy shift (section 3b, point 4) likely explains most of the 0.9645 → 0.947 gap | lead: testing on v3's saved scores |
| J | **e5-base cross-encoder** (278M, MIT) | stronger pairwise model | lead: training, ETA 18:07 |
| K | **Listwise group reranker** (`ber.listwise`): one transformer pass over an S1 record and all its uncertain candidates; content-only, so orphans cannot fool it. Plus pool-side context for every scorer in stage 2, a score cache, `stage2_exclude`/`stage2_monotone`, `scripts/stage2_search.py` (every model combination on the holdout) and `scripts/shift_diagnostics.py` (orphans vs extra decoys vs France). Plan and runbook: [docs/v4_listwise.md](docs/v4_listwise.md), `scripts/gpu_runbook.sh` | residual errors are group-level; stage 2's best signal (pool-side competition) is what test's orphans break | branch `listwise-reranker`, (owner) |
| — | LLM matcher (7-8B) | rejected: tens of pairs per second against millions of uncertain pairs | rejected |

Every run writes `train_scores.npz` (each pair's final probability, label, entity sets
and country) into its run directory, so decision-rule work (D, E) can be done offline
in minutes without retraining.

## 7. Working together

- Branch from `main`: `git switch -c <you>/<direction>`. Keep data, models, caches,
  runs and outputs out of git (`.gitignore` covers them).
- Before pushing: `ruff check src tests scripts`, `ruff format --check src tests`,
  `pytest -q` (about 175 tests, ~1 minute).
- Report results as the **holdout macro F0.5** from `metrics.json`, the same measure
  as the table in section 2, so versions compare.
- Only upload to the portal what beats the current best holdout, and record every
  public score in section 2.
- The methodology document (`docs/Documentation_template.md`) must describe what
  ships. Update it when a direction lands in the submitted version.
