# All ideas: Amazon ML Challenge 2026, Business Entity Resolution

Team PerryDaPlatypus · 25 Sep 2026, about 19:45 IST · deadline about **28 Sep 00:00 IST**

Every idea found while studying the repo (`main` at `a70d538`), including the expensive
ones, each with a recipe. Pick by the master table (section 2). Run them in parallel
tracks (section 4).

**Status tags**
- **[built]**: implemented and tested on branch `listwise-reranker`.
- **[config]**: a setting away.
- **[lead]**: the lead owns it (handbook direction letter in brackets).
- **[recipe]**: not built; the steps are here.

> **Gains are estimates** from the handbook's numbers, not measurements; nothing here
> has run on the real data yet. Every new model plugs into stage 2 as extra columns, and
> `scripts/stage2_search.py` (F1) measures on the holdout whether it helps. So trying an
> idea is cheap to judge, and a bad idea costs time, not score.

---

## 0. Reality check

- **1.0 is impossible.** The v2 candidate sets cap a perfect matcher at 0.998 on train.
  Some copies cannot be linked by design: renames with no alias, addresses cut down to a
  city.
- **Realistic target: 0.985 to 0.99 public.** The LB top on 25 Sep was 0.9805.
- **The biggest single loss is the train→test gap** (v1: 0.9645 holdout → 0.947 public),
  not the model.

## 1. Findings behind the ideas

| Finding | Number | So what |
|---|---|---|
| Blocking is solved | v2 pair recall 0.9936, oracle F0.5 0.9982 | blocking work gains ≤ 0.001 |
| Train→test drop | 0.9645 → 0.947 (v1) | about 1.75 points lost to the shift |
| Test has more unmatched pool records | 5.5–5.8 pool records per S1 in test vs 4.68 in train, all countries | about 40% of the test pool matches nothing vs 26% in train |
| Likely cause: orphans | about 19% of test businesses' S1 records missing, copies kept | their copies fall to the next-best S1 suitor |
| The matcher leans on the signal orphans break | 79% of v2 stage-1 gain is embedding cosine, mostly pool-side rank/gap/margin | "best suitor" is wrong when the owner is missing |
| Leftover errors are group-level | neighbour house numbers, one token apart at the same address, renames | the evidence is in the other candidates |
| Weakest group | 1-true-match entities: 0.866 (v1) | count-awareness is missing |
| India weaker | 0.955 vs US 0.970 (v1) | Indic scripts, transliteration |
| France unseen | about 15% of test S1 | cannot be measured locally |
| Pairwise cross-encoder already strong | ce-small-v2 AUC 0.99985 | next gains are group- and shift-level |
| Stage 2 never saw cross-encoder scores in pool-side context | only `ce_rank_s1` existed | fixed in E1 |

---

## 2. Master table (every idea)

Cost is wall time on 1× H100/A100 unless it says CPU. Gain is on the public score, estimated.

| # | Idea | Status | Est. gain | Cost | Main risk |
|---|---|---|---|---|---|
| **A. Diagnose** |||||
| A1 | Shift diagnostics (orphans vs decoys vs France) | built | enables B | 10 min CPU | none |
| A2 | LB probe: France emptied | recipe | diagnostic | 1 submission | slot use |
| A3 | LB probes per country / threshold sweep | recipe | diagnostic, +0.001 from a tuned threshold | 1–4 submissions | overfitting public LB |
| A4 | Error audit by error type | recipe | directs work | 1 h | none |
| A5 | Leak audit (ids, row order) | recipe | diagnostic | 30 min | see I1 before using anything |
| **B. Train→test shift** |||||
| B1 | Orphan simulation `drop_s1_share=0.186` | lead (I), built | +0.005 to +0.015 if orphans | CPU run | wrong if A1 says decoys |
| B2 | Mixture of drop shares for robustness | recipe | +0.001 to +0.004 | 2–3 CPU runs | time |
| B3 | EM prior adjustment on test | lead, built | ± | none | can hurt if prior misestimated |
| B4 | Per-country thresholds / priors | recipe | +0.000 to +0.003 | 1 h | overfit without France labels |
| B5 | Inject extra neighbour decoys into train | recipe | +0.002 to +0.008 if A1 says decoys | half a day | realism of injected decoys |
| B6 | Leave-one-country-out feature check | lead (C) | +0.000 to +0.003 on France | 1 h CPU | none |
| B7 | Self-training on French test pairs | recipe | +0.000 to +0.005 on France | 2–4 h GPU | confirmation bias |
| B8 | Self-supervised French adaptation (noised test records) | recipe | +0.000 to +0.004 on France | 2–3 h GPU | unlabelled test use; check rules |
| B9 | Transductive pseudo-labels for stage 2 | recipe | ± | 2 h CPU | confirmation bias |
| **C. Content models** |||||
| C1 | **Listwise group reranker** | built | +0.003 to +0.010 | 1–1.5 h | new model |
| C2 | Bigger listwise backbones (e5-base, bge-m3, xlm-r-large) | config | +0.001 to +0.004 over C1 | 1–4 h each | GPU memory (use grad-ckpt) |
| C3 | Listwise: 2 epochs, share 0.6, K 24, max_len 768 | config | +0.000 to +0.002 | +20–60 min | smaller holdout |
| C4 | Listwise test-time augmentation (2–3 orders) | recipe (few lines) | +0.0005 to +0.001 | ×2–3 scoring | none |
| C5 | Listwise with probability hint tokens | recipe | +0.001 to +0.003 | as C1 | loses orphan immunity |
| C6 | Listwise with competitor text per candidate | recipe | +0.001 to +0.004 | as C1 ×1.5 | longer sequences |
| C7 | **Pool-side listwise** (which S1 owns this record, or none) | recipe | +0.002 to +0.006 | 2–3 h | new code, about half a day |
| C8 | Listwise hard-negative round 2 | recipe | +0.000 to +0.002 | +1 h | none |
| C9 | Bigger pairwise cross-encoder (e5-base, bge-reranker-v2-m3) | lead (J) | +0.001 to +0.003 | 1–3 h | none |
| C10 | Multi-seed cross-encoder ensemble | recipe | +0.000 to +0.002 | 30–60 min per seed | none |
| C11 | Name-only and address-only cross-encoders | recipe | +0.001 to +0.003 | 1–2 h | none |
| C12 | Character-level models (CANINE-s, ByT5-small) | recipe | +0.000 to +0.003 (India) | 2–4 h | slower inference |
| C13 | Noise augmentation for cross-encoder/listwise training | recipe | +0.000 to +0.002 | +30 min | unrealistic noise |
| **D. Embeddings and blocking** |||||
| D1 | ANCE round-2 bi-encoder | lead (H) | +0.001 to +0.004 | 1–1.5 h | re-run features |
| D2 | Bigger bi-encoder (e5-base, bge-m3) | recipe | +0.001 to +0.004 | 2–4 h | re-run features |
| D3 | Separate name-only / address-only embeddings as features | recipe | +0.001 to +0.003 | 1 h | new feature group |
| D4 | Wider blocking (`k_emb` 20, `k_emb_reverse` 3) | config | ≤ +0.001 | more pairs | slower runs |
| **E. Features** |||||
| E1 | Rival-S1 context for every scorer | built (on) | +0.001 to +0.003 | none | none |
| E2 | Consensus features (majority number/street/name across candidates) | lead (F) | +0.001 to +0.004 | 2–3 h code | overlaps C1 |
| E3 | Address component parsing + per-component comparison | recipe | +0.001 to +0.003 | half a day | parser coverage |
| E4 | Better Indic transliteration | recipe | +0.000 to +0.002 (India) | 2–4 h | licence check |
| E5 | Pool-pool sibling features | recipe | ± | 1 h GPU kNN + 2 h code | decoys cluster with copies |
| E6 | Correlation-clustering features | recipe | ± | half a day | same as E5 |
| **F. Stage 2 and ensembling** |||||
| F1 | Stage-2 combination search | built | +0.001 to +0.003 | 1–3 h CPU | none |
| F2 | Monotone constraints | built | robustness | none | none |
| F3 | Seed bagging, bigger stage-1 sample | lead (B) | +0.0005 to +0.002 | CPU | time |
| F4 | Regime experts | lead (G) | +0.001 to +0.002 | CPU | thin regimes |
| F5 | Stage 2 trained on p1 ≥ 0.001 only | recipe (one line) | small | none | none |
| F6 | XGBoost + CatBoost + LightGBM blend | recipe | +0.0005 to +0.002 | 1–2 h CPU/GPU | deps + fair-play allowlist |
| F7 | Neural set model over numeric features | recipe | +0.001 to +0.003 | 2–3 h | tuning |
| F8 | Stage 3 stacking | recipe | +0.000 to +0.001 | 1 h CPU | overfitting |
| F9 | **GNN over the candidate graph** | recipe | +0.002 to +0.006 | 1 day of work + 2–4 h GPU | build time |
| **G. Decision** |||||
| G1 | Count-aware expected-F decision | lead (D) | +0.001 to +0.003 | offline | calibration |
| G2 | Isotonic calibration before the decision | recipe | +0.000 to +0.002 | 30 min | none |
| G3 | F-aware reassignment of contested pool records | recipe | +0.000 to +0.002 | 2 h | small effect |
| G4 | Sibling-consistent selection | lead (E) | +0.000 to +0.003 | needs E5 | decoy siblings |
| **H. Large models** |||||
| H1 | **LLM judge**: Qwen2.5-7B-Instruct + LoRA on uncertain entities | recipe | +0.002 to +0.008 (unknown) | 8–12 h GPU + 1 day of work | time, integration |
| H2 | LLM listwise generator (outputs the matching set) | recipe | as H1 | as H1 | parsing outputs |
| H3 | Other compliant LLMs (Mistral-7B, Phi-4-mini, Qwen2.5-3B) | recipe | as H1 | less for smaller | none |
| H4 | LLM-generated transliteration/abbreviation variants for augmentation | recipe | +0.000 to +0.003 | 3–5 h | check the augmentation rule |
| **I. Touch the rules (listed for completeness; read first)** |||||
| I1 | Using a leak in ids/row order | check rules | could be large | — | **disqualification risk** |
| I2 | Models > 8B or not MIT/Apache (Qwen3-8B, Llama, Gemma) | breaks rule | — | — | **disqualification** |
| I3 | External data (geocoding, registries, web) | breaks rule | — | — | **disqualification** |

---

## 3. Recipes

The same interface everywhere: a new model is a *scorer*. For a pairs file
(`rows, cols, p1`), it writes one float16 score per pair, plus optionally `<out>_grp.npy`.
Point `scale.ce_model` at it and stage 2 gets `<prefix>`, `<prefix>_rank/gap/margin_s1`,
`<prefix>_rank/gap/margin_pool` and `<prefix>_rest_s1`.

- Listwise models: `"kind": "listwise"` in `crossenc.json`.
- A new scorer module (LLM, char model): one line in `scale.cross_scores` (the kind→module
  map) plus one line in `scale._SCORER` (its prefix), and add its imports to
  `tests/test_fair_play.py` EXCEPTIONS.

### A. Diagnose

**A1 Shift diagnostics [built].**
`python scripts/shift_diagnostics.py --data-dir DATA --emb-dir artifacts/emb --run runs/<v3>`
- **Orphan-like** (low best-suitor similarity, S1 crowd flat) → B1/B2.
- **Crowd rises** → extra near decoys → B5.
- **One country stands out** → B6–B8.

**A2 LB probe, France.** Resubmit the best file with France rows emptied (keep the rows,
empty the lists). France mean F0.5 ≈ (S − S_noFR) × N/N_FR + 0.054 (France singleton
rate). N/N_FR ≈ 6.7, so LB rounding gives about ±0.003.

**A3 More probes.**
- Same trick per country (US, India).
- A threshold sweep (the same probabilities at t−0.05, t, t+0.05) reads test's best
  threshold directly.
- Limit: the public LB is a subset, so do not chase ±0.001.

**A4 Error audit.** From `runs/<run>/errors_false_positive.tsv` and
`errors_false_negative.tsv`, tag each row: number differs / one token differs / rename /
script / empty address / other. Count per country. Whatever type dominates tells you which
of C/E/G to push.

**A5 Leak audit.**
- Spearman correlation between the numeric part of each S1 id and its matched S2/S3 ids.
- Same for the row positions in the files.
- If either shows structure, read I1 before doing anything with it.

### B. Train→test shift

**B1 Orphan simulation [lead].** `--set scale.drop_s1_share=0.186` for run A.
Listwise groups built under the drop match test better too.

**B2 Mixture of drop shares.**
- Run stage 1 with drop 0.186; save stage-2 inputs from runs at drop 0, 0.1, 0.186, 0.25.
- Train stage 2 on their union, deduplicated by entity so no entity appears twice per
  fold.
- The model learns to be right across shift sizes.
- Simpler version: average the stage-2 predictions of drop-0.1 and drop-0.25 models.

**B3 EM prior adjust [lead].** `--set decision.prior_adjust=true`. Check it on a
drop-simulated holdout (simulate_decoys.py) before trusting it.

**B4 Per-country decision.**
- Tune the threshold per country on the holdout (US, India).
- For France, use the rule tuned on all countries, shifted by France's EM-estimated prior
  (B3 per country).
- Ship only if A3 agrees.

**B5 Inject neighbour decoys into train** (if A1 says "extra near decoys").
- For a random 25% of train S1, create 1–2 pool records: same name, house number ±1–30,
  the same noise functions as `synth.py` (typo, abbreviation, reorder).
- Append them to the train pool (label: no match); re-block and re-train.
- Keep test untouched.

**B6 Leave-one-country-out [lead C].**
- `scripts/learning_curve.py --run RUN --features F --loco`.
- Features whose gain collapses from US to India are likely not to transfer to France;
  drop them or monotone-constrain them.

**B7 Self-training on France.**
- Take French test pairs with final p ≥ 0.98 (positives) and p ≤ 0.02 among each
  entity's top-5 (negatives).
- Fine-tune the listwise model or cross-encoder 0.3 epoch on them mixed 1:3 with train
  groups.
- Keep only if the US/India holdout does not drop and A2 improves.

**B8 Self-supervised French adaptation.**
- Take each French test S1 record and create a positive by applying noise rules: accents
  removed, `rue`→`r.`, `avenue`→`av.`, legal forms SARL/SAS dropped or moved, token swap,
  one typo.
- Use other French records in the same postcode as negatives.
- Contrastive fine-tune the bi-encoder a little (0.1 epoch).
- No labels are used, but it does use test inputs: confirm the rules allow unsupervised
  use of test text.

**B9 Transductive pseudo-labels for stage 2.**
- Add test pairs with p2 ≥ 0.99 / ≤ 0.01 as extra stage-2 training rows (weight 0.3).
- Retrain stage 2 once.
- Measure on the train holdout; ship only if not worse.

### C. Content models

**C1 Listwise group reranker [built].** `src/ber/listwise.py`, `bash scripts/gpu_runbook.sh lw`
and `lweval`.
- **Input:** `[CLS] S1 [SEP] c1 … c16 [SEP]`, candidates with p1 ≥ 0.02, random order.
- **Heads:** pair head plus group head.
- **Data:** trained on the listwise share (`scale.lw_share=0.5`), which the holdout
  leaves out.
- **Stage-2 columns:** `lw*`, `lw_grp`.

**C2 Bigger listwise backbones.**
- Fetch the backbone: `python -m ber.cli fetch-model intfloat/multilingual-e5-base --dest models/multilingual-e5-base`
  (bge-m3 and xlm-roberta-large: add them to `ber/pretrained.py` REGISTRY first).
- Train: `LW_INIT=models/multilingual-e5-base LW_MODEL=models/lw-base bash scripts/gpu_runbook.sh lw`.
  Large models: `GRAD_CKPT=1 BATCH=16`.
- Stage 2 gets `lw2`: `MODELS=models/ce-small-v2,models/lw-small,models/lw-base`.
- Or init from a pairwise CE trained on that backbone (C9) for a warm start.

**C3 Listwise knobs.**
- `EPOCHS=2`; `SHARE=0.6`.
- `--k 24 --max-len 768` (bge-m3 only; XLM-R/e5 stop at 512).

**C4 Test-time augmentation.** In `listwise.main` score branch: call `score(..., seed=s)` for
s in 0, 1, 2 and average `pp` and `gp`. Candidate order differs, content does not.

**C5 Hint tokens.**
- Prefix each candidate with a bucket token of its p1 (and CE score): `[p9]` for
  0.9–1.0 and so on, added to the tokenizer as new special tokens.
- The model reasons over content and scores jointly.
- Loses orphan immunity (p1 carries pool-side competition), so keep C1 as well; stage 2
  uses both.

**C6 Competitor text.**
- After each candidate, append `⟂ <its best other S1 suitor, name | address>` (by p1),
  truncated to 16 tokens.
- The model sees who else wants the record.
- Train under drop simulation (B1) so an absent owner is common.
- About 1.5× sequence length: `--k 12`.

**C7 Pool-side listwise** (the mirror of C1).
- **Groups:** one pool record + its S1 suitors with p1 ≥ 0.02 (≤ 8): `[CLS] pool [SEP] s1_a [SEP] s1_b …`.
- **Labels:** which suitor owns it; group label "some suitor owns it".
- **Training:** pool records of the listwise-share entities, with the B1 drop so "no owner
  here" is well represented.
- **Code:** copy `listwise.py`; swap the roles in `build_groups` (group by `cols`), in
  `_stores` (the query is a pool text) and in `score` (key by pool record). It adds
  `lwp*` columns, including `lwp_rank_s1`, which exclusivity needs.
- **Size:** about 7M pool records → 20–40 min scoring (small backbone).

**C8 Hard-negative round 2.** Score the listwise share with the trained C1. Oversample
groups whose loss is in the top 20% (×3) in a second 0.5-epoch pass.

**C9 Bigger pairwise cross-encoder [lead J].**
- `python -m ber.crossenc train --init models/multilingual-e5-base --blocking ... --model models/ce-base`.
- Or `BAAI/bge-reranker-v2-m3` (Apache-2.0, native head supported); register it first.

**C10 Multi-seed cross-encoder ensemble.** Train `ce-small` with seeds 1–3 (different
negatives sample). Put all of them in `scale.ce_model`; stage 2 weighs them. Averaging them
into one scorer saves stage-2 columns.

**C11 Field-specific cross-encoders.**
- `name_only`: text = name; `addr_only`: text = address. Change `crossenc._text` via a
  `--field` flag.
- Each learns its field's noise without the other field dominating.
- Two extra columns families: `ce2`, `ce3`.

**C12 Character-level models.**
- `google/canine-s` (Apache-2.0, 132M) or `google/byt5-small` (Apache-2.0, 300M) as a
  pairwise CE.
- Robust to typos, digit-for-letter swaps and transliteration.
- Slower (4× tokens): score only pairs with 0.05 ≤ p1 ≤ 0.95.

**C13 Noise augmentation.**
- When training the CE or listwise model, with 20% probability apply one noise op to the
  positive's text: drop the legal form, abbreviate street type, swap two tokens, one
  character edit, zero-pad the house number.
- These are the noise types the handbook measured.

### D. Embeddings and blocking

**D1 ANCE round 2 [lead H].**
`python -m ber.embed finetune --base models/e5-small-ft --model models/e5-small-ft2 --knn-negatives artifacts/emb/train_knn.npz`,
then `encode` into a new `--emb-dir`. Features re-run for `cos_emb`.

**D2 Bigger bi-encoder.** Same with `--base models/multilingual-e5-base`, `--batch 256`.
Encoding about 24M texts takes about 40 min on an H100.

**D3 Name-only / address-only embeddings.**
- Encode names and addresses separately (a `--field` flag in `embed.record_texts`).
- Add `cos_emb_name`, `cos_emb_addr` and their group context as a new `FEATURE_GROUPS`
  entry (reuses cached columns; see the handbook on feature versions).

**D4 Wider blocking.** `--set scale.k_emb=20 --set scale.k_emb_reverse=3`. Check the
oracle F0.5 moves before paying for the extra pairs.

### E. Features

**E1 Rival-S1 context [built, on by default].** `ce_rank_pool`, `ce_gap_pool`,
`ce_margin_pool`, `ce_rest_s1`, and the same for `lw`.

**E2 Consensus features [lead F].**
- Across an entity's candidates with p1 ≥ 0.3: the majority first house number, majority
  street key and majority name key.
- Per pair: does it agree with the majority; does S1 agree; how many candidates support
  the majority.

**E3 Address components.**
- Rule-based split per country pattern into house number / unit, street, locality,
  city, state, postcode (`normalize/address.py` already extracts some).
- Per component: equal, missing, token-set ratio.
- A missing component is not a conflict.

**E4 Indic transliteration.**
- Romanise Indic-script names with a rule-based ITRANS/IAST transliterator (check the
  package licence is MIT/Apache).
- Add `name_translit_ratio` against the S1 name.
- Or train a tiny char seq2seq on train pairs (script name ↔ Latin S1 name).

**E5 Pool-pool siblings.**
- `ber.embed.knn(pool, pool, country, country, k=10)` on GPU (about 10 min for 10M on an
  H100).
- Siblings = neighbours with similarity ≥ τ (τ: median same-business similarity on train).
- Per pair: number of siblings; share of siblings whose best suitor is this S1; max p1
  from this S1 to siblings.
- Caveat: a neighbour decoy is a near-duplicate too, so the search may reject these.

**E6 Correlation clustering.**
- Graph over pool records, with edges = sibling similarity combined with CE/listwise
  agreement.
- Cluster (connected components at a high threshold, then split on house-number
  conflict).
- Features: cluster size; share of the cluster that this S1 already claims.

### F. Stage 2 and ensembling

**F1 Combination search [built].** `scale.save_stage2=true` in the run, then
`python scripts/stage2_search.py --run RUN --monotone --bags 3 --max-sample 250000`.
Ship with the printed `--set scale.stage2_exclude=...`.

**F2 Monotone [built].** `--set scale.stage2_monotone=true`. Take it if it is within 0.0003
of the best: it is safer under shift.

**F3 Bagging and sample [lead B].** `SAMPLE=800000`; add seeds per fold (repeat `fit_folds`
with `cfg.seed + 100 × bag`, average).

**F4 Regime experts [lead G].** Regimes: (name ≈ same, number differs), (name differs,
address same), rest. Train stage 2 per regime (or add a regime id as a categorical and
let the search decide).

**F5 Stage 2 on p1 ≥ 0.001.** In `scale.run`: `sidx = sidx[p1[sidx] >= 0.001]` before
fitting stage 2. Faster, and the model focuses on the region that matters.

**F6 Other GBMs.**
- `xgboost` (Apache-2.0, `tree_method="hist"`, GPU ok) and `catboost` (Apache-2.0) on the
  same stage-2 matrix and folds (`stage2_inputs.npy`).
- Average logits with LightGBM and tune the weights on sample OOF.
- Add both to `requirements.txt` pins and to `THIRD_PARTY` in `tests/test_fair_play.py`.

**F7 Neural set model.**
- Per entity, take its candidates' stage-2 feature vectors (≤ 32, standardised).
- A 2-layer transformer over the set (no text) → per-candidate logit plus a group logit.
- Train on sample OOF; add as a column family or blend.
- Tiny and fast on GPU. It is C1's idea applied to numbers.

**F8 Stage 3.** `stage2_columns` again on p2 (rank/gap/margin both sides) plus the top
columns; one more LightGBM with the same folds. Stop if the holdout gain is < 0.0003.

**F9 GNN over the candidate graph.**
- **Nodes:** S1 + pool records (init: e5 embeddings, 384-d).
- **Edges:** candidate pairs with p1 ≥ 0.001 (features: the stage-2 vector), plus
  pool-pool kNN edges (E5).
- **Model:** 2–3 message-passing layers (GraphSAGE/GAT; PyTorch Geometric is MIT, or plain
  torch scatter), then an edge MLP → match logit.
- **Training:** edges of sample entities (entity-grouped folds); holdout as usual; output
  column `gnn`.
- **Size:** about 15M edges × 64-d fits in 80 GB. This does the collective reasoning
  (exclusivity, siblings, consensus) jointly.

### G. Decision

**G1 Count-aware expected-F [lead D].**
- Calibrate p (G2).
- Per entity, choose the top-k by p that maximises E[F0.5]:
  (1+β²)·Σp_top-k / (k + β²·E|true|).
- E|true| comes from `lw_rest_s1 + p` or a small regressor.
- Choose "none" when P(no match) from `lw_grp` beats it.

**G2 Isotonic calibration.** Fit `sklearn.isotonic.IsotonicRegression` on sample OOF
(p2 → label). Apply to all, then run `tune` as usual.

**G3 F-aware reassignment.**
- For pool records whose second suitor is within 0.1 of the first, compute each
  entity's expected-F with and without the record.
- Give it to whichever entity gains more, or to neither.
- Greedy, one pass.

**G4 Sibling-consistent selection [lead E].** With E5 siblings, a sibling set goes to one
S1 or none: majority vote of the siblings' decided owners, overriding only when ≥ 2/3
agree.

### H. Large models

**H1 LLM judge (Qwen2.5-7B-Instruct, Apache-2.0, 7.6B).**
- **Register:** add `Qwen/Qwen2.5-7B-Instruct` to `ber/pretrained.py` (Apache-2.0,
  7.6e9). Deps: `peft` (Apache-2.0) and `vllm` (Apache-2.0) in a separate env; add them to
  the fair-play exceptions for the new `ber/llm.py`.
- **Which entities:** those with any pair where 0.1 ≤ p2 ≤ 0.9 (expect 5–10% of
  entities).
- **Prompt** (group context first, so vLLM's prefix cache shares it across candidates):
  `Record A: <S1>. Candidates: 1) … 2) … n) …. Is candidate k the same business as A? Answer yes or no.`
- **Train:** LoRA r=16, α=32, lr 1e-4, 1 epoch on about 100k listwise-share entities
  (one example per candidate). About 2–3 h on an H100 in bf16.
- **Score:** vLLM with `max_tokens=1` and `logprobs`: P(yes) = softmax over the yes/no
  logits.
- **Integrate:** write `llm_scores.npy` per pair, NaN outside the uncertain set, as a
  scorer with prefix `llm`.
- **Cost:** about 8–12 h in all. Start it on a second GPU in parallel with track 1.

**H2 LLM listwise generator.** Same model and data. The target is the matching candidate
numbers (`2,5` or `none`). Parse the output into 0/1 per candidate and use the tokens'
logprob as confidence. One generation per entity: cheaper than H1, but coarser
probabilities.

**H3 Other compliant LLMs.**
- `mistralai/Mistral-7B-Instruct-v0.3` (Apache-2.0, 7.25B).
- `microsoft/Phi-4-mini-instruct` (MIT, 3.8B).
- `Qwen/Qwen2.5-3B-Instruct`: check its licence (Qwen2.5-3B uses the Qwen Research
  licence, not Apache).
- Smaller means 2–3× faster at some quality cost.

**H4 LLM-generated augmentation.**
- Prompt a compliant LLM to produce transliteration/abbreviation variants of *train*
  names (for example Indic-script ↔ Latin, "Pvt Ltd" forms).
- Use them as extra positives for the CE/listwise model.
- Uses no outside facts about the businesses, but confirm with the rules that
  "internet data augmentation is prohibited" does not cover model-generated variants.

### I. Ideas that touch the rules (read before using)

- **I1 Leaks.** If A5 finds that ids or row order encode the matches, using them is not
  "external data", but it goes against the challenge's intent. Top packages are reviewed
  in detail, and the methodology document must explain everything the code uses.
  **Ask the organisers before relying on it.**
- **I2 Over-limit or wrongly licensed models.** Qwen3-8B (8.2B > 8B), Llama (Llama
  licence), Gemma (Gemma terms): each disqualifies.
  - Compliant substitutes: Qwen2.5-7B-Instruct (Apache-2.0, 7.6B), Mistral-7B-v0.3
    (Apache-2.0), Phi-4-mini (MIT), bge-m3 / e5 (MIT).
- **I3 External data.** Geocoding, business registries, web lookups, and any external
  entity-resolution API: explicitly disqualifying. The substitute is everything above,
  which uses only the provided data and licensed model weights.

---

## 4. Running everything: parallel tracks

Not all of this fits into one GPU and 52 hours, but most of it does across three or four
machines. Everything feeds stage 2, and F1 decides what ships.

| Track | Machine | Ideas, in order |
|---|---|---|
| **1: core (ship first)** | GPU #1 + CPU | A1 → B1 (if A1 agrees) → C1 → E1 → F1/F2 → first submission, then C4 |
| **2: big content models** | GPU #2 (H100) | C2 (bge-m3 listwise) → C7 (pool-side listwise) → H1 (LLM judge) |
| **3: decision and stacking** | CPU (any big box) | G2 → G1 → F5 → F6 → F7 → F8 → G3 → B4 |
| **4: representations** | GPU #3 | D1 → D3 → C11 → C12 → E5/E6 → F9 (GNN) → G4 |
| **5: shift experiments** | CPU + small GPU | B2 → B5 (if decoys) → B6 → B7/B8 (if A2 says France) → B9 |
| any time | — | A2–A5 |

Suggested timeline (IST):

| When | Track 1 | Others |
|---|---|---|
| 25 Sep night | A1; runA (or fast lane); ce | track 2 starts C2; track 4 starts D1 |
| 26 Sep morning | lw, lweval, runB | track 3 on runB's stage-2 inputs |
| 26 Sep afternoon | F1 search → **submit** | track 2 C7; track 5 B2 |
| 26 Sep night | fold in C2, D1, C7 → re-run F1 | track 2 H1 training; track 4 F9 |
| 27 Sep until 15:00 | fold in H1, F9, F6/F7 if ready → F1 → **submit best** | — |
| 27 Sep 15:00 → deadline | freeze, validate, package, README + methodology | no new ideas |

Gates:
1. **Holdout F0.5 decides** (handbook rule); submit only what beats the best so far.
2. **Every new column family must earn its place in F1.** If it doesn't, exclude it.
3. **Prefer monotone / simpler** when within 0.0003.
4. **Freeze by 27 Sep afternoon.**

## 5. Risks and guards

| Risk | Guard |
|---|---|
| Leakage from a new model into the holdout | train new models only on the listwise share or the encoder share; the holdout excludes the listwise share (tested) |
| Share entities compete with holdout entities using in-sample scores | small; trust differences above about 0.0005 |
| Tuning to the wrong shift story | A1 before B1/B5; A2 if in doubt |
| GPU out of memory | grad checkpointing, smaller batch; listwise scoring splits OOM batches |
| fp16-only GPUs | GradScaler; `BER_LISTWISE_AMP=bf16/fp16/off` |
| Too many columns overfit stage 2 | F1 search + monotone + bags |
| The package must regenerate outputs | every trained model is code (`ber.listwise train`, `ber.crossenc train`, …); document commands in README and methodology |
| New dependencies | pin in requirements; add to the fair-play allowlist; licences MIT/Apache only |
| Deadline | runA fallback; freeze 27 Sep afternoon |

## 6. Files on branch `listwise-reranker`

| File | What |
|---|---|
| `src/ber/listwise.py` | C1 model: train / score / eval |
| `src/ber/scale.py` | listwise share, scorer dispatch + score cache, E1 columns, `stage2_exclude`, `stage2_monotone`, `save_stage2` |
| `scripts/shift_diagnostics.py` | A1 |
| `scripts/stage2_search.py` | F1 / F2 / bags |
| `scripts/dump_lw_groups.py` | listwise training data from any finished run (for example v3) |
| `scripts/gpu_runbook.sh` | track 1, step by step |
| `docs/v4_listwise.md` | design, gates, risks |
| `tests/test_listwise.py` | tests for all of the above (all 200 tests pass) |
