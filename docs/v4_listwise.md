# v4: listwise group reranker, pool-side scorer context, and a stage-2 combination search

Branch `listwise-reranker`. Everything here is additive and off by default: with the new
settings at their defaults, `ber run` behaves as on `main`, except that each cross-encoder
now also gets pool-side context columns in stage 2 (`ce_rank_pool`, `ce_gap_pool`,
`ce_margin_pool`, `ce_rest_s1`). Tests: 200 pass (`pytest -q`), ruff clean.

## 1. Where the points are

| Fact (handbook) | What it implies |
|---|---|
| v2 blocking oracle F0.5 0.9982 | blocking is not the gap; the matcher and the decision are |
| v1: holdout 0.9645 vs public 0.947 | about 1.75 points are lost to the train-to-test shift, more than any model gain so far |
| test has 5.5-5.8 pool records per S1 against 4.68 in train | ~40% of the test pool matches nothing, against 26% in train |
| 79% of v2 stage-1 gain is the embedding cosine, mostly **pool-side** rank/gap/margin | the matcher's best signal is "is this S1 the pool record's best suitor?", and that is exactly what a missing owner (orphan) fools |
| residual errors: same name at a neighbouring number; one token apart at the same address; renames | **group-level** confusions: the evidence is in the other candidates (three copies say 450, one says 471) |
| entities with exactly one true match: F0.5 0.866 | count-awareness is missing |

Every model in the stack today reads one candidate at a time. None reads the candidates
of an entity together, and none produces a signal that is immune to a missing owner.

## 2. Options considered

The gains are estimates on the public score, from the numbers above, not measurements.
They are there to rank options, and the gates in section 6 replace them with holdout numbers.

| # | Option | Targets | Est. gain | GPU cost | Risk | Verdict |
|---|---|---|---|---|---|---|
| 1 | **Listwise group reranker** (`ber.listwise`, new) | neighbour/namesake/rename confusions, 1-match entities, orphan robustness | +0.003 to +0.010 | 20-40 min train, 20-40 min score (small) | medium: new model | **do** |
| 2 | **Pool-side context for every scorer** in stage 2 (new) | exclusivity decisions (who owns this pool record) | +0.001 to +0.003 | none | low | **do** (on by default) |
| 3 | Orphan simulation `drop_s1_share` (lead, direction I) | train-to-test shift | +0.005 to +0.015 if the shift is orphans | none | low, if the diagnosis is right | **do**, after the shift check (#5) |
| 4 | **Stage-2 combination search** + monotone + seed bags (new) | picking the right stack; robustness under shift | +0.001 to +0.003 | CPU 1-3 h | low | **do** |
| 5 | **Shift diagnostics** (new, numpy only) | deciding between orphans / extra decoys / France | enables #3 | none | none | **do first** |
| 6 | Bigger pairwise CE (e5-base, bge-reranker-v2-m3) | pairwise errors | +0.001 to +0.003 | 1-3 h | low | lead (J) |
| 7 | Bigger listwise backbone (multilingual-e5-base 278M, bge-m3 568M, both MIT) | as #1 | +0.001 to +0.004 over #1 | 1-4 h | medium | if #1 passes its gate and time remains |
| 8 | Decoder LLM judge (Qwen2.5-7B-Instruct, Apache-2.0; LoRA) on uncertain entities | hardest cases | unknown | 8-12 h train + infer | high: time, integration | **no** for this deadline; #7 gets most of it for a tenth of the cost |
| 9 | Pool-to-pool sibling clustering features (direction E) | orphan clusters, consistency | ambiguous: a neighbour decoy embeds as close as a true copy, so siblings pull decoys in too | 1 h kNN | medium | **no**; #1 does the consensus on exact text |
| 10 | Self-training on confident test pairs (France) | unseen country | unknown sign | 2-4 h | high: confirmation bias | only if France is shown to be the loss |
| 11 | Leaderboard probe: same file with France emptied | where the public loss is | diagnostic | 1 submission | low | optional; F(France) ≈ (S − S₋F)·N/N_F + singleton rate |
| 12 | Seed bagging / bigger stage-1 sample (lead, direction B) | variance | +0.0005 to +0.002 | CPU | low | lead |

A perfect 1.0 is not reachable. Blocking alone caps train at 0.998, and some copies are
unlinkable by design (renames with no alias, addresses cut to a city). The realistic target
is 0.985 to 0.99 public.

## 3. The listwise group reranker (`src/ber/listwise.py`)

**Input.** For each Source 1 entity, its candidates with stage-1 probability ≥ `ce_min_p1`
(the pairs the cross-encoder already scores), up to 16 per group in random order:

```
[CLS] S1 name | address [SEP] cand 1 [SEP] cand 2 [SEP] ... cand 16 [SEP]      (≤ 512 tokens)
```

An entity with more candidates gets overflow groups: its 8 likeliest candidates as anchors
plus the next 8, so every pair is scored next to the entity's best copies. Every pair is
scored exactly once.

**Model.** Any Hugging Face encoder (default init: the lead's `ce-small-v2`, which already
knows pairwise matching). Each candidate's span is mean-pooled; a pair head reads
`[c, q, c·q, |c−q|]` against the pooled S1 span; a group head on `[CLS]` and the S1 span
predicts "some candidate here matches". Loss: BCE per candidate + 0.3 × group BCE.

**Content only.** No probability or pool-side competition goes in, so an orphan's copies
look the same whether or not their owner is in Source 1. The model has to reject them on
what they say (another house number, another name). That is the signal stage 2 lacks
when the best-suitor features are fooled.

**Data and leakage.**
- A new *listwise share* (`scale.lw_share=0.5`) of eligible entities outside the stage-1/2
  sample trains it. It is picked by a fixed hash, so every run agrees, and the stage-1/2
  sample is unchanged, so the lead's stage-1 models stay reusable (tested).
- The share is removed from the holdout, from `train_full` and from `eligible` in
  `train_scores.npz`. The holdout stays out-of-sample for every model.
- Training groups come from stage-1 probabilities that are out-of-sample for those
  entities, as on test. The encoder-share alternative would give optimistic, easier groups.
- Each training entity draws its own p1 floor (0.002 to 0.05), so the model sees small and
  large, clean and contaminated groups. Test's orphan-inflated groups are inside that range.

**Outputs to stage 2.**
- `lw`: the pair probability.
- `lw_rank_s1`, `lw_gap_s1`, `lw_margin_s1`, `lw_rest_s1`: its context among the entity's
  candidates. `lw_rest_s1` is the expected number of other matches.
- `lw_rank_pool`, `lw_gap_pool`, `lw_margin_pool`: its context among the pool record's suitors.
- `lw_grp`: P(the entity has any match), for singleton decisions.

Several listwise models become `lw`, `lw2`, and so on, next to `ce`, `ce2`.

## 4. Other changes

| Change | Where | Why |
|---|---|---|
| pool-side context and rest-mass for every scorer | `scale.ce_columns` | stage 2 saw CE ranks only on the S1 side, while pool-side ranks carry most of stage 1's gain |
| scorer dispatch by model kind; p1 passed to scorers | `scale.cross_scores` | a listwise model groups pairs by p1 |
| score cache keyed by model + exact pairs | `artifacts/cache/scores/` | reruns (search → final) do not rescore ~10M pairs |
| `scale.stage2_exclude` | `scale._stage2_names` | ship the combination the search picks |
| `scale.stage2_monotone` | `scale.stage2_monotone` | stage 2 may only rise with each model's probability; guards against interactions that flip under shift |
| `scale.save_stage2` | `scale.save_stage2_inputs` | the search retrains stage 2 offline in minutes |
| `scripts/stage2_search.py` | new | every family subset (≤5 optional, else greedy), exclusive + tuned decision, exact holdout F0.5; `--monotone`, `--bags` |
| `scripts/shift_diagnostics.py` | new | orphans or extra decoys or country effect, from kNN lists (+ a run's scores) |
| `scripts/dump_lw_groups.py` | new | listwise training groups from any finished run (for example v3), no rerun; reproduces the in-run dump exactly (tested) |
| `scripts/gpu_runbook.sh` | new | the whole v4 on a Linux GPU box, step by step |

## 5. Two ways to run it

**A. Own GPU box** (independent of the lead's laptop). Use 1× H100/A100 80 GB, 32+ vCPU,
128+ GB RAM, 300 GB disk: roughly $2.5-3.5/h, 12-20 h in all. Follow
`scripts/gpu_runbook.sh`: `setup → embed → shift → runA → ce → lw → lweval → runB → search → final`.
Wall time is about 10-12 h cold. `ce` can start while `runA` is still computing features,
because it only needs the train blocking cache. `runA` alone gives a v2-level fallback
submission.

**B. Fast lane with the lead's caches** (fewest hours; a free Colab or Kaggle T4 is enough
for the small model).

1. Lead: `python scripts/dump_lw_groups.py --run runs/<v3-run> --data-dir DATA --share 0.5`
   (CPU, reuses v3's caches). Share `lw_groups_train.npz` and `models/ce-small-v2/`.
2. GPU: `python -m ber.listwise train --data-dir DATA --groups lw_groups_train.npz --init models/ce-small-v2 --model models/lw-small`,
   then `python -m ber.listwise eval ... --ce models/ce-small-v2`. Only the train TSVs are needed.
3. Share `models/lw-small/` back.
4. Lead: the v3 command plus `--set scale.stage1_model=runs/<v3-run>/scale_model.joblib --set scale.lw_share=0.5 --set scale.ce_model=models/ce-small-v2,models/lw-small --set scale.save_stage2=true`.
5. `python scripts/stage2_search.py --run runs/<that run> --monotone --bags 3`, then the final run
   with `--set scale.stage2_exclude=<printed>`.

On Linux, always pass `--set scale.ce_python=python` (the default `py -3.13` is the Windows launcher).

## 6. Gates: what to look at before going on

1. **`shift`**: if the test pool is orphan-like (`pool_best<0.85` and
   `unexplained_below_matched_p1` above train's, `s1_crowd` flat), rerun A with `DROP=0.186`.
   If `s1_crowd` rises instead, the extra records are near decoys: keep `DROP=0`, and let
   the listwise model and a stricter decision handle them.
2. **`lweval`** (60k holdout entities): the listwise pair AUC should be within 0.0002 of the
   CE's (≈0.9998), and its blend with stage 1 (`blend:...+stage1_p1`, groups from a run) or
   with the CE (`blend:...+ce:...`, groups from `make_lw_groups.py`) should beat the CE alone on the macro proxy. If it
   loses on both, stop spending on the listwise model and ship the CE stack with pool-side
   context.
3. **`runB`**: stage-2 holdout against run A's and against the lead's v3. Submit only what
   beats the best holdout so far (handbook rule).
4. **`search`**: take the top row; run `final` with its `EXCLUDE`. Also take the monotone
   row if it is within 0.0003 of the top, because it is the safer bet under shift.

## 7. What can go wrong, and the guard

| Risk | Guard |
|---|---|
| leakage from the listwise share into the holdout | share excluded from holdout, `train_full`, `eligible` (tested) |
| share entities still compete for pool records with holdout entities, with in-sample listwise scores | small, and the same kind as the encoder-share entities' in-sample CE scores today; compare runs on the same holdout, trust differences above ~0.0005 |
| the listwise model undertrained after 1 epoch | `lweval` shows it; rerun `lw` with `EPOCHS=2` (about 20 min more for the small model) |
| stage-1 reuse broken by the new share | the sample is drawn before the share and unchanged (tested) |
| training groups easier than test groups | out-of-sample p1 groups plus random p1 floors per entity |
| GPU out of memory on long Indic-script groups | per-candidate token budget shrinks to fit 512; OOM batches are split (scoring) or skipped (training) |
| float16-only GPUs (T4, P100) | GradScaler; bfloat16 used where supported |
| rescoring cost on every rerun | score cache keyed by model + pairs (tested) |
| the listwise model is worse than the CE | stage 2 weighs it; `stage2_search` drops it if it does not help; `stage2_exclude` ships without it |
| France unseen | multilingual backbone, content-only input; the shift report gives per-country numbers for test |
| deadline (28 Sep 00:00 IST) | run A's output is a fallback; freeze a submission by the evening of 27 Sep |
| the package must reproduce outputs | listwise training is code (`python -m ber.listwise train`); add its command to the README and methodology when it ships |
| licence / size | every backbone listed is MIT (e5-small/base, bge-m3) or Apache-2.0, all ≤ 568M parameters |
