# Branch `nikhil_exp`: plan, code and hand-off (for v6)

Built on `main` at `cb483ba` (26 Sep, 00:36 IST). Everything here is additive: with the new
settings at their defaults, `ber run` behaves exactly as on `main`. Tests: 212 pass
(`pytest -q`), `ruff check` and `ruff format --check` are clean. **Nothing here has run on
the real data yet**: the numbers that decide v6 come from the runs in section 3.

Owner of the runs: **Nambi Rajan** (section 3). Nikhil wrote the code; the lead folds
whatever wins into v6.

## 1. What is new against `main`

`main` already has, and this branch keeps unchanged: pool-side cross-encoder context
(`ce_pool_context`), `ce_band`, `ce_train_all`, the France self-trained cross-encoders,
orphan simulation (`drop_s1_share`), name-cluster features, `k_emb` 20. Where this
branch had its own version of any of them, main's was kept.

| New on this branch | What it is | Why (evidence) |
|---|---|---|
| **C1: listwise reranker** (`ber.listwise`) | one transformer pass over an S1 record and all its candidates (≤16): `[CLS] S1 [SEP] c1 [SEP] c2 …`; pair head + group head ("some candidate matches") | v3's remaining errors are group-level (a neighbour at #471 among copies that say #450; namesakes). Every model on main reads one candidate at a time. Lead's TODO: "the strongest idea we have not measured yet" |
| **C7: pool-side listwise** (`ber.listwise`, `side=pool`) | one pass over a pool record and its S1 suitors: which one owns it, or none. A fifth of training groups lose their owner (`drop_owner=0.2`), teaching "none of these" | 44% of v3's missed pairs lose their record to another suitor; address-less best candidates score 0.882 against 0.991 overall; test's orphans are records whose owner is missing |
| **Raw-data training groups** (`scripts/make_lw_groups.py`) | C1/C7 training groups straight from the TSVs on the **encoder share**, no pipeline run or cache needed | trains anywhere; the matcher's holdout is untouched, so with/without is directly comparable to v5 |
| **French région/département** (`normalize/address.py`) | a component that is exactly a French région or département name becomes the région's token; `5BIS` → `5 bis` | measured on test: French S1 writes only régions (Hauts-de-France 102k, Nouvelle-Aquitaine 85k, Pays de la Loire 73k); ~1/3 of the French pool writes the département instead (Nord 151k, Gironde 150k, …). US/India had their state names mapped; France had nothing, so a third of French pairs carried an address mismatch no train pair has. France is "most of the public gap" (handbook 3c) |
| Stage-2 combination search (`scripts/stage2_search.py`) + `scale.save_stage2` / `stage2_exclude` / `stage2_monotone` | retrain stage 2 offline on every family subset, exact holdout F0.5 | pick what ships from measurements, not guesses |
| Score cache (`artifacts/cache/scores/`) | a scorer's output is reused when the model and pairs are the same | reruns do not rescore ~10M pairs |
| Shift diagnostics (`scripts/shift_diagnostics.py`) | orphans vs extra decoys vs country, from kNN lists | explains test behaviour per country |

Data facts found tonight (train, measured):
- Copies inside one source share a per-source address variant, often with invented
  tokens: `HN 883` on all 4 S2 copies, `6513` (S1 says 6515) on both S3 copies, `Glendle`,
  `Tuullahoma`. A neighbour decoy nearby has the neighbour number and the *correct*
  spelling. But sharing an exact address inside a source is only 73% pure (buildings), so
  this is a feature idea (section 5), not a rule.
- Address-less records are 4.4% of true copies and 0.29% of decoys.
- No row-order or id leak (ids random, rows not grouped).

## 2. How the pieces plug in

- C1 and C7 are **scorers** like the cross-encoders: put their model directories in
  `scale.ce_model` after v5's two cross-encoders. Stage 2 gets:
  - C1: `lw`, `lw_rank/gap/margin_s1`, `lw_rest_s1` (expected number of other matches),
    `lw_grp` (P(entity has a match)), and with `ce_pool_context` `lw_*_pool`;
  - C7: `lwp`, `lwp_grp` (P(someone owns this record)) and the same context columns.
- `ce_band` applies only to pairwise models; listwise models always read whole groups.
- The French change touches **test only**: US/India normalise byte for byte as before
  (tested), so train caches stay valid. **Delete `artifacts/cache/test_*`** before the run
  that should use it, or test keeps the old tables.

## 3. Task for Nambi Rajan (due 26 Sep 14:00 IST, before v6)

Machine: any Linux/Windows box with a CUDA GPU ≥ 12 GB and ≥ 32 GB RAM (the lead's
laptop class). The group builders are CPU-only, about 6-10 GB of RAM.

```bash
git fetch origin && git switch nikhil_exp
pip install -r requirements-dev.txt && pip install -e . --no-deps   # + torch/transformers (requirements-gpu.txt)
export DATA=student_resource/dataset PY=python   # Windows: PY="py -3.13" as the lead
```

1. **Init model.** Ask the lead for `models/ce-small-v2` (the pairwise cross-encoder; best
   init). If not available: `python -m ber.cli fetch-model intfloat/multilingual-e5-small --dest models/multilingual-e5-small`
   and use `LW_INIT=models/multilingual-e5-small` (then train 2 epochs).
2. **Groups** (CPU, ~20-30 min each):
   `bash scripts/gpu_runbook.sh groups` and `bash scripts/gpu_runbook.sh pgroups`.
3. **Train C1** (~30-60 min on a 16 GB GPU, small backbone):
   `LW_INIT=models/ce-small-v2 bash scripts/gpu_runbook.sh lw`
   then `bash scripts/gpu_runbook.sh lweval`. Gate: `lw` AUC within 0.0002 of the CE's,
   and `blend:lw+p1` above `blend:ce+p1` on the macro proxy.
4. **Train C7**: `LW_INIT=models/ce-small-v2 bash scripts/gpu_runbook.sh lwp`
   then `bash scripts/gpu_runbook.sh lwpeval`. Gate: `lwp` owner accuracy above the CE's.
5. **Measure against v5** (on the lead's machine, which has v5's caches; or copy
   `models/lw-small` and `models/lwp-small` there): the v5 command plus
   ```
   --set scale.stage1_model=runs/<v5 run>/scale_model.joblib
   --set scale.ce_model=<v5's two cross-encoders>,models/lw-small,models/lwp-small
   --set scale.save_stage2=true --set scale.drop_s1_share=0.186
   ```
   then `python scripts/stage2_search.py --run runs/<that run> --only <v5 families>` and
   `--only <v5 families>,lw,lw_grp,lwp,lwp_grp,lw_pool,lwp_pool` (or `--mode greedy`).
   Report the **holdout macro F0.5 with and without** the listwise columns (same holdout
   as v5: the listwise models trained on the encoder share, `lw_share` stays 0).
6. **France A/B** (public LB only, no French labels): delete `artifacts/cache/test_*`,
   rerun test prediction with the same model (or run v6 twice); submit both **only after
   telling the lead**. Look at `scripts/shift_diagnostics.py --run` for France's
   uncertain-pairs-per-entity before and after.

Report in the handbook's section 2: holdout with/without C1, with/without C7, the chosen
`stage2_exclude`, and the public score of any upload.

## 4. Risks and guards

| Risk | Guard |
|---|---|
| leakage | C1/C7 train on the encoder share only (C7's suitors too); the matcher never trains or is judged there |
| raw-data groups differ from a run's p1 ≥ 0.02 groups | content-only models, random group sizes, hardest-first negatives (shared rare name/address tokens + house number); `lweval` measures on held-out groups |
| stale test caches after the French change | delete `artifacts/cache/test_*`; train caches are unaffected |
| GPU memory | `GRAD_CKPT=1 BATCH=16`; scoring splits out-of-memory batches |
| float16-only GPUs | GradScaler; `BER_LISTWISE_AMP=bf16/fp16/off` |
| time | C1 first (the lead's TODO), C7 second, French A/B last |

## 5. Next ideas (not built; for whoever has time)

- **Source-fingerprint features**: for pair (S1 A, pool p), whether p shares its rare
  address tokens (house-number variants, typos absent from A's own address) with A's
  confident copies *from the same source*, and whether p lacks a variant all of them share.
  Separates a typo'd true copy from a neighbour decoy.
- **B2**: train stage 2 on a mix of drop shares (0, 0.1, 0.186, 0.25) for robustness.
- **F9**: a GNN over the S1-pool candidate graph (joint exclusivity).
- **H1**: an Apache-2.0 7B LLM judge (Qwen2.5-7B-Instruct + LoRA) on uncertain entities only.
- Full list with recipes: `docs/ideas_worth_trying.md`.
